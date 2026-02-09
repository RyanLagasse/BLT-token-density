#!/usr/bin/env python3
"""
BLT Patch Granularity Experiment - Threshold Sweep

Loads the BLT entropy model (patcher) from the entropy_model/ subdirectory,
varies the entropy patching threshold, counts and prints byte patches.

The entropy model is a small byte-level causal LM (LLaMA-architecture).
Weights are in Meta .pth format with Meta key naming; we map them to HF
LlamaForCausalLM conventions and build a standalone model for inference.
"""

import json
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "The capital of France is Paris. "
    "What is 2 + 2? The answer is 4. "
    "Photosynthesis converts sunlight into chemical energy."
)

THRESHOLDS = [7.75, 7.775, 7.8, 7.825, 7.85, 7.9]

MODEL_DIRS = {
    "blt-1b": Path("blt-1b"),
    "blt-7b": Path("blt-7b"),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Meta -> HF weight key mapping
#
# Raw checkpoint keys have model. prefix and Meta naming:
#   model.layers.0.attention.wq.weight
#   model.tok_embeddings.weight
#   model.output.weight
#   model.norm.weight
# ---------------------------------------------------------------------------

_KEY_REPLACEMENTS = [
    (".attention.wq.", ".self_attn.q_proj."),
    (".attention.wk.", ".self_attn.k_proj."),
    (".attention.wv.", ".self_attn.v_proj."),
    (".attention.wo.", ".self_attn.o_proj."),
    (".feed_forward.w1.", ".mlp.gate_proj."),
    (".feed_forward.w2.", ".mlp.down_proj."),
    (".feed_forward.w3.", ".mlp.up_proj."),
    (".attention_norm.", ".input_layernorm."),
    (".ffn_norm.", ".post_attention_layernorm."),
    ("tok_embeddings.", "embed_tokens."),
]


def _map_meta_key(key: str) -> str:
    """Map a Meta-format weight key to HF LlamaForCausalLM naming."""
    for old, new in _KEY_REPLACEMENTS:
        key = key.replace(old, new)
    # model.output.weight -> lm_head.weight (top-level head, outside model.)
    if key in ("model.output.weight", "output.weight"):
        return "lm_head.weight"
    # Only add model. prefix if not already present
    if not key.startswith("model.") and not key.startswith("lm_head"):
        key = "model." + key
    return key

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_patching_config(model_dir: Path) -> dict:
    """Read patching-related fields from the main model config.json."""
    cfg = json.loads((model_dir / "config.json").read_text())
    return {
        "patching_mode": cfg.get("patching_mode", "entropy"),
        "patching_threshold": cfg.get("patching_threshold"),
        "max_patch_length": cfg.get("max_patch_length"),
    }


def load_entropy_model(model_dir: Path):
    """
    Load the entropy/patcher model from {model_dir}/entropy_model/.

    Reads params.json for architecture config, loads consolidated.pth weights
    with Meta->HF key mapping, builds a LlamaForCausalLM.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    entropy_dir = model_dir / "entropy_model"
    if not entropy_dir.exists():
        raise FileNotFoundError(f"No entropy_model/ directory in {model_dir}")

    # --- Read config from params.json ---
    params = json.loads((entropy_dir / "params.json").read_text())
    ecfg = params.get("entropy_model", params)
    print(f"  Entropy model config: dim={ecfg['dim']}, layers={ecfg['n_layers']}, "
          f"heads={ecfg['n_heads']}, vocab={ecfg['vocab_size']}")

    # --- Load raw weights from consolidated.pth ---
    pth_path = entropy_dir / "consolidated.pth"
    if not pth_path.exists():
        raise FileNotFoundError(f"No consolidated.pth in {entropy_dir}")
    print(f"  Loading weights from {pth_path} ...")
    raw = torch.load(str(pth_path), map_location="cpu", weights_only=True)
    print(f"  Raw checkpoint: {len(raw)} keys")

    # --- Infer intermediate_size from weight shape ---
    w1_key = next((k for k in raw if "feed_forward.w1" in k), None)
    intermediate_size = raw[w1_key].shape[0] if w1_key else int(ecfg["dim"] * 8 / 3)

    # --- Build LlamaForCausalLM ---
    n_kv_heads = ecfg.get("n_kv_heads") or ecfg["n_heads"]  # null means same as n_heads
    llama_cfg = LlamaConfig(
        hidden_size=ecfg["dim"],
        intermediate_size=intermediate_size,
        num_hidden_layers=ecfg["n_layers"],
        num_attention_heads=ecfg["n_heads"],
        num_key_value_heads=n_kv_heads,
        vocab_size=ecfg["vocab_size"],
        max_position_embeddings=ecfg.get("max_seqlen", 8192),
        rms_norm_eps=ecfg.get("norm_eps", 1e-5),
        rope_theta=ecfg.get("rope_theta", 10000.0),
    )
    print(f"  LlamaConfig: hidden={llama_cfg.hidden_size}, intermediate={llama_cfg.intermediate_size}, "
          f"layers={llama_cfg.num_hidden_layers}, heads={llama_cfg.num_attention_heads}, "
          f"kv_heads={llama_cfg.num_key_value_heads}")

    model = LlamaForCausalLM(llama_cfg)

    # --- Map keys and load ---
    mapped = {_map_meta_key(k): v for k, v in raw.items()}
    info = model.load_state_dict(mapped, strict=False)
    if info.missing_keys:
        print(f"  Missing keys ({len(info.missing_keys)}): {info.missing_keys[:5]}")
    if info.unexpected_keys:
        print(f"  Unexpected keys ({len(info.unexpected_keys)}): {info.unexpected_keys[:5]}")
    if not info.missing_keys and not info.unexpected_keys:
        print("  All weights loaded successfully!")

    model = model.to(dtype=torch.bfloat16, device=DEVICE)
    model.eval()
    return model

# ---------------------------------------------------------------------------
# Entropy computation & patching
# ---------------------------------------------------------------------------

def compute_byte_entropies(entropy_model, byte_ids: list[int]) -> list[float]:
    """
    Run the entropy model on a byte sequence and return per-position
    Shannon entropy (in bits).

    The entropy model is autoregressive: logits[i] predicts byte_{i+1}.
    So entropy[i] = H(P(byte_{i+1} | byte_0 ... byte_i)).
    """
    input_ids = torch.tensor([byte_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        output = entropy_model(input_ids)
        logits = output.logits

    # (seq_len, vocab_size) -- use float32 for numerical stability
    probs = torch.softmax(logits[0].float(), dim=-1)
    ent = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1)
    return ent.cpu().tolist()


def make_patches(
    byte_ids: list[int],
    entropies: list[float],
    threshold: float,
    max_patch_length: int | None = None,
) -> list[list[int]]:
    """
    Segment a byte sequence into patches using entropy thresholding.

    A new patch starts at byte i when entropy[i-1] > threshold,
    i.e. the model was uncertain about byte i given the preceding context.
    """
    if not byte_ids:
        return []

    patches: list[list[int]] = []
    current = [byte_ids[0]]

    for i in range(1, len(byte_ids)):
        hit_max = max_patch_length is not None and len(current) >= max_patch_length
        if entropies[i - 1] > threshold or hit_max:
            patches.append(current)
            current = [byte_ids[i]]
        else:
            current.append(byte_ids[i])

    if current:
        patches.append(current)
    return patches

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def patch_to_str(patch_bytes: list[int]) -> str:
    return bytes(patch_bytes).decode("utf-8", errors="replace")


def print_patches(patches: list[list[int]]):
    for idx, p in enumerate(patches):
        print(f"  [{idx:3d}] ({len(p):2d} bytes) {patch_to_str(p)!r}")

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_threshold_sweep(model_name: str, entropy_model, patching_cfg: dict, text: str):
    byte_ids = list(text.encode("utf-8"))
    print(f"\n{'=' * 70}")
    print(f"Model: {model_name}")
    print(f"Default threshold: {patching_cfg['patching_threshold']}")
    print(f"Input ({len(byte_ids)} bytes): {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"{'=' * 70}")

    entropies = compute_byte_entropies(entropy_model, byte_ids)
    max_patch_len = patching_cfg.get("max_patch_length")

    for threshold in THRESHOLDS:
        patches = make_patches(byte_ids, entropies, threshold, max_patch_len)
        print(f"\n--- Threshold {threshold:.1f} | Total patches: {len(patches)} ---")
        print_patches(patches)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("BLT Patch Granularity - Threshold Sweep")
    print(f"Device: {DEVICE}")
    print(f"Thresholds: {THRESHOLDS}")

    for model_name, model_dir in MODEL_DIRS.items():
        if not model_dir.exists():
            print(f"\n[SKIP] {model_name}: directory '{model_dir}' not found")
            continue

        patching_cfg = read_patching_config(model_dir)
        print(f"\n{model_name} patching config: {patching_cfg}")

        entropy_model = load_entropy_model(model_dir)
        run_threshold_sweep(model_name, entropy_model, patching_cfg, SAMPLE_TEXT)

        del entropy_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
