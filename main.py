#!/usr/bin/env python3
"""
BLT Patch Granularity Experiment - Threshold Sweep

Loads the BLT entropy model (patcher) from the entropy_model/ subdirectory,
varies the entropy patching threshold, counts and prints byte patches.

The entropy model is a small byte-level causal LM (LLaMA-architecture).
The checkpoint uses Meta naming conventions (attention.wq, feed_forward.w1, etc.)
which must be mapped to HF naming (self_attn.q_proj, mlp.gate_proj, etc.).
"""

import json
import torch
from pathlib import Path
from safetensors.torch import load_file as load_safetensors

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "The capital of France is Paris. "
    "What is 2 + 2? The answer is 4. "
    "Photosynthesis converts sunlight into chemical energy."
)

THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0]

MODEL_DIRS = {
    "blt-1b": Path("blt-1b"),
    "blt-7b": Path("blt-7b"),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Meta -> HF weight key mapping
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
    if key == "output.weight":
        return "lm_head.weight"
    if not key.startswith("lm_head"):
        key = "model." + key
    return key

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_patching_config(model_dir: Path) -> dict:
    """Read patching-related fields from the main model config."""
    cfg = json.loads((model_dir / "config.json").read_text())
    return {
        "patching_mode": cfg.get("patching_mode", "entropy"),
        "patching_threshold": cfg.get("patching_threshold"),
        "max_patch_length": cfg.get("max_patch_length"),
    }


def _load_raw_weights(directory: Path) -> dict:
    """Load weights from safetensors (preferred) or .bin files."""
    weights = {}
    for sf in sorted(directory.glob("*.safetensors")):
        weights.update(load_safetensors(str(sf)))
    if not weights:
        for bf in sorted(directory.glob("*.bin")):
            weights.update(torch.load(str(bf), map_location="cpu", weights_only=True))
    if not weights:
        raise FileNotFoundError(f"No weight files found in {directory}")
    return weights


def _infer_llama_config(entropy_dir: Path, raw_weights: dict):
    """Build a LlamaConfig from the entropy model config.json + weight shapes."""
    from transformers import LlamaConfig

    cfg_path = entropy_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    # Determine hidden size from embedding weight shape
    emb_key = next((k for k in raw_weights if "tok_embeddings" in k or "embed_tokens" in k), None)
    if emb_key is not None:
        vocab_size_w, hidden_size_w = raw_weights[emb_key].shape
    else:
        vocab_size_w, hidden_size_w = None, None

    # Count layers from weight keys
    layer_indices = [
        int(k.split(".")[1])
        for k in raw_weights
        if k.startswith("layers.")
    ]
    n_layers_w = max(layer_indices) + 1 if layer_indices else None

    # Config values with fallbacks: config.json (Meta names) -> config.json (HF names) -> weight shapes
    hidden_size = cfg.get("dim", cfg.get("hidden_size", hidden_size_w or 512))
    n_layers = cfg.get("n_layers", cfg.get("num_hidden_layers", n_layers_w or 14))
    n_heads = cfg.get("n_heads", cfg.get("num_attention_heads", 8))
    n_kv_heads = cfg.get("n_kv_heads", cfg.get("num_key_value_heads", n_heads))
    vocab_size = cfg.get("vocab_size", vocab_size_w or 260)

    # Intermediate size: check config, else infer from weight shape
    ffn_key = next((k for k in raw_weights if "feed_forward.w1" in k or "mlp.gate_proj" in k), None)
    if "hidden_dim" in cfg:
        intermediate_size = cfg["hidden_dim"]
    elif "intermediate_size" in cfg:
        intermediate_size = cfg["intermediate_size"]
    elif ffn_key is not None:
        intermediate_size = raw_weights[ffn_key].shape[0]
    else:
        intermediate_size = int(hidden_size * 8 / 3)

    return LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        vocab_size=vocab_size,
        max_position_embeddings=cfg.get("max_seq_len", cfg.get("max_position_embeddings", 8192)),
        rms_norm_eps=cfg.get("norm_eps", cfg.get("rms_norm_eps", 1e-5)),
        rope_theta=cfg.get("rope_theta", 500000.0),
    )


def load_entropy_model(model_dir: Path):
    """
    Load the entropy/patcher model from {model_dir}/entropy_model/.

    Strategy:
      1. Try direct AutoModel load (works if weights are already HF-formatted).
      2. Fall back to building a LlamaForCausalLM and loading with Meta->HF key mapping.
    """
    from transformers import AutoModelForCausalLM, LlamaForCausalLM

    entropy_dir = model_dir / "entropy_model"
    if not entropy_dir.exists():
        raise FileNotFoundError(f"No entropy_model/ directory in {model_dir}")

    # --- Approach 1: direct HF load ---
    try:
        m = AutoModelForCausalLM.from_pretrained(
            entropy_dir, torch_dtype=torch.bfloat16, device_map=DEVICE,
        )
        m.eval()
        print(f"  Entropy model loaded directly from {entropy_dir}")
        return m
    except Exception as e:
        print(f"  Direct AutoModel load failed: {e}")
        print("  Falling back to LlamaForCausalLM + key mapping ...")

    # --- Approach 2: manual load with key mapping ---
    raw_weights = _load_raw_weights(entropy_dir)
    print(f"  Raw weight keys (sample): {list(raw_weights.keys())[:5]}")

    llama_cfg = _infer_llama_config(entropy_dir, raw_weights)
    print(f"  Inferred config: layers={llama_cfg.num_hidden_layers}, "
          f"hidden={llama_cfg.hidden_size}, heads={llama_cfg.num_attention_heads}, "
          f"vocab={llama_cfg.vocab_size}")

    m = LlamaForCausalLM(llama_cfg)
    mapped = {_map_meta_key(k): v for k, v in raw_weights.items()}

    info = m.load_state_dict(mapped, strict=False)
    if info.missing_keys:
        print(f"  Missing keys ({len(info.missing_keys)}): {info.missing_keys[:5]}")
    if info.unexpected_keys:
        print(f"  Unexpected keys ({len(info.unexpected_keys)}): {info.unexpected_keys[:5]}")

    m = m.to(dtype=torch.bfloat16, device=DEVICE)
    m.eval()
    print("  Entropy model loaded via LlamaForCausalLM + key mapping")
    return m

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
        logits = output.logits if hasattr(output, "logits") else output[0]

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
