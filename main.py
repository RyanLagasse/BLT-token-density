#!/usr/bin/env python3
"""
BLT Patch Granularity Experiment - Threshold Sweep

Loads BLT models (1B, 7B), varies the entropy patching threshold,
counts and prints the resulting patches at each level.
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "The capital of France is Paris. "
    "What is 2 + 2? The answer is 4. "
    "Photosynthesis converts sunlight into chemical energy."
)

# Threshold sweep from the experiment spec
THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0]

MODEL_DIRS = {
    "blt-1b": Path("blt-1b"),
    "blt-7b": Path("blt-7b"),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir: Path):
    """Load a BLT model from a local directory."""
    print(f"\nLoading model from {model_dir} ...")
    config = AutoConfig.from_pretrained(model_dir)

    # Print relevant patching config
    for key in ("patching_mode", "patching_threshold", "patch_in_forward", "max_patch_length"):
        print(f"  {key}: {getattr(config, key, 'N/A')}")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    model.eval()
    return model

# ---------------------------------------------------------------------------
# Entropy model access
# ---------------------------------------------------------------------------

def get_entropy_model(model):
    """
    Extract the entropy sub-model from the full BLT model.
    Tries several common attribute paths; prints discovered
    module names if none match so you can adjust.
    """
    candidates = [
        "model.entropy_model",
        "entropy_model",
        "model.blt_encoder.entropy_model",
        "model.encoder.entropy_model",
    ]
    for attr_path in candidates:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            print(f"  Found entropy model at: {attr_path}")
            return obj
        except AttributeError:
            continue

    # Discovery fallback: show modules that look relevant
    print("\n[!] Could not auto-detect entropy model. Relevant modules found:")
    for name, _ in model.named_modules():
        if any(k in name.lower() for k in ("entropy", "patch", "local_encoder")):
            print(f"    {name}")
    raise RuntimeError(
        "Cannot locate entropy sub-model. Update get_entropy_model() "
        "with the correct attribute path from the list above."
    )

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

def run_threshold_sweep(model_name: str, model, text: str):
    byte_ids = list(text.encode("utf-8"))
    print(f"\n{'=' * 70}")
    print(f"Model: {model_name}")
    print(f"Input ({len(byte_ids)} bytes): {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"{'=' * 70}")

    entropy_model = get_entropy_model(model)
    entropies = compute_byte_entropies(entropy_model, byte_ids)
    max_patch_len = getattr(model.config, "max_patch_length", None)

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

        model = load_model(model_dir)
        run_threshold_sweep(model_name, model, SAMPLE_TEXT)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
