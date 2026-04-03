#!/usr/bin/env python3
"""
Abliteration pipeline for Gemma 4 31B-IT.

Removes refusal guardrails by identifying a "refusal direction" in activation space
(via contrasting harmful vs harmless prompt activations) and orthogonalizing weight
matrices to eliminate the model's ability to write to that direction.

Designed for Apple Silicon M4 Max with 64GB RAM — processes safetensor shards one
at a time to stay within memory limits.

Usage:
    python abliterate.py --weight 1.0 --num-harmful 256 --num-harmless 256 \
        --output-dir ./models/gemma-4-31b-abliterated-bf16
"""

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file as torch_save_file
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MLX_MODEL_ID = "mlx-community/gemma-4-31b-it-4bit"
BF16_MODEL_ID = "google/gemma-4-31b-it"
HIDDEN_SIZE = 5376
NUM_LAYERS = 60

PROMPTS_DIR = Path(__file__).parent / "prompts"
CHECKPOINT_DIR = Path(__file__).parent / ".abliterate_checkpoints"


# ---------------------------------------------------------------------------
# Phase 1: Collect activations using the MLX 4-bit model
# ---------------------------------------------------------------------------
def load_prompts(path: Path, max_count: int) -> list[str]:
    """Load prompts from a text file, one per line."""
    with open(path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts[:max_count]


def forward_with_hidden_states(inner_model, h, cache, per_layer_inputs=None):
    """
    Custom forward pass through the Gemma 4 inner model that returns hidden
    states after each layer. This replaces the model's __call__ because MLX
    Module.__call__ is implemented in C and cannot be monkey-patched.

    Returns:
        output: final normed hidden state (same as original __call__)
        hidden_states: list of mx.array, one per layer, each (batch, seq, hidden)
    """
    from mlx_vlm.models.gemma4.language import create_attention_mask

    layers = inner_model.layers
    num_layers = len(layers)
    hidden_states = []

    # Build attention masks (mirrors the original __call__)
    global_cache_idx = None
    sliding_cache_idx = None
    for i, layer in enumerate(layers):
        if layer.layer_type == "full_attention" and global_cache_idx is None:
            global_cache_idx = i
        elif layer.layer_type == "sliding_attention" and sliding_cache_idx is None:
            sliding_cache_idx = i

    global_mask = create_attention_mask(
        h, cache[global_cache_idx] if global_cache_idx is not None else None
    )
    sliding_window_mask = create_attention_mask(
        h,
        cache[sliding_cache_idx] if sliding_cache_idx is not None else None,
        window_size=inner_model.window_size,
    )

    shared_kv_store = {}

    for i, (layer, c) in enumerate(zip(layers, cache)):
        is_global = layer.layer_type == "full_attention"

        layer_shared_kv = None
        if (
            layer.self_attn.is_kv_shared_layer
            and layer.self_attn.kv_shared_layer_index is not None
        ):
            ref_idx = layer.self_attn.kv_shared_layer_index
            if ref_idx in shared_kv_store:
                layer_shared_kv, ref_offset = shared_kv_store[ref_idx]
                if c is not None:
                    c.offset = ref_offset

        local_mask = global_mask if is_global else sliding_window_mask

        per_layer_input = None
        if per_layer_inputs is not None:
            per_layer_input = per_layer_inputs[:, :, i, :]

        pre_offset = c.offset if c is not None else 0

        h = layer(
            h,
            local_mask,
            c,
            per_layer_input=per_layer_input,
            shared_kv=layer_shared_kv,
        )

        hidden_states.append(h)

        if layer.self_attn.store_full_length_kv:
            shared_kv_store[i] = (layer.self_attn._last_kv, pre_offset)

    return inner_model.norm(h), hidden_states


def collect_activations_mlx(
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    checkpoint_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run forward passes through the MLX 4-bit model and collect the hidden state
    at the last token position after each transformer layer.

    Returns:
        harmful_acts:  (num_harmful,  num_layers, hidden_size)  float32
        harmless_acts: (num_harmless, num_layers, hidden_size)  float32
    """
    # Check for cached activations
    harmful_cache = checkpoint_path / "harmful_activations.npy"
    harmless_cache = checkpoint_path / "harmless_activations.npy"
    if harmful_cache.exists() and harmless_cache.exists():
        print("[Phase 1] Loading cached activations...")
        return np.load(harmful_cache), np.load(harmless_cache)

    print("[Phase 1] Loading MLX 4-bit model for activation collection...")
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(MLX_MODEL_ID)
    language_model = model.language_model
    inner_model = language_model.model

    def get_last_token_activations(prompt_text: str) -> np.ndarray:
        """Run a single prompt and return (num_layers, hidden_size) numpy array."""
        formatted = apply_chat_template(processor, model.config, prompt_text)

        # Tokenize
        if hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer
        else:
            tokenizer = processor

        inputs = tokenizer.encode(formatted, return_tensors="mlx")
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
        else:
            input_ids = inputs

        # Compute embeddings (mirrors inner_model.__call__ preamble)
        h = inner_model.embed_tokens(input_ids)
        h = h * inner_model.embed_scale

        # Per-layer inputs (Gemma 4 has per-layer embeddings)
        per_layer_inputs = None
        if inner_model.hidden_size_per_layer_input:
            per_layer_inputs = inner_model.get_per_layer_inputs(input_ids)
            per_layer_inputs = inner_model.project_per_layer_inputs(h, per_layer_inputs)

        cache = language_model.make_cache()
        normed_h, hidden_states = forward_with_hidden_states(
            inner_model, h, cache, per_layer_inputs
        )

        # Evaluate all hidden states
        mx.eval(hidden_states)

        # Extract last token position from each layer
        acts = np.zeros((NUM_LAYERS, HIDDEN_SIZE), dtype=np.float32)
        for layer_idx in range(NUM_LAYERS):
            hs = hidden_states[layer_idx]          # (1, seq_len, hidden_size)
            last = hs[:, -1, :].astype(mx.float32) # cast bfloat16 -> float32
            acts[layer_idx] = np.array(last[0])

        return acts

    # Collect harmful activations
    print(f"[Phase 1] Collecting activations for {len(harmful_prompts)} harmful prompts...")
    harmful_acts_list = []
    for prompt in tqdm(harmful_prompts, desc="Harmful"):
        try:
            acts = get_last_token_activations(prompt)
            harmful_acts_list.append(acts)
        except Exception as e:
            print(f"  Warning: skipped prompt due to error: {e}")
            continue

    # Collect harmless activations
    print(f"[Phase 1] Collecting activations for {len(harmless_prompts)} harmless prompts...")
    harmless_acts_list = []
    for prompt in tqdm(harmless_prompts, desc="Harmless"):
        try:
            acts = get_last_token_activations(prompt)
            harmless_acts_list.append(acts)
        except Exception as e:
            print(f"  Warning: skipped prompt due to error: {e}")
            continue

    # Free model from memory
    del model, processor, language_model, inner_model
    gc.collect()

    harmful_acts = np.stack(harmful_acts_list)    # (N, 60, 5376)
    harmless_acts = np.stack(harmless_acts_list)   # (N, 60, 5376)

    # Cache to disk
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    np.save(harmful_cache, harmful_acts)
    np.save(harmless_cache, harmless_acts)
    print(f"[Phase 1] Saved activations to {checkpoint_path}")

    return harmful_acts, harmless_acts


# ---------------------------------------------------------------------------
# Phase 2: Compute per-layer refusal directions
# ---------------------------------------------------------------------------
def compute_refusal_directions(
    harmful_acts: np.ndarray,
    harmless_acts: np.ndarray,
    weight: float,
    checkpoint_path: Path,
) -> np.ndarray:
    """
    Compute the refusal direction for each layer as:
        d[layer] = normalize( mean(harmful[layer]) - mean(harmless[layer]) )

    Returns:
        refusal_dirs: (num_layers, hidden_size) float32, L2-normalized per row
    """
    cache_file = checkpoint_path / "refusal_directions.npy"
    if cache_file.exists():
        print("[Phase 2] Loading cached refusal directions...")
        return np.load(cache_file)

    print("[Phase 2] Computing per-layer refusal directions...")
    # harmful_acts shape: (N_harmful, num_layers, hidden_size)
    mean_harmful = harmful_acts.mean(axis=0)     # (60, 5376)
    mean_harmless = harmless_acts.mean(axis=0)   # (60, 5376)

    refusal_dirs = mean_harmful - mean_harmless   # (60, 5376)

    # L2 normalize each row
    norms = np.linalg.norm(refusal_dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # avoid division by zero
    refusal_dirs = refusal_dirs / norms

    # Apply weight scaling
    refusal_dirs = refusal_dirs * weight

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, refusal_dirs)

    # Print some diagnostics
    raw_norms = np.linalg.norm(mean_harmful - mean_harmless, axis=1)
    print(f"  Raw direction norms — min: {raw_norms.min():.4f}, "
          f"max: {raw_norms.max():.4f}, mean: {raw_norms.mean():.4f}")
    top5 = np.argsort(raw_norms)[-5:][::-1]
    print(f"  Layers with strongest refusal signal: {top5.tolist()}")
    print(f"  Applied weight factor: {weight}")
    print(f"[Phase 2] Refusal directions saved to {cache_file}")

    return refusal_dirs


# ---------------------------------------------------------------------------
# Phase 3: Modify bf16 weights
# ---------------------------------------------------------------------------
def orthogonalize_matrix(W: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Remove the refusal direction from a weight matrix:
        W_new = W - (W @ d) outer d
    where d is the refusal direction vector (already unit-normalized and scaled).

    W: (out_features, in_features) or (vocab_size, hidden_size) for embeddings
    d: (hidden_size,) — refusal direction, may be pre-scaled by weight factor
    """
    # (W @ d) gives a column vector of projections, shape (out_features,)
    proj = W @ d                       # (out_features,)
    correction = np.outer(proj, d)     # (out_features, hidden_size)
    return W - correction


def get_layer_index(tensor_name: str) -> int | None:
    """Extract layer index from a tensor name like 'model.language_model.layers.42.mlp.down_proj.weight'."""
    parts = tensor_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def should_modify_tensor(tensor_name: str) -> bool:
    """Check if this tensor is a language model weight we should abliterate.
    Excludes vision_tower and other non-language-model weights."""
    if "language_model" not in tensor_name:
        return False
    targets = [
        "embed_tokens.weight",
        "self_attn.o_proj.weight",
        "mlp.down_proj.weight",
    ]
    return any(t in tensor_name for t in targets)


def modify_bf16_weights(
    refusal_dirs: np.ndarray,
    output_dir: Path,
    checkpoint_path: Path,
):
    """
    Download the bf16 safetensors, modify relevant weight matrices by
    orthogonalizing out the refusal direction, and save to output_dir.
    Processes one shard at a time to conserve memory.
    """
    done_marker = checkpoint_path / "phase3_done"
    if done_marker.exists() and output_dir.exists():
        print("[Phase 3] Weights already modified, skipping...")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download the index file to understand shard layout
    print("[Phase 3] Downloading model index...")
    idx_path = hf_hub_download(BF16_MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    print(f"  Model has {len(shard_names)} shard(s): {shard_names}")

    # Build a reverse map: shard -> list of tensor names to modify
    shard_targets: dict[str, list[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        if should_modify_tensor(tensor_name):
            shard_targets.setdefault(shard_name, []).append(tensor_name)

    total_modified = sum(len(v) for v in shard_targets.values())
    print(f"  Will modify {total_modified} tensors across {len(shard_targets)} shard(s)")

    # Copy non-safetensor files (config, tokenizer, etc.)
    print("[Phase 3] Copying config and tokenizer files...")
    copy_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    ]
    for fname in copy_files:
        try:
            src = hf_hub_download(BF16_MODEL_ID, fname)
            dst = output_dir / fname
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"    Copied {fname}")
        except Exception:
            pass  # Some files may not exist

    # Process each shard
    for shard_idx, shard_name in enumerate(shard_names):
        shard_done = checkpoint_path / f"shard_{shard_name}.done"
        out_shard = output_dir / shard_name

        if shard_done.exists() and out_shard.exists():
            print(f"[Phase 3] Shard {shard_idx+1}/{len(shard_names)}: {shard_name} — already done, skipping")
            continue

        print(f"[Phase 3] Processing shard {shard_idx+1}/{len(shard_names)}: {shard_name}")
        t0 = time.time()

        # Download the shard
        shard_path = hf_hub_download(BF16_MODEL_ID, shard_name)

        # Load all tensors from this shard (use torch framework for bfloat16 support)
        tensors = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        targets_in_shard = shard_targets.get(shard_name, [])
        modified_count = 0

        for tensor_name in tqdm(targets_in_shard, desc=f"  Modifying {shard_name}"):
            W = tensors[tensor_name]
            original_dtype = W.dtype

            # Determine which refusal direction to use
            layer_idx = get_layer_index(tensor_name)

            if "embed_tokens" in tensor_name:
                # Embedding: shape (vocab_size, hidden_size)
                # Refusal dir is in dim 1 (hidden_size). Project rows onto d.
                d_np = refusal_dirs.mean(axis=0).astype(np.float32)
                d_np = d_np / max(np.linalg.norm(d_np), 1e-8)
                d = torch.from_numpy(d_np)
                W_f32 = W.float()
                # proj_coeffs[i] = W[i] . d, then subtract proj_coeffs[i] * d from each row
                proj_coeffs = W_f32 @ d          # (vocab_size,)
                correction = proj_coeffs.unsqueeze(1) * d.unsqueeze(0)  # (vocab_size, hidden_size)
                tensors[tensor_name] = (W_f32 - correction).to(original_dtype)
                modified_count += 1

            elif layer_idx is not None and 0 <= layer_idx < NUM_LAYERS:
                # o_proj: (hidden_size, head_dim), down_proj: (hidden_size, intermediate)
                # Refusal dir is in dim 0 (output hidden_size).
                # Remove component of each row along d: W_new[i] = W[i] - (W[i].d)*d
                d = torch.from_numpy(refusal_dirs[layer_idx].astype(np.float32))
                W_f32 = W.float()
                # For (out_features, in_features) where refusal dir is in out_features:
                # We want to zero out the d-component of each column of W
                # W_new = W - d (d^T W) = W - outer(d, d^T @ W)
                proj_row = d @ W_f32              # (in_features,) — d projected through W
                correction = d.unsqueeze(1) * proj_row.unsqueeze(0)  # (out, in)
                tensors[tensor_name] = (W_f32 - correction).to(original_dtype)
                modified_count += 1

            else:
                print(f"    Warning: could not determine layer for {tensor_name}, skipping")

        # Save modified shard
        print(f"  Saving modified shard ({modified_count} tensors modified)...")
        torch_save_file(tensors, str(out_shard))

        # Free memory
        del tensors
        gc.collect()

        # Mark shard as done
        shard_done.touch()
        elapsed = time.time() - t0
        print(f"  Shard completed in {elapsed:.1f}s")

    # Mark phase as done
    done_marker.touch()
    print(f"[Phase 3] All shards saved to {output_dir}")


# ---------------------------------------------------------------------------
# Phase 4: Convert bf16 to MLX 4-bit
# ---------------------------------------------------------------------------
def convert_to_mlx_4bit(bf16_dir: Path, mlx_dir: Path, checkpoint_path: Path):
    """Convert the abliterated bf16 model to MLX 4-bit format using mlx-lm."""
    done_marker = checkpoint_path / "phase4_done"
    if done_marker.exists() and mlx_dir.exists():
        print("[Phase 4] MLX 4-bit model already exists, skipping...")
        return

    print(f"[Phase 4] Converting bf16 model to MLX 4-bit...")
    print(f"  Source: {bf16_dir}")
    print(f"  Target: {mlx_dir}")

    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-m", "mlx_lm.convert",
            "--hf-path", str(bf16_dir),
            "--mlx-path", str(mlx_dir),
            "-q",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  mlx_lm.convert stderr:\n{result.stderr}")
        print(f"  mlx_lm.convert stdout:\n{result.stdout}")
        raise RuntimeError(f"mlx_lm.convert failed with return code {result.returncode}")

    done_marker.touch()
    print(f"[Phase 4] MLX 4-bit model saved to {mlx_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Abliterate Gemma 4 31B-IT — remove refusal guardrails"
    )
    parser.add_argument(
        "--weight", type=float, default=1.0,
        help="Scaling factor for the refusal direction (default: 1.0)"
    )
    parser.add_argument(
        "--num-harmful", type=int, default=256,
        help="Number of harmful prompts to use (default: 256)"
    )
    parser.add_argument(
        "--num-harmless", type=int, default=256,
        help="Number of harmless prompts to use (default: 256)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="./models/gemma-4-31b-abliterated-bf16",
        help="Directory to save modified bf16 weights"
    )
    parser.add_argument(
        "--mlx-output-dir", type=str,
        default="./models/gemma-4-31b-abliterated-mlx-4bit",
        help="Directory to save MLX 4-bit quantized model"
    )
    parser.add_argument(
        "--skip-mlx-convert", action="store_true",
        help="Skip Phase 4 (MLX 4-bit conversion)"
    )
    parser.add_argument(
        "--clean-checkpoints", action="store_true",
        help="Remove checkpoint files and start fresh"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    mlx_dir = Path(args.mlx_output_dir).resolve()
    checkpoint_path = CHECKPOINT_DIR.resolve()

    if args.clean_checkpoints and checkpoint_path.exists():
        print(f"Removing checkpoints at {checkpoint_path}...")
        shutil.rmtree(checkpoint_path)

    checkpoint_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Gemma 4 31B-IT Abliteration Pipeline")
    print("=" * 70)
    print(f"  Weight factor:    {args.weight}")
    print(f"  Harmful prompts:  {args.num_harmful}")
    print(f"  Harmless prompts: {args.num_harmless}")
    print(f"  BF16 output:      {output_dir}")
    print(f"  MLX 4-bit output: {mlx_dir}")
    print(f"  Checkpoints:      {checkpoint_path}")
    print("=" * 70)

    t_start = time.time()

    # --- Phase 1: Collect activations ---
    harmful_prompts = load_prompts(PROMPTS_DIR / "harmful.txt", args.num_harmful)
    harmless_prompts = load_prompts(PROMPTS_DIR / "harmless.txt", args.num_harmless)
    print(f"\nLoaded {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts")

    harmful_acts, harmless_acts = collect_activations_mlx(
        harmful_prompts, harmless_prompts, checkpoint_path
    )
    print(f"  Harmful activations shape:  {harmful_acts.shape}")
    print(f"  Harmless activations shape: {harmless_acts.shape}")

    # --- Phase 2: Compute refusal directions ---
    refusal_dirs = compute_refusal_directions(
        harmful_acts, harmless_acts, args.weight, checkpoint_path
    )
    print(f"  Refusal directions shape: {refusal_dirs.shape}")

    # Free activation memory
    del harmful_acts, harmless_acts
    gc.collect()

    # --- Phase 3: Modify bf16 weights ---
    modify_bf16_weights(refusal_dirs, output_dir, checkpoint_path)

    # --- Phase 4: Convert to MLX 4-bit ---
    if not args.skip_mlx_convert:
        convert_to_mlx_4bit(output_dir, mlx_dir, checkpoint_path)
    else:
        print("[Phase 4] Skipped (--skip-mlx-convert)")

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  Abliteration complete! Total time: {elapsed/60:.1f} minutes")
    print("=" * 70)
    print(f"  BF16 model:      {output_dir}")
    if not args.skip_mlx_convert:
        print(f"  MLX 4-bit model: {mlx_dir}")
    print()


if __name__ == "__main__":
    main()
