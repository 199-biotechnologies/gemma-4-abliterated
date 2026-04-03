#!/usr/bin/env python3
"""
Abliteration Quality Optimizer
===============================
Autoresearch-style iterative optimization to find the best abliteration
configuration for Gemma 4 31B. Tests multiple weight factors, layer subsets,
and weight matrix targets to maximize refusal removal while preserving capability.

Usage:
    python optimize_abliteration.py

Requires: completed Phase 1+2 checkpoints (activations + refusal directions).
"""

import gc
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as np_save_file

# ---------------------------------------------------------------------------
# Configuration space
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = Path(__file__).parent / ".abliterate_checkpoints"
MODELS_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"
BF16_MODEL_ID = "google/gemma-4-31b-it"

HIDDEN_SIZE = 5376
NUM_LAYERS = 60

# Configurations to test — each is a (name, params) tuple
CONFIGS = [
    {
        "name": "conservative_w1.0_all",
        "weight": 1.0,
        "layers": "all",           # all 60 layers
        "matrices": "all",         # embed + o_proj + down_proj
    },
    {
        "name": "moderate_w1.25_all",
        "weight": 1.25,
        "layers": "all",
        "matrices": "all",
    },
    {
        "name": "aggressive_w1.5_all",
        "weight": 1.5,
        "layers": "all",
        "matrices": "all",
    },
    {
        "name": "conservative_w1.0_top20",
        "weight": 1.0,
        "layers": "top20",         # only the 20 layers with strongest refusal signal
        "matrices": "all",
    },
    {
        "name": "moderate_w1.25_top20",
        "weight": 1.25,
        "layers": "top20",
        "matrices": "all",
    },
    {
        "name": "conservative_w1.0_no_embed",
        "weight": 1.0,
        "layers": "all",
        "matrices": "no_embed",    # only o_proj + down_proj (skip embedding)
    },
    {
        "name": "moderate_w1.25_no_embed",
        "weight": 1.25,
        "layers": "all",
        "matrices": "no_embed",
    },
    {
        "name": "gentle_w0.75_all",
        "weight": 0.75,
        "layers": "all",
        "matrices": "all",
    },
]


# ---------------------------------------------------------------------------
# Weight modification (mirrors abliterate.py but parameterized)
# ---------------------------------------------------------------------------
def orthogonalize_matrix(W: np.ndarray, d: np.ndarray) -> np.ndarray:
    proj = W @ d
    correction = np.outer(proj, d)
    return W - correction


def get_layer_index(tensor_name: str):
    parts = tensor_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def should_modify_tensor(tensor_name: str, matrices: str) -> bool:
    if "language_model" not in tensor_name:
        return False

    if matrices == "all":
        targets = ["embed_tokens.weight", "self_attn.o_proj.weight", "mlp.down_proj.weight"]
    elif matrices == "no_embed":
        targets = ["self_attn.o_proj.weight", "mlp.down_proj.weight"]
    elif matrices == "only_o_proj":
        targets = ["self_attn.o_proj.weight"]
    else:
        targets = ["embed_tokens.weight", "self_attn.o_proj.weight", "mlp.down_proj.weight"]

    return any(t in tensor_name for t in targets)


def create_abliterated_model(
    config: dict,
    refusal_dirs_raw: np.ndarray,  # unscaled, unit-normalized (60, 5376)
    top_layers: list[int],
    output_dir: Path,
):
    """Create an abliterated bf16 model with the given configuration."""
    from huggingface_hub import hf_hub_download

    weight = config["weight"]
    layer_mode = config["layers"]
    matrices = config["matrices"]

    # Determine which layers to modify
    if layer_mode == "all":
        active_layers = set(range(NUM_LAYERS))
    elif layer_mode == "top20":
        active_layers = set(top_layers[:20])
    elif layer_mode == "top10":
        active_layers = set(top_layers[:10])
    else:
        active_layers = set(range(NUM_LAYERS))

    # Scale refusal directions
    refusal_dirs = refusal_dirs_raw * weight

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load index
    idx_path = hf_hub_download(BF16_MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))

    # Copy config files
    for fname in ["config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "processor_config.json",
                  "chat_template.jinja", "model.safetensors.index.json"]:
        try:
            src = hf_hub_download(BF16_MODEL_ID, fname)
            dst = output_dir / fname
            if not dst.exists():
                shutil.copy2(src, dst)
        except Exception:
            pass

    modified_total = 0

    for shard_name in shard_names:
        shard_path = hf_hub_download(BF16_MODEL_ID, shard_name)
        tensors = {}
        with safe_open(shard_path, framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        for tensor_name in list(tensors.keys()):
            if not should_modify_tensor(tensor_name, matrices):
                continue

            W = tensors[tensor_name]
            original_dtype = W.dtype
            layer_idx = get_layer_index(tensor_name)

            if "embed_tokens" in tensor_name:
                # Use mean direction across active layers
                active_dirs = refusal_dirs[list(active_layers)]
                d = active_dirs.mean(axis=0).astype(np.float32)
                d = d / max(np.linalg.norm(d), 1e-8)
                W_f32 = W.astype(np.float32)
                tensors[tensor_name] = orthogonalize_matrix(W_f32, d).astype(original_dtype)
                modified_total += 1

            elif layer_idx is not None and layer_idx in active_layers:
                d = refusal_dirs[layer_idx].astype(np.float32)
                W_f32 = W.astype(np.float32)
                tensors[tensor_name] = orthogonalize_matrix(W_f32, d).astype(original_dtype)
                modified_total += 1

        np_save_file(tensors, str(output_dir / shard_name))
        del tensors
        gc.collect()

    print(f"  Modified {modified_total} tensors")
    return modified_total


def convert_to_mlx_4bit(bf16_dir: Path, mlx_dir: Path):
    """Convert bf16 to MLX 4-bit."""
    result = subprocess.run(
        [sys.executable, "-m", "mlx_lm.convert",
         "--hf-path", str(bf16_dir),
         "--mlx-path", str(mlx_dir),
         "-q"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Conversion error: {result.stderr[:500]}")
        return False
    return True


def run_test_suite(model_path: str, output_path: str) -> dict:
    """Run the 100-prompt test suite and return parsed results."""
    result = subprocess.run(
        [sys.executable, "test_suite.py",
         "--model", model_path,
         "--output", output_path,
         "--max-tokens", "512"],
        capture_output=True, text=True, timeout=7200,
    )
    if result.returncode != 0:
        print(f"  Test suite error: {result.stderr[:500]}")
        return None

    with open(output_path) as f:
        return json.load(f)


def score_config(results: dict) -> dict:
    """Score a configuration based on test results."""
    summary = results.get("summary", {})
    cap_answered = summary.get("capability_answered", 0)
    cap_total = summary.get("capability_total", 1)
    ref_answered = summary.get("refusal_answered", 0)
    ref_total = summary.get("refusal_total", 1)

    capability_pct = cap_answered / cap_total * 100
    refusal_removal_pct = ref_answered / ref_total * 100

    # Composite score: heavily penalize capability loss
    # Capability preservation is 3x more important than refusal removal
    composite = (capability_pct * 3 + refusal_removal_pct) / 4

    return {
        "capability_pct": round(capability_pct, 1),
        "refusal_removal_pct": round(refusal_removal_pct, 1),
        "composite_score": round(composite, 1),
        "cap_answered": cap_answered,
        "cap_total": cap_total,
        "ref_answered": ref_answered,
        "ref_total": ref_total,
        "avg_tps": summary.get("avg_tps", 0),
    }


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Abliteration Quality Optimizer")
    print("  Testing multiple configurations to find the best tradeoff")
    print("=" * 70)

    # Load precomputed refusal directions (unscaled, unit-normalized)
    dirs_path = CHECKPOINT_DIR / "refusal_directions.npy"
    if not dirs_path.exists():
        print("ERROR: No refusal directions found. Run abliterate.py Phase 1+2 first.")
        sys.exit(1)

    refusal_dirs = np.load(dirs_path)
    print(f"Loaded refusal directions: {refusal_dirs.shape}")

    # Compute layer ranking by refusal signal strength
    # Load raw activations to compute this
    harmful_acts = np.load(CHECKPOINT_DIR / "harmful_activations.npy")
    harmless_acts = np.load(CHECKPOINT_DIR / "harmless_activations.npy")
    raw_diff = harmful_acts.mean(axis=0) - harmless_acts.mean(axis=0)
    raw_norms = np.linalg.norm(raw_diff, axis=1)
    top_layers = np.argsort(raw_norms)[::-1].tolist()
    print(f"Top 10 layers by refusal signal: {top_layers[:10]}")
    del harmful_acts, harmless_acts, raw_diff
    gc.collect()

    # Results tracker
    all_results = []
    best_score = -1
    best_config = None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for i, config in enumerate(CONFIGS):
        name = config["name"]
        print(f"\n{'='*70}")
        print(f"  Config {i+1}/{len(CONFIGS)}: {name}")
        print(f"  Weight={config['weight']}, Layers={config['layers']}, "
              f"Matrices={config['matrices']}")
        print(f"{'='*70}")

        bf16_dir = MODELS_DIR / f"opt-{name}-bf16"
        mlx_dir = MODELS_DIR / f"opt-{name}-mlx-4bit"
        results_path = RESULTS_DIR / f"opt-{name}.json"

        # Skip if already evaluated
        if results_path.exists():
            print(f"  Already evaluated, loading cached results...")
            with open(results_path) as f:
                cached = json.load(f)
            scores = score_config(cached)
        else:
            # Phase A: Create abliterated bf16 model
            t0 = time.time()
            print(f"  Creating abliterated bf16 model...")
            create_abliterated_model(config, refusal_dirs, top_layers, bf16_dir)
            print(f"  bf16 model created in {time.time()-t0:.0f}s")

            # Phase B: Convert to MLX 4-bit
            print(f"  Converting to MLX 4-bit...")
            t0 = time.time()
            if not convert_to_mlx_4bit(bf16_dir, mlx_dir):
                print(f"  FAILED — skipping this config")
                continue
            print(f"  Conversion done in {time.time()-t0:.0f}s")

            # Phase C: Run test suite
            print(f"  Running 100-prompt test suite...")
            t0 = time.time()
            test_results = run_test_suite(str(mlx_dir), str(results_path))
            if test_results is None:
                print(f"  FAILED — skipping this config")
                continue
            print(f"  Tests done in {time.time()-t0:.0f}s")

            scores = score_config(test_results)

            # Cleanup bf16 to save disk (keep MLX for now)
            if bf16_dir.exists():
                shutil.rmtree(bf16_dir)
                print(f"  Cleaned up bf16 dir to save disk")

        print(f"\n  Results: capability={scores['capability_pct']}%, "
              f"refusal_removal={scores['refusal_removal_pct']}%, "
              f"composite={scores['composite_score']}")

        all_results.append({"config": config, "scores": scores})

        if scores["composite_score"] > best_score:
            best_score = scores["composite_score"]
            best_config = config
            print(f"  >>> NEW BEST CONFIG <<<")

        # Cleanup non-best MLX models to save disk
        for prev in all_results[:-1]:
            prev_mlx = MODELS_DIR / f"opt-{prev['config']['name']}-mlx-4bit"
            if prev_mlx.exists() and prev["config"]["name"] != best_config["name"]:
                shutil.rmtree(prev_mlx)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"\n{'Config':<35} {'Capability':>10} {'Refusal':>10} {'Composite':>10}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: x["scores"]["composite_score"], reverse=True):
        name = r["config"]["name"]
        s = r["scores"]
        marker = " <<<" if r["config"]["name"] == best_config["name"] else ""
        print(f"{name:<35} {s['capability_pct']:>9.1f}% {s['refusal_removal_pct']:>9.1f}% "
              f"{s['composite_score']:>9.1f}{marker}")

    print(f"\nBest config: {best_config['name']}")
    print(f"  Weight={best_config['weight']}, Layers={best_config['layers']}, "
          f"Matrices={best_config['matrices']}")

    # Save optimization summary
    summary_path = RESULTS_DIR / "optimization_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "best_config": best_config,
            "best_scores": score_config(
                json.load(open(RESULTS_DIR / f"opt-{best_config['name']}.json"))
            ),
            "all_results": [
                {"config": r["config"], "scores": r["scores"]}
                for r in all_results
            ],
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Rename best model to final name
    best_mlx = MODELS_DIR / f"opt-{best_config['name']}-mlx-4bit"
    final_mlx = MODELS_DIR / "gemma-4-31b-abliterated-mlx-4bit"
    if best_mlx.exists() and not final_mlx.exists():
        best_mlx.rename(final_mlx)
        print(f"Best model moved to {final_mlx}")
    elif best_mlx.exists():
        shutil.rmtree(final_mlx)
        best_mlx.rename(final_mlx)
        print(f"Best model replaced at {final_mlx}")

    print("\nOptimization complete.")


if __name__ == "__main__":
    main()
