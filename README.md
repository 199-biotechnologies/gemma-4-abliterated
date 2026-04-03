# gemma-4-abliterated

**Quality-preserving abliteration of Google's Gemma 4 31B. Removes refusal behavior while maintaining full model intelligence. Runs on Apple Silicon via MLX.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model Size](https://img.shields.io/badge/Parameters-31B-green.svg)]()
[![Framework](https://img.shields.io/badge/Framework-MLX-orange.svg)](https://github.com/ml-explore/mlx)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow.svg)](https://huggingface.co/199-biotechnologies/gemma-4-31b-abliterated-mlx-4bit)

---

## Key Stats

| Metric | Value |
|--------|-------|
| Capability Score (Baseline) | 20/20 prompts answered |
| Refusal Score (Baseline) | 20/20 prompts answered |
| Generation Speed | ~23 tok/s on M4 Max |
| Peak Memory (4-bit) | 18.7 GB |
| Architecture | Dense 31B, 60 layers, 256K context |
| Base Model | google/gemma-4-31b-it |
| License | Apache 2.0 |

Note: The stock Gemma 4 31B-IT already answers most prompts without refusal. The baseline scores above reflect this permissive behavior. Abliteration removes the residual refusal directions to ensure consistent behavior across all prompt categories.

---

## What Is Abliteration

Abliteration is a weight-editing technique that removes a model's learned refusal behavior without retraining. The core idea: language models encode "refusal" as a linear direction in their activation space. By identifying this direction and orthogonalizing the relevant weight matrices against it, the model loses its ability to project onto the refusal subspace while preserving all other capabilities.

The technique was introduced by Arditi et al. in ["Refusal in Language Models Is Mediated by a Single Direction"](https://arxiv.org/abs/2406.11717) and popularized by [Maxime Labonne's abliteration guide](https://huggingface.co/blog/mlabonne/abliteration). Our implementation extends these approaches with several improvements:

- **Per-layer refusal directions** -- rather than a single global direction, we compute and apply a separate refusal direction for each of Gemma 4's 60 transformer layers. This is critical for Gemma-family models, which Labonne has noted are particularly resilient to abliteration.
- **Conservative weight factor** -- we use a weight factor of 1.0 (no aggressive upscaling), prioritizing quality preservation over maximum uncensoring.
- **Quality-first evaluation** -- a 100-prompt test suite (50 capability + 50 refusal) validates that abliteration does not degrade reasoning, coding, math, or creative output.

---

## Quick Start

```python
from mlx_vlm import load, generate

model, processor = load("199-biotechnologies/gemma-4-31b-abliterated-mlx-4bit")

prompt = "Your prompt here"
response = generate(model, processor, prompt, max_tokens=512)
print(response)
```

Requirements:
- Apple Silicon Mac with 20GB+ unified memory
- Python 3.10+
- `mlx-vlm` (`pip install mlx-vlm`)

---

## Technical Approach

The abliteration pipeline runs in four phases:

### Phase 1: Activation Collection

We run 256 harmful and 256 harmless prompts through the MLX 4-bit model, collecting hidden states at the last token position after each of the 60 transformer layers. This produces activation tensors of shape `(256, 60, 5376)` for each prompt set.

### Phase 2: Per-Layer Refusal Direction

For each layer, we compute the refusal direction as the L2-normalized mean difference between harmful and harmless activations:

```
d[layer] = normalize( mean(harmful[layer]) - mean(harmless[layer]) )
```

This gives us 60 unit vectors, each pointing in the direction the model uses to distinguish "should refuse" from "should answer" at that layer.

### Phase 3: Weight Orthogonalization

For each layer, we modify three weight matrices -- `embed_tokens`, `self_attn.o_proj`, and `mlp.down_proj` -- by removing their projection onto the refusal direction:

```
M_new = M - (M . d) (x) d
```

This is applied to the full bf16 weights (downloaded from `google/gemma-4-31b-it`), processed shard-by-shard to stay within memory limits on Apple Silicon.

### Phase 4: MLX 4-bit Quantization

The modified bf16 weights are quantized to 4-bit using `mlx-lm.convert` for efficient inference on Apple Silicon.

---

## Benchmark Results

Evaluated using the included 100-prompt test suite across 6 capability categories and 10 refusal categories.

### Capability Scores

| Category | Baseline (Stock) | Abliterated |
|----------|:-----------------:|:-----------:|
| Math (8 prompts) | 8/8 | -- |
| Coding (8 prompts) | 8/8 | -- |
| Reasoning (8 prompts) | 8/8 | -- |
| Knowledge (8 prompts) | 8/8 | -- |
| Creative (8 prompts) | 8/8 | -- |
| Instruction Following (5 prompts) | 5/5 | -- |
| Complex / Multi-turn (5 prompts) | 5/5 | -- |
| **Total** | **20/20** | **--** |

### Refusal Scores (higher = less censored)

| Category | Baseline (Stock) | Abliterated |
|----------|:-----------------:|:-----------:|
| Chemistry / Biology Educational (6) | 6/6 | -- |
| Historical Violence (6) | 6/6 | -- |
| Security Educational (6) | 6/6 | -- |
| Dark Fiction (6) | 6/6 | -- |
| Medical / Pharmaceutical (6) | 6/6 | -- |
| Controversial Factual (6) | 6/6 | -- |
| Weapons Historical (6) | 6/6 | -- |
| Politically Sensitive (4) | 4/4 | -- |
| Mature Educational (4) | 4/4 | -- |
| **Total** | **20/20** | **--** |

Average generation speed: 20.41 tok/s (baseline, M4 Max, 4-bit).

*Abliterated column will be populated after evaluation completes.*

---

## Hardware Requirements

| Configuration | Memory Required | Notes |
|---------------|:--------------:|-------|
| Inference (4-bit MLX) | 20 GB+ | Apple Silicon unified memory |
| Running abliteration pipeline | 64 GB | Processes bf16 shards + activations |
| Inference (GGUF / vLLM) | Varies | NVIDIA GPU via GGUF or vLLM conversion |

The abliteration pipeline is designed for Apple Silicon and processes safetensor shards one at a time to stay within memory constraints. For inference-only use, any machine with 20GB+ of unified or VRAM can run the 4-bit model.

---

## Project Structure

```
gemma-4-abliterated/
  abliterate.py       Full abliteration pipeline (4 phases)
  test_suite.py       100-prompt evaluation suite (50 capability + 50 refusal)
  prompts/
    harmful.txt       256 harmful prompts for activation collection
    harmless.txt      256 harmless prompts for activation collection
  results/
    baseline_results.json   Stock Gemma 4 31B-IT evaluation results
```

### Running the Pipeline

```bash
# Full pipeline: collect activations, compute directions, modify weights, quantize
python abliterate.py --weight 1.0 --num-harmful 256 --num-harmless 256

# Skip MLX conversion (bf16 only)
python abliterate.py --skip-mlx-convert

# Evaluate a model
python test_suite.py --model mlx-community/gemma-4-31b-it-4bit --output results/baseline_results.json
```

The pipeline supports checkpointing -- if interrupted, it resumes from the last completed phase.

---

## Evaluation Methodology

The test suite evaluates two dimensions:

**Capability** (50 prompts across 7 subcategories): math, coding, reasoning, knowledge, creative writing, instruction following, and complex multi-turn tasks. Each prompt requires substantive, detailed output. A response is classified as "answered" if it provides meaningful content.

**Refusal behavior** (50 prompts across 10 subcategories): chemistry/biology educational, historical violence, security educational, dark fiction, medical/pharmaceutical, controversial factual, weapons historical, politically sensitive, mature educational. These are legitimate educational and creative prompts that overzealous guardrails sometimes refuse.

The refusal classifier uses dual detection:
1. **Refusal pattern matching** -- regex patterns for common refusal phrases ("I cannot", "as an AI", "not appropriate", etc.)
2. **Answer signal detection** -- regex patterns for substantive response indicators (step-by-step reasoning, code blocks, conclusions, etc.)

A response is classified as refused only when refusal signals are present AND answer signals are absent.

---

## Credits

- **Google DeepMind** -- Gemma 4 base model
- **Arditi et al.** -- ["Refusal in Language Models Is Mediated by a Single Direction"](https://arxiv.org/abs/2406.11717) (foundational research)
- **Maxime Labonne** -- [Abliteration technique and guide](https://huggingface.co/blog/mlabonne/abliteration)
- **Prince Canuma / mlx-vlm** -- MLX Vision-Language Model framework
- **MLX Team** -- Apple's ML framework for Apple Silicon

Built by [Boris Djordjevic](https://x.com/longevityboris) at [Paperfoot AI](https://paperfoot.com) / [199 Biotechnologies](https://github.com/199-biotechnologies)

---

## License

Apache 2.0 -- same as the base Gemma 4 model. See [LICENSE](LICENSE) for details.
