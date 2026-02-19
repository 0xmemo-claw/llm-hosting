# LLM Hosting — One Model to Rule Them All

## TL;DR

**Qwen3.5-397B-A17B** on **3× H200 141GB** — single model for everything, no tiers, ~$7,755/mo:

- 397B total params, 17B active per token (sparse MoE — fast as a 17B, smart as a 397B)
- FP8 precision, tensor parallel across 3 GPUs — full quality, maximum throughput
- Beats Claude Opus 4.5 on instruction following (IFBench 76.5 vs 58.0) and agentic tasks
- 262K native context, extensible to 1M with YaRN
- Apache 2.0 — fully open-weight, no API keys, no rate limits

> **Tradeoff acknowledged:** This setup costs more than the 3-tier approach. You get one best-in-class open-weight model for everything instead of juggling haiku/sonnet/opus routing. If coding peak performance (LCB 90%+) is your only metric, Gemini-3 Pro edges it. If instruction following and agentic reliability matter more — Qwen3.5-397B wins.

---

## Why Qwen3.5-397B-A17B

### Benchmarks (vs frontier closed models)

| Benchmark | GPT-5.2 | Claude 4.5 Opus | Gemini-3 Pro | **Qwen3.5-397B** |
|---|---:|---:|---:|---:|
| MMLU-Pro | 87.4 | 89.5 | 89.8 | 87.8 |
| MMLU-Redux | 95.0 | 95.6 | 95.9 | 94.9 |
| IFBench | 75.4 | 58.0 | 70.4 | **76.5 🥇** |
| MultiChallenge | 57.9 | 54.2 | 64.2 | **67.6 🥇** |
| GPQA | 92.4 | 87.0 | 91.9 | 88.4 |
| LiveCodeBench v6 | 87.7 | 84.8 | 90.7 | 83.6 |
| AIME26 | 96.7 | 93.3 | 90.6 | 91.3 |
| BrowseComp | — | — | — | **78.6 🥇** |
| BFCL-V4 | 63.1 | **77.5** | 72.5 | 72.9 |

Source: [Qwen3.5-397B-A17B HuggingFace model card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)

**The signal that matters for agents:**

- **IFBench (instruction following):** 76.5 — beats every frontier model including GPT-5.2. This is the benchmark most correlated with agent reliability.
- **MultiChallenge (multi-turn, complex instructions):** 67.6 — #1 overall. Agents live in multi-turn loops.
- **BrowseComp (agentic web browsing):** 78.6 — only model with a published score. Built for agentic use.

**Where it's not #1:**

- LiveCodeBench v6: 83.6 vs Gemini-3 Pro's 90.7. Competitive programming is not its ceiling.
- HLE: 28.7 vs GPT-5.2's 35.5. Frontier scientific reasoning is a gap.
- BFCL-V4 (function calling): Claude 4.5 Opus wins at 77.5.

**Bottom line:** Best open-weight model for agents that need to follow instructions reliably across long contexts. Coding is good (LCB 83.6 > Claude 4.5 Opus 84.8 — effectively tied), just not Gemini-level competitive programming.

### Architecture Highlights

- **Sparse MoE:** 512 experts, 10 routed + 1 shared active per token → 17B active params, 397B total. Throughput of a 17B model, quality of a much larger one.
- **Gated Delta Networks:** Novel SSM-hybrid architecture replacing standard attention in some layers — better long-context efficiency.
- **Multi-Token Prediction (MTP):** Built-in speculative decoding target. SGLang exploits this with NEXTN algorithm for measurable throughput gains.
- **262K native context:** Extensible to ~1M tokens via YaRN (4× factor).
- **Thinking mode:** `<think>...</think>` enabled by default. Pass `enable_thinking=False` via the API to disable for latency-sensitive paths.
- **Multimodal:** Vision + language via early fusion. One model handles text and images.
- **201 languages:** Multilingual out of the box.
- **License:** Apache 2.0. Use commercially, modify freely, no royalties.

---

## Hardware Options

> VRAM math first, then pricing. Don't guess — get this wrong and the model doesn't load.

### VRAM Requirements

| Precision | Bytes/param | Total weights (397B) | Notes |
|---|---:|---:|---|
| BF16 | 2 | **~794GB** | Needs 6× H200 — impractical |
| FP8 | 1 | **~397GB** | 3× H200 or 5× H100 |
| 4-bit AWQ/GGUF | 0.5 | **~200GB** | 2× H200 or 3× H100 |

> These are **weight-only** estimates. Add KV cache + activations on top. Rule of thumb: weights × 1.15–1.25 total at moderate concurrency.

### Option A — 3× H200 FP8 ✅ Recommended (high perf)

```
3× H200 141GB = 423GB total VRAM
FP8 weights:    ~397GB
KV headroom:    ~26GB (modest — use shorter max_model_len or limit concurrency)
─────────────────────────────────────────────────────
Cost: 3 × $2,585 = ~$7,755/mo
```

**Why:** Full FP8 quality (indistinguishable from BF16 in practice). Tensor parallel across 3 GPUs with NVLink. Best throughput via SGLang MTP speculative decoding. KV cache is tight — keep `--max-model-len` at 131072 (128K) unless you need longer.

**Serving:** SGLang `--tp-size 3` or vLLM `--tensor-parallel-size 3`

### Option B — 2× H200 4-bit ⚖️ Budget

```
2× H200 141GB = 282GB total VRAM
4-bit weights:  ~200GB
KV headroom:    ~82GB (comfortable — 262K context works)
─────────────────────────────────────────────────────
Cost: 2 × $2,585 = ~$5,170/mo
```

**Why:** Significant cost reduction (~$2.6k/mo savings). 4-bit AWQ quality loss is measurable but small on instruction-following tasks. KV cache headroom is actually better than Option A — you can run full 262K context comfortably. Good choice if budget matters more than peak benchmark scores.

**Serving:** vLLM `--tensor-parallel-size 2` with AWQ quantized checkpoint (e.g., `Qwen/Qwen3.5-397B-A17B-AWQ`)

### Option C — 4× H100 4-bit ❌ Skip (expensive for less)

```
4× H100 80GB = 320GB total VRAM
FP8 weights:   ~397GB — DOES NOT FIT (320 < 397)
4-bit weights: ~200GB — fits with 120GB KV headroom
─────────────────────────────────────────────────────
Cost: 4 × ~$2,000 = ~$8,000/mo
```

**Why not:** More expensive than Option A (~$245/mo more) while running 4-bit instead of FP8. Worse quality AND higher cost. Only useful if H200s are unavailable on RunPod.

---

## Serving Setup

### SGLang — Recommended

SGLang exploits MTP for speculative decoding (NEXTN algorithm), giving measurable throughput gains on this model specifically.

```bash
# Option A: 3× H200, FP8, 128K context
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-397B-A17B \
  --tp-size 3 \
  --mem-fraction-static 0.8 \
  --context-length 131072 \
  --reasoning-parser qwen3 \
  --speculative-algo NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

```bash
# Option A: 3× H200, FP8, full 262K context (tighter memory — reduce concurrency)
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-397B-A17B \
  --tp-size 3 \
  --mem-fraction-static 0.8 \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --speculative-algo NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

### vLLM

```bash
# Option A: 3× H200, FP8
vllm serve Qwen/Qwen3.5-397B-A17B \
  --tensor-parallel-size 3 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only

# Option B: 2× H200, 4-bit AWQ
vllm serve Qwen/Qwen3.5-397B-A17B-AWQ \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

### Extended context via YaRN (~1M tokens)

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3.5-397B-A17B \
  --tensor-parallel-size 3 \
  --reasoning-parser qwen3 \
  --hf-overrides '{"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}}}' \
  --max-model-len 1010000
```

> YaRN at 1M context is memory-intensive. KV cache at 1M tokens will overflow a 3× H200 setup with any concurrency. Use only for single-request long-doc workloads, scale down `--max-model-len` for production.

### Disable thinking mode per-request

The model defaults to `<think>...</think>` reasoning. Disable for fast, latency-sensitive paths:

```python
# Via OpenAI-compatible API
response = client.chat.completions.create(
    model="qwen3.5-397b",
    messages=[...],
    extra_body={"enable_thinking": False}  # skip CoT, faster response
)
```

### LiteLLM config for OpenClaw

No tiers — one model for everything. Map all aliases to the same endpoint:

```yaml
# litellm_config.yaml
model_list:
  - model_name: qwen3.5-397b
    litellm_params:
      model: openai/qwen3.5-397b
      api_base: http://localhost:30000/v1  # SGLang port
      api_key: none

  # Alias mapping — single model serves all roles
  - model_name: haiku
    litellm_params:
      model: openai/qwen3.5-397b
      api_base: http://localhost:30000/v1
      api_key: none

  - model_name: sonnet
    litellm_params:
      model: openai/qwen3.5-397b
      api_base: http://localhost:30000/v1
      api_key: none

  - model_name: opus
    litellm_params:
      model: openai/qwen3.5-397b
      api_base: http://localhost:30000/v1
      api_key: none

litellm_settings:
  drop_params: true
  set_verbose: false
```

```bash
litellm --config litellm_config.yaml --port 8000
```

**OpenClaw integration:**
- `base_url`: `http(s)://<host>:8000/v1`
- `api_key`: LiteLLM `master_key` (if set)
- Model: `qwen3.5-397b` (or any alias above)

---

## VRAM Reality Check

<details>
<summary><strong>How vLLM/SGLang use GPU memory (expand)</strong></summary>

GPU memory = **weights + KV cache + activations + overhead**. Quantization only shrinks weights.

### Weight memory (rule of thumb)

| Precision | Bytes/param | 17B active | 397B total |
|---|---:|---:|---:|
| BF16 | 2 | ~34GB | ~794GB |
| FP8 | 1 | ~17GB | **~397GB** |
| 4-bit | 0.5 | ~8.5GB | **~200GB** |

> For MoE models: only active params run computation, but **all expert weights must be loaded into VRAM**. You load 397B params, compute with 17B. No shortcut.

### KV cache grows with context × concurrency

```
KV cache ≈ 2 × layers × heads × head_dim × seq_len × batch_size × dtype_bytes
```

For Qwen3.5-397B at 131K context, 4 concurrent sequences, FP16 KV: roughly **80–120GB** depending on architecture details.

### Practical guidance for Option A (3× H200 FP8)

| Context length | Concurrent seqs | KV cache est. | Fits in 26GB headroom? |
|---|---:|---:|---|
| 32K | 1–2 | ~10–20GB | ✅ comfortable |
| 128K | 1–2 | ~40–80GB | ⚠️ reduce `mem-fraction-static` |
| 262K | 1 | ~80–120GB | ❌ need to reduce weight fraction |

> At 262K context on Option A: lower `--mem-fraction-static` to 0.65–0.70 to give KV cache more room. Accept lower concurrent throughput.

### Validate before deploying

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Watch memory during load
watch -n 2 nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# SGLang reports allocation on startup — check logs
```

**Sources:**
- vLLM memory: https://docs.vllm.ai/en/latest/configuration/conserving_memory/
- SGLang docs: https://docs.sglang.ai/
- Qwen3.5-397B model card: https://huggingface.co/Qwen/Qwen3.5-397B-A17B

</details>

---

## RunPod Pricing

<details>
<summary><strong>Pods + Serverless pricing tables (expand)</strong></summary>

### Pods (on-demand)

> Typical range = min–median from RunPod pricing + community snapshots.

| GPU | On-demand $/hr | $/30d |
|---|---:|---:|
| **H200 141GB** | $3.59 | **$2,585** |
| **H100 80GB** | $2.69–$2.84 | **$1,937–$2,045** |
| **A100 80GB** | $1.19–$1.54 | **$857–$1,109** |
| **L40S 48GB** | $0.69–$0.74 | **$497–$533** |
| **RTX 4090 24GB** | $0.68–$0.74 | **$490–$533** |

Sources (checked 2026-02-18):
- https://www.runpod.io/gpu-pricing
- https://compute.hivenet.com/post/runpod-pricing-complete-guide-to-gpu-cloud-costs-in-2025

### Serverless — Flex pricing (upper bound)

> RunPod Serverless Flex = upper-bound proxy for interruptible/spot economics. Not the same as always-on pods.

| GPU | $/s | $/hr | $/day | $/30d |
|---|---:|---:|---:|---:|
| **B200 180GB** | $0.00240 | $8.64 | $207.36 | $6,221 |
| **H200 141GB** | $0.00155 | $5.58 | $133.92 | $4,018 |
| **H100 80GB** | $0.00116 | $4.18 | $100.22 | $3,007 |
| **A100 80GB** | $0.00076 | $2.74 | $65.66 | $1,970 |
| **48GB (L40/L40S/6000 Ada)** | $0.00053 | $1.91 | $45.79 | $1,374 |
| **24GB (RTX 4090)** | $0.00031 | $1.12 | $26.78 | $804 |
| **24GB (L4/A5000/3090)** | $0.00019 | $0.68 | $16.42 | $492 |

### Serverless — Active pricing

| GPU | $/s | $/hr | $/day | $/30d |
|---|---:|---:|---:|---:|
| **B200 180GB** | $0.00190 | $6.84 | $164.16 | $4,925 |
| **H200 141GB** | $0.00124 | $4.46 | $107.14 | $3,214 |
| **H100 80GB** | $0.00093 | $3.35 | $80.35 | $2,411 |
| **A100 80GB** | $0.00060 | $2.16 | $51.84 | $1,555 |
| **48GB (L40/L40S/6000 Ada)** | $0.00037 | $1.33 | $31.97 | $959 |
| **24GB (RTX 4090)** | $0.00021 | $0.76 | $18.14 | $544 |
| **24GB (L4/A5000/3090)** | $0.00013 | $0.47 | $11.23 | $337 |

Sources: https://docs.runpod.io/serverless/pricing · https://www.runpod.io/pricing

</details>

---

## Caveats

**Cost:** $7,755/mo for Option A is ~3× the old 3-tier single-H200 setup. You're paying for having one frontier-tier open-weight model instead of tiered routing. If budget is a hard constraint, the previous 3-tier README is still valid.

**Benchmark dates:** Qwen3.5-397B benchmarks are from the HuggingFace model card at release. The landscape moves fast — these numbers will be outdated within months.

**IFBench / MultiChallenge:** These matter for agent instruction following, but they're not the only thing. Run your own evals on your actual workload before committing.

**Thinking mode overhead:** `<think>...</think>` mode adds latency. For high-throughput agent loops, disable per-request with `enable_thinking=False`. For complex reasoning tasks, leave it on.

**Multi-GPU ops:** Running tensor-parallel across 3 H200s is straightforward on RunPod with NVLink pods. Make sure you request a multi-GPU pod (not 3 separate single-GPU pods) — inter-GPU bandwidth matters.

**4-bit quality:** Option B (2× H200, AWQ 4-bit) is measurably lower quality than FP8, but for instruction following specifically the gap is smaller than for coding benchmarks. If your primary use case is agent orchestration (not code generation), Option B is a reasonable tradeoff.

**No haiku/sonnet routing:** This setup sends every request to the same model. For workloads where 90% of requests are trivial tool calls that would be faster on a small model, consider whether the throughput of MoE (17B active) is already enough — it often is.
