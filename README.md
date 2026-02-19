# LLM Hosting — Self-Hosted Open-Weight Options

## Quick Comparison

| | **Option 1 — Budget ($1–1.5k/mo)** | **Option 2 — Frontier ($5–8k/mo)** |
|---|---|---|
| **Model** | Qwen3-Coder-Next (80B/3B active) | Qwen3.5-397B-A17B (397B/17B active) |
| **Best for** | Coding-focused teams, high code quality/$  | Agentic workloads, instruction following, multi-modal |
| **Budget** | ~$857–1,109/mo (A100) | ~$5,170–7,755/mo (2–3× H200) |
| **Context** | 256K native | 262K native, ~1M w/ YaRN |
| **SWE-Bench Verified** | 70.6% 🥇 open-weight value/$ | 68.1% |
| **IFBench** | — | **76.5** (beats GPT-5.2) |
| **Architecture** | MoE hybrid (Gated DeltaNet + Gated Attention + MoE) | Sparse MoE (Gated Delta Networks) |
| **License** | Apache 2.0 | Apache 2.0 |
| **VRAM (4-bit)** | ~40–46GB → 1× A100 80GB | ~200GB → 2–3× H200 |

**Rule of thumb:** If your team primarily writes code and budget is a constraint → Option 1. If you're running agents, handling multi-turn complex instructions, or need frontier-tier instruction following → Option 2.

---

## Throughput & Economics

> Numbers below are from community benchmarks and RunPod on-demand pricing. Estimates are marked — treat them as planning baselines, not guarantees.

### Qwen3-Coder-Next (80B/3B active) — Throughput Data

| Source | Setup | Generation tok/s | Prompt tok/s |
|---|---|---:|---:|
| Community benchmark | vLLM FP8, H100 (80GB) | ~120 | — |
| Community benchmark | vLLM FP8, DGX Spark | ~44 | — |
| Community benchmark | llama.cpp Q8, 4090+3090 | ~15 | ~221 |
| Community benchmark | llama.cpp Q8 (optimized) | ~46 | — |
| **Estimate** | **vLLM 4-bit, A100 80GB** | **~40–60** | — |

### Qwen3.5-397B-A17B (397B/17B active) — Throughput Data

| Source | Setup | Generation tok/s | Notes |
|---|---|---:|---|
| apidog article | 4-bit GGUF, single A100 | ~28 | — |
| Model card | vs Qwen3-Max @ 256K | — | 19× faster decoding |
| Model card | vs Qwen3-235B-A22B | — | 7.2× faster |
| **Estimate** | **vLLM/SGLang FP8, 3× H200** | **~40–70** | MoE efficiency: only 17B active |
| **Estimate** | **SGLang FP8 + MTP speculative** | **~60–140** | 1.5–2× multiplier from NEXTN algorithm |
| **Estimate** | **vLLM 4-bit, 2× H200** | **~35** | AWQ quantized |

### $/1M Output Tokens

> At 100% utilization (theoretical ceiling). Real-world assumes 30–50% GPU utilization.

| Setup | Hourly cost | Raw tok/s | Tok/hr | $/1M tokens (100% util) | $/1M tokens (real-world) |
|---|---:|---:|---:|---:|---:|
| **Coder-Next, 1× A100** | $1.19/hr | ~50 | 180K | ~$6.61 | ~$13–22 |
| **Coder-Next, 2× L40S (TP=2)** | $1.38/hr | ~80 | 288K | ~$4.79 | ~$10–16 |
| **397B, 3× H200 FP8** | $10.77/hr | ~50 | 180K | ~$59.83 | ~$80–133 |
| **397B, 3× H200 FP8 + MTP (1.5×)** | $10.77/hr | ~75 | 270K | ~$39.89 | ~$53–89 |
| **397B, 2× H200 4-bit** | $7.18/hr | ~35 | 126K | ~$56.98 | ~$114–190 |

**For comparison — Claude API output pricing:**

| Model | $/1M output | $/1M input |
|---|---:|---:|
| Claude Opus 4.6 | $75 | $15 |
| Claude Sonnet 4.5 | $15 | $3 |
| Claude Haiku 4.5 | $1.25 | $0.25 |

> At real-world utilization, Coder-Next on A100 (~$13–22/M) is cheaper than Claude Sonnet 4.5. The 397B at real-world costs (~$80–133/M) is competitive with or cheaper than Claude Opus 4.6 ($75/M at 100% output).

### Concurrent Users Estimate

> **Assumptions:** average agent session = ~20 tok/s sustained during active generation; 70% of the time agents are idle (waiting, processing, not generating).

| Setup | Raw tok/s | Concurrent active streams | Effective concurrent users (70% idle) |
|---|---:|---:|---:|
| **1× A100** (Coder-Next 4-bit) | ~50 | 2–3 | ~7–10 |
| **2× L40S** (Coder-Next 4-bit TP=2) | ~80 | 3–4 | ~10–13 |
| **3× H200** (397B FP8) | ~50 | 2–3 | ~7–10 |
| **3× H200** (397B FP8 + MTP) | ~75 | 3–4 | ~10–13 |
| **2× H200** (397B 4-bit) | ~35 | 1–2 | ~5–7 |

---

## Option 1 — Budget: Qwen3-Coder-Next (~$1k–1.5k/mo)

### Why

- **80B total / 3B active** — hybrid MoE: Gated DeltaNet + Gated Attention + MoE, 512 experts
- **70.6% SWE-Bench Verified** — best open-weight value-per-dollar for coding
- **Best pass@5 on SWE-rebench** among all open-source models
- **256K context** (native, extendable)
- Fits on **one A100 80GB** at 4-bit — cheapest serious GPU option
- Apache 2.0

### VRAM

| Precision | VRAM | Notes |
|---|---:|---|
| FP8 | ~80GB | Fits H100 80GB — barely. No KV cache room. |
| 4-bit | ~40–46GB | Fits A100 80GB with ~34GB free for KV. **Use this.** |

### Hardware Options

**Option A — 1× A100 80GB (~$857–1,109/mo) ✅ BEST VALUE**

```
VRAM: 80GB total
4-bit weights: ~46GB
KV cache: ~34GB free
Context: comfortable 32K, up to ~64K with care
Concurrency: 1–2 requests
```

Cheapest option that runs the model comfortably. Plenty of KV headroom at 32K context. Best value for coding-focused teams running moderate concurrency.

**Option B — 2× L40S 48GB (~$1,000–1,066/mo) ✅ MORE THROUGHPUT**

```
VRAM: 96GB total (2× 48GB)
4-bit weights: ~46GB split across 2 GPUs
KV cache: ~50GB combined free
Context: 65K+ comfortable
Concurrency: 2–4 requests
```

TP=2 across two L40S gives meaningfully better throughput and more KV headroom for longer context. ~$150–200/mo more than a single A100 — worth it if you're handling multiple concurrent requests or want longer context reliably.

**Option C — 1× H100 80GB (~$1,937–2,045/mo) ⚠️ OVER BUDGET**

```
VRAM: 80GB total
FP8 weights: ~80GB (tight — almost no KV room)
4-bit weights: ~46GB with ~34GB KV
```

Faster than A100 (HBM3 vs HBM2e), but ~2× the cost for this model. Can run FP8 for full quality with minimal context, or 4-bit with good headroom. Only choose this if H100 availability is better and you're willing to go over the $1.5k target.

### Serving

**1× A100 (or any single GPU):**

```bash
vllm serve Qwen/Qwen3-Coder-Next \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1
```

**2× L40S (TP=2):**

```bash
vllm serve Qwen/Qwen3-Coder-Next \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 2
```

### LiteLLM Config

Single model serves all tiers — no routing complexity:

```yaml
# litellm_config.yaml
model_list:
  - model_name: haiku
    litellm_params:
      model: openai/qwen3-coder-next
      api_base: http://localhost:8001/v1
      api_key: none

  - model_name: sonnet
    litellm_params:
      model: openai/qwen3-coder-next
      api_base: http://localhost:8001/v1
      api_key: none

  - model_name: opus
    litellm_params:
      model: openai/qwen3-coder-next
      api_base: http://localhost:8001/v1
      api_key: none

litellm_settings:
  drop_params: true
  set_verbose: false
```

```bash
litellm --config litellm_config.yaml --port 8000
```

---

## Option 2 — Frontier: Qwen3.5-397B-A17B (~$5k–8k/mo)

### TL;DR

**Qwen3.5-397B-A17B** on **3× H200 141GB** — single model for everything, ~$7,755/mo:

- 397B total params, 17B active per token (sparse MoE — fast as a 17B, smart as a 397B)
- FP8 precision, tensor parallel across 3 GPUs — full quality, maximum throughput
- Beats Claude Opus 4.5 on instruction following (IFBench 76.5 vs 58.0) and agentic tasks
- 262K native context, extensible to 1M with YaRN
- Apache 2.0 — fully open-weight, no API keys, no rate limits

> **Tradeoff acknowledged:** This setup costs more than the 3-tier approach. You get one best-in-class open-weight model for everything instead of juggling haiku/sonnet/opus routing. If coding peak performance (LCB 90%+) is your only metric, Gemini-3 Pro edges it. If instruction following and agentic reliability matter more — Qwen3.5-397B wins.

### Why Qwen3.5-397B-A17B

#### Benchmarks (vs frontier closed models)

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

#### Architecture Highlights

- **Sparse MoE:** 512 experts, 10 routed + 1 shared active per token → 17B active params, 397B total. Throughput of a 17B model, quality of a much larger one.
- **Gated Delta Networks:** Novel SSM-hybrid architecture replacing standard attention in some layers — better long-context efficiency.
- **Multi-Token Prediction (MTP):** Built-in speculative decoding target. SGLang exploits this with NEXTN algorithm for measurable throughput gains.
- **262K native context:** Extensible to ~1M tokens via YaRN (4× factor).
- **Thinking mode:** `<think>...</think>` enabled by default. Pass `enable_thinking=False` via the API to disable for latency-sensitive paths.
- **Multimodal:** Vision + language via early fusion. One model handles text and images.
- **201 languages:** Multilingual out of the box.
- **License:** Apache 2.0. Use commercially, modify freely, no royalties.

### Hardware Options

> VRAM math first, then pricing. Don't guess — get this wrong and the model doesn't load.

#### VRAM Requirements

| Precision | Bytes/param | Total weights (397B) | Notes |
|---|---:|---:|---|
| BF16 | 2 | **~794GB** | Needs 6× H200 — impractical |
| FP8 | 1 | **~397GB** | 3× H200 or 5× H100 |
| 4-bit AWQ/GGUF | 0.5 | **~200GB** | 2× H200 or 3× H100 |

> These are **weight-only** estimates. Add KV cache + activations on top. Rule of thumb: weights × 1.15–1.25 total at moderate concurrency.

#### Option A — 3× H200 FP8 ✅ Recommended (high perf)

```
3× H200 141GB = 423GB total VRAM
FP8 weights:    ~397GB
KV headroom:    ~26GB (modest — use shorter max_model_len or limit concurrency)
─────────────────────────────────────────────────────
Cost: 3 × $2,585 = ~$7,755/mo
```

**Why:** Full FP8 quality (indistinguishable from BF16 in practice). Tensor parallel across 3 GPUs with NVLink. Best throughput via SGLang MTP speculative decoding. KV cache is tight — keep `--max-model-len` at 131072 (128K) unless you need longer.

**Serving:** SGLang `--tp-size 3` or vLLM `--tensor-parallel-size 3`

#### Option B — 2× H200 4-bit ⚖️ Budget

```
2× H200 141GB = 282GB total VRAM
4-bit weights:  ~200GB
KV headroom:    ~82GB (comfortable — 262K context works)
─────────────────────────────────────────────────────
Cost: 2 × $2,585 = ~$5,170/mo
```

**Why:** Significant cost reduction (~$2.6k/mo savings). 4-bit AWQ quality loss is measurable but small on instruction-following tasks. KV cache headroom is actually better than Option A — you can run full 262K context comfortably. Good choice if budget matters more than peak benchmark scores.

**Serving:** vLLM `--tensor-parallel-size 2` with AWQ quantized checkpoint (e.g., `Qwen/Qwen3.5-397B-A17B-AWQ`)

#### Option C — 4× H100 4-bit ❌ Skip (expensive for less)

```
4× H100 80GB = 320GB total VRAM
FP8 weights:   ~397GB — DOES NOT FIT (320 < 397)
4-bit weights: ~200GB — fits with 120GB KV headroom
─────────────────────────────────────────────────────
Cost: 4 × ~$2,000 = ~$8,000/mo
```

**Why not:** More expensive than Option A (~$245/mo more) while running 4-bit instead of FP8. Worse quality AND higher cost. Only useful if H200s are unavailable on RunPod.

### Serving Setup

#### SGLang — Recommended

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

#### vLLM

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

#### Extended context via YaRN (~1M tokens)

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3.5-397B-A17B \
  --tensor-parallel-size 3 \
  --reasoning-parser qwen3 \
  --hf-overrides '{"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}}}' \
  --max-model-len 1010000
```

> YaRN at 1M context is memory-intensive. KV cache at 1M tokens will overflow a 3× H200 setup with any concurrency. Use only for single-request long-doc workloads, scale down `--max-model-len` for production.

#### Disable thinking mode per-request

The model defaults to `<think>...</think>` reasoning. Disable for fast, latency-sensitive paths:

```python
# Via OpenAI-compatible API
response = client.chat.completions.create(
    model="qwen3.5-397b",
    messages=[...],
    extra_body={"enable_thinking": False}  # skip CoT, faster response
)
```

#### LiteLLM Config

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

### Practical guidance for Option 2A (3× H200 FP8)

| Context length | Concurrent seqs | KV cache est. | Fits in 26GB headroom? |
|---|---:|---:|---|
| 32K | 1–2 | ~10–20GB | ✅ comfortable |
| 128K | 1–2 | ~40–80GB | ⚠️ reduce `mem-fraction-static` |
| 262K | 1 | ~80–120GB | ❌ need to reduce weight fraction |

> At 262K context on Option 2A: lower `--mem-fraction-static` to 0.65–0.70 to give KV cache more room. Accept lower concurrent throughput.

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

**Option 1 vs Option 2 coding quality:** Qwen3-Coder-Next beats Qwen3.5-397B on SWE-Bench Verified (70.6% vs 68.1%). For pure coding tasks, the cheaper model actually wins. The 397B earns its premium on instruction following, multi-turn agents, and multimodal workloads.

**Option 2 cost:** $7,755/mo for Option 2A is ~3× the budget option. You're paying for frontier-tier open-weight instruction following and agentic reliability instead of tiered routing.

**Benchmark dates:** Numbers are from model cards at release. The landscape moves fast — these will be outdated within months.

**IFBench / MultiChallenge:** These matter for agent instruction following, but they're not the only thing. Run your own evals on your actual workload before committing.

**Thinking mode overhead:** `<think>...</think>` mode adds latency. For high-throughput agent loops, disable per-request with `enable_thinking=False`. For complex reasoning tasks, leave it on.

**Multi-GPU ops:** Running tensor-parallel across multiple H200s requires a multi-GPU pod (not separate single-GPU pods) — inter-GPU bandwidth matters. Same applies to 2× L40S for Option 1B.

**4-bit quality:** AWQ 4-bit is measurably lower quality than FP8, but for instruction following specifically the gap is smaller than for coding benchmarks.

**No haiku/sonnet routing:** Both options send every request to the same model. For workloads where 90% of requests are trivial tool calls, the MoE active-param count (3B for Option 1, 17B for Option 2) is usually fast enough without tiering.
