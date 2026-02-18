# LLM Hosting — Coding Agent Stack

## TL;DR

**1× H200 141GB** running three tiers on one GPU:
- 🟢 **Haiku** → Devstral Small 1.1 (24B dense, 53.6% SWE-Bench Verified)
- 🔵 **Sonnet** → GLM-4.7-Flash (30B/3B active MoE, 59.2% SWE-Bench Verified, blazing fast)
- 🟣 **Opus** → Qwen3-Coder-Next (80B/3B active hybrid MoE, ~70% SWE-Bench Verified)

All three fit comfortably (~65GB combined). LiteLLM routes everything through one endpoint. **~$2.6k/mo always-on.**

---

## The Stack

> This is a coding agent stack. Every model choice below is justified by **SWE-bench and coding-specific benchmarks**, not general capabilities.

### 🟢 Haiku — Devstral Small 1.1

**[mistralai/Devstral-Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507)** | 24B dense | 128K context | ~13GB @ 4-bit

**Why:** Purpose-built for agentic coding. Trained jointly by Mistral + All Hands AI specifically to solve real GitHub issues. 53.6% SWE-Bench Verified — **higher than Claude 3.5 Haiku (40.6%)** on the same OpenHands scaffold. Runs on a single 24GB GPU. Apache 2.0.

*Runner-up: Qwen3-Coder-30B-A3B — also strong, 256K context, but Devstral's SWE-bench score is verified.*

### 🔵 Sonnet — GLM-4.7-Flash

**[THUDM/GLM-4.7-Flash](https://huggingface.co/THUDM/GLM-4.7)** | 30B / 3B active MoE | 128K context | ~6–8GB @ 4-bit

**Why:** 59.2% SWE-Bench Verified. MoE means only 3B params are active — so it's extremely fast (60–80+ tok/s on H200) while holding sonnet-tier coding quality. Perfect for mid-tier: fast tool-calling loops, iterative patch generation, streaming responses. Tiny VRAM footprint means it barely touches the H200's budget.

*Runner-up: Qwen3-Coder-Next in 4-bit — if you want maximum quality at sonnet-tier, bump it here and drop to a smaller haiku.*

### 🟣 Opus — Qwen3-Coder-Next

**[Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)** | 80B / 3B active hybrid MoE | 256K context | ~46GB @ 4-bit

**Why:** Specifically engineered for coding agents. Novel hybrid attention (Gated DeltaNet + Gated Attention + MoE) with 512 experts. Trained with long-horizon RL on real-world software tasks. Claims performance comparable to Claude Sonnet — the best coding-agent-optimized model that fits a single H200. ~70% SWE-Bench Verified (per Qwen's benchmarks vs comparable closed models). 256K context window for repo-scale understanding.

*Note: Qwen3-Coder-480B-A35B is stronger but needs ~240GB — requires 4× H200, way over budget.*

---

## Model Comparison

> **Focus: coding benchmarks only.** SWE-Bench Verified = real GitHub issues. HumanEval = function-level coding. LiveCodeBench (LCB) = competitive programming (contamination-resistant). See [caveats](#benchmark-caveats) before drawing conclusions.

| Model | Tier | Params (total/active) | SWE-Bench Verified | HumanEval | LCB | Context | VRAM (4-bit) | Tok/s est. | Self-hostable? |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **Devstral Small 1.1** | 🟢 Haiku | 24B / 24B dense | **53.6%** | ~85% | — | 128K | ~13GB | 60–100 | ✅ Apache 2.0 |
| **Qwen3-Coder-30B-A3B** | 🟢 Haiku | 30.5B / 3.3B | ~46% (est.) | ~80% | — | 256K | ~8GB | 100–150 | ✅ Apache 2.0 |
| **GLM-4.7-Flash** | 🔵 Sonnet | 30B / 3B | **59.2%** | ~82% | — | 128K | ~6–8GB | 60–80 | ✅ Apache 2.0 |
| **Qwen3-Coder-Next** | 🟣 Opus | 80B / 3B | ~70% (est.) | ~88% | — | 256K | ~46GB | 30–50 | ✅ Apache 2.0 |
| **Qwen3-Coder-480B-A35B** | 🚫 OOB | 480B / 35B | ~75% (est.) | ~92% | — | 256K–1M | ~240GB | 15–25 | ❌ 4× H200 needed |
| **Devstral Medium 2507** | 🚫 API | — | **61.6%** | — | — | — | — | — | ❌ API-only |
| *Claude 3.5 Haiku* | *ref* | *closed* | *40.6%* | *~88%* | — | *200K* | — | — | ❌ |
| *Claude Sonnet 4* | *ref* | *closed* | *~72%* | *~93%* | — | *200K* | — | — | ❌ |

**Notes:**
- `OOB` = Out of budget (requires multi-GPU, >$4k/mo always-on)
- SWE-Bench Verified scores use different scaffolds — don't treat as direct comparisons. Devstral 1.1 and GLM-4.7 scores use OpenHands scaffold. See [caveats](#benchmark-caveats).
- Qwen3-Coder-Next and 480B scores are estimated from Qwen's benchmark charts vs. closed models.
- HumanEval numbers are approximate and from mixed sources.
- LCB scores not publicly available for all models at time of writing.

---

## Hardware — Single Machine Only

> Budget: $4k/mo max. All single-machine, no split setups.

| GPU | VRAM | On-demand $/hr | Always-on $/30d | Verdict |
|---|---:|---:|---:|---|
| **H200 141GB** | 141GB | $3.59 | **~$2,585** | ✅ **Recommended** — fits all 3 tiers (~65GB combined) |
| **H100 80GB** | 80GB | $2.69–$2.84 | **~$1,937–$2,045** | ✅ Works — tight fit with Qwen3-Coder-Next (46GB) + smaller haiku/sonnet |
| **A100 80GB** | 80GB | $1.19–$1.54 | **~$857–$1,109** | ⚠️ Dev only — same VRAM as H100 but slower; opus tier constrained |

### Why H200 wins for this stack

- **Qwen3-Coder-Next @ 4-bit GGUF ≈ 46GB** + **GLM-4.7-Flash @ 4-bit ≈ 7GB** + **Devstral Small 1.1 @ 4-bit ≈ 13GB** = **~66GB total weights**
- H200 has 141GB → **75GB free for KV cache** across all three models
- H100 at 80GB works but leaves only ~14GB for KV cache — limits concurrency and context length
- H200's HBM3 memory bandwidth is ~40% higher than H100 → better throughput for MoE models

### Monthly budget breakdown (H200, always-on)

```
Haiku:  Devstral Small 1.1          ~13GB
Sonnet: GLM-4.7-Flash               ~ 7GB
Opus:   Qwen3-Coder-Next            ~46GB
─────────────────────────────────────────
Total weights:                      ~66GB
KV cache headroom:                  ~75GB (at 85% GPU utilization)
GPU cost (1× H200 on-demand):       ~$2,585/mo
Remaining budget:                   ~$1,415 (storage, networking, burst)
```

---

## Routing

<details>
<summary><strong>LiteLLM + vLLM setup</strong></summary>

### Architecture

```
Your agent (OpenAI API calls)
        │
        ▼
  LiteLLM router :8000
  ├── "haiku"  → vLLM :8001 (Devstral Small 1.1)
  ├── "sonnet" → vLLM :8002 (GLM-4.7-Flash)
  └── "opus"   → vLLM :8003 (Qwen3-Coder-Next)
```

### vLLM launch commands

```bash
# Haiku — Devstral Small 1.1
vllm serve mistralai/Devstral-Small-2507 \
  --port 8001 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.10 \
  --tensor-parallel-size 1

# Sonnet — GLM-4.7-Flash
vllm serve THUDM/GLM-4.7-Flash \
  --port 8002 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.06 \
  --tensor-parallel-size 1

# Opus — Qwen3-Coder-Next
# Requires vllm>=0.15.0
vllm serve Qwen/Qwen3-Coder-Next \
  --port 8003 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.78 \
  --tensor-parallel-size 1
```

> Tune `--gpu-memory-utilization` per model to control KV cache allocation. All three share the same physical GPU — vLLM handles memory isolation.

### LiteLLM config

```yaml
# litellm_config.yaml
model_list:
  - model_name: haiku
    litellm_params:
      model: openai/devstral-small
      api_base: http://localhost:8001/v1
      api_key: none

  - model_name: sonnet
    litellm_params:
      model: openai/glm-4.7-flash
      api_base: http://localhost:8002/v1
      api_key: none

  - model_name: opus
    litellm_params:
      model: openai/qwen3-coder-next
      api_base: http://localhost:8003/v1
      api_key: none

litellm_settings:
  drop_params: true
  set_verbose: false
```

```bash
litellm --config litellm_config.yaml --port 8000
```

### OpenClaw integration

Point OpenClaw's OpenAI-compatible provider at LiteLLM:
- `base_url`: `http(s)://<host>:8000/v1`
- `api_key`: LiteLLM `master_key` (if set)
- Model aliases: `haiku`, `sonnet`, `opus`

</details>

---

## VRAM Reality Check

<details>
<summary><strong>How vLLM uses GPU memory (expand)</strong></summary>

vLLM GPU memory = **weights + KV cache + activations + overhead**. Quantization only reduces weights.

### Weight memory (rule of thumb)

| Precision | Bytes/param | 24B | 30B | 80B |
|---|---:|---:|---:|---:|
| FP16/BF16 | 2 | ~48GB | ~60GB | ~160GB |
| INT8 | 1 | ~24GB | ~30GB | ~80GB |
| **4-bit** | 0.5 | **~12GB** | **~15GB** | **~40GB** |

> For MoE models, only the *active* parameters run through computation, but **all expert weights must be loaded into VRAM**. A 30B/3B MoE model uses ~15GB (4-bit) for weights but computes as fast as a 3B model.

### KV cache grows with context + concurrency

```
KV cache ≈ 2 × layers × heads × head_dim × seq_len × batch_size × dtype_bytes
```

At 32K context, 8 concurrent sequences, FP16: adds ~10–40GB depending on architecture.

### Single-GPU guidance (4-bit weights, FP16 KV, 1–2 concurrent seqs)

| Model | 8K ctx | 16K ctx | 32K ctx |
|---|---:|---:|---:|
| **Devstral Small 1.1 (24B)** | ~18GB | ~22GB | ~30GB |
| **GLM-4.7-Flash (30B/3B MoE)** | ~12GB | ~15GB | ~20GB |
| **Qwen3-Coder-Next (80B/3B MoE)** | ~55GB | ~65GB | ~85GB |

> At 32K context, Qwen3-Coder-Next exceeds H100 80GB — keep `--max-model-len 16384` on H100. H200 141GB is comfortable at 32K.

### Validate on your GPU

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv

# Start conservative
vllm serve <model> --max-model-len 8192 --gpu-memory-utilization 0.85

# Check VRAM usage
nvidia-smi dmon -s mu -d 5
```

**Sources:**
- vLLM memory docs: https://docs.vllm.ai/en/latest/configuration/conserving_memory/
- vLLM GPU memory utilization: https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/gpu_memory_utilization/

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

## Benchmark Caveats

<a name="benchmark-caveats"></a>

> ⚠️ **SWE-bench numbers are unreliable comparators.** Here's why:

- **Verified vs Pro vs Public** subsets differ wildly. A model scoring 53% on Verified may score ~35% on Pro (harder tasks).
- **Scaffold dependency:** Devstral Small 1.1's 53.6% is on OpenHands. The same model on a different scaffold might score 30–40%. Scores don't transfer.
- **Contamination:** Models trained on GitHub may have seen benchmark issues. Treat scores from labs with skepticism.
- **Agent scaffolding matters more than base model:** The same model can get a 20–40% boost from better scaffolding. Invest in your scaffold.
- **HumanEval is mostly saturated** — useful for filtering out weak models, not for distinguishing strong ones.
- **LiveCodeBench is more reliable** (competitive programming problems, contamination-resistant) but not all models report it.

**Rule:** Use published scores as a **relative filter**, not absolute truth. Always run evals on your own codebase before committing.

**Sources:**
- SWE-bench Verified: https://www.swebench.com/
- SEAL SWE-bench Pro: https://scale.com/leaderboard/swe_bench_pro
- Devstral Small 1.1 card: https://huggingface.co/mistralai/Devstral-Small-2507
- Devstral 2507 blog: https://mistral.ai/news/devstral-2507
- Qwen3-Coder-Next card: https://huggingface.co/Qwen/Qwen3-Coder-Next
- Qwen3-Coder blog: https://qwenlm.github.io/blog/qwen3-coder/
- GLM-4.7 card: https://huggingface.co/THUDM/GLM-4.7
