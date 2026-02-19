# LLM Hosting — Open-Weight Coding Models

## TL;DR

**For most teams:** Run **Qwen3-Coder-Next** on 1× A100 80GB (~$857–1,109/mo) or 2× L40S (~$1,000–1,066/mo). Best coding quality per dollar in open-weight. If you need the absolute highest SWE-Bench score and can spend $2.6–5.2k/mo, go **MiniMax M2.5**. Forget self-hosting **GLM-5** — it needs 8× H100 minimum; use the API.

---

## Model Landscape

| | **Qwen3-Coder-Next** | **MiniMax M2.5** | **GLM-5** |
|---|---|---|---|
| **Total params** | 80B | 230B | 744B |
| **Active params** | 3B | 10B | 40B |
| **Architecture** | MoE hybrid (Gated DeltaNet + GDA + MoE) | MoE | MoE (MLA + DSA) |
| **SWE-Bench Verified** | 70.6% | **80.2% 🥇** | 77.8% |
| **BrowseComp** | — | **76.3%** | — |
| **Context** | 256K | 196K | 203K |
| **License** | Apache 2.0 | Apache 2.0 | Open weight (check license) |
| **VRAM (FP8)** | ~80GB | ~230GB | ~800GB |
| **VRAM (4-bit)** | ~40–46GB | ~115GB | ~370GB |
| **VRAM (Q3_K)** | — | ~101GB | ~241GB |
| **Min GPU (practical)** | 1× A100 80GB | 1× H200 141GB (4-bit) | 8× H100 80GB |
| **Min $/mo (self-hosted)** | **~$857** | ~$2,585 | ~$15,000+ |
| **Fits team budget?** | ✅ Yes | ✅ Yes (if $2.6–5.2k/mo OK) | ❌ API only |

---

## Throughput & Economics

### Tokens per Second (measured)

> Sources noted inline. "Estimated" = extrapolated from comparable hardware.

| Model | Setup | Precision | tok/s | Source |
|---|---|---|---:|---|
| Qwen3-Coder-Next | 1× H100 80GB | FP8 | ~120 | community benchmark |
| Qwen3-Coder-Next | 1× A100 80GB | 4-bit | ~50 | estimated |
| Qwen3-Coder-Next | 2× L40S 48GB (TP=2) | 4-bit | ~80 | estimated |
| Qwen3-Coder-Next | DGX Spark (single node) | 4-bit | ~44 | community benchmark |
| MiniMax M2.5 | 8× RTX Pro 6000 48GB | FP8 | ~70 (single stream) | community benchmark |
| MiniMax M2.5 | 8× RTX Pro 6000 48GB | FP8 | ~122 (2 connections) | community benchmark |
| MiniMax M2.5 | FelloAI deployment | FP8 | ~100 | vendor claim |
| MiniMax M2.5 | 1× H200 141GB | Q3_K | ~25 | community benchmark |
| MiniMax M2.5 | 1× H200 141GB | 4-bit | ~30–40 | estimated |
| GLM-5 | 8× H100 80GB | FP8 | ~30–50 | estimated |

> MiniMax M2.5 on 8× RTX Pro 6000 can serve **~9 sessions in parallel** at full 200K context (FP8).

### Cost per 1M Output Tokens

> Based on RunPod pod pricing. GPU cost ÷ (tok/s × 3600) × 1,000,000.

| Setup | $/hr | tok/s | $/M @ 100% util | $/M real-world (30–50% util) |
|---|---:|---:|---:|---:|
| Coder-Next — 1× A100 4-bit | $1.19 | ~50 | ~$6.61 | ~$13–22 |
| Coder-Next — 2× L40S TP=2 | $1.38 | ~80 | ~$4.79 | ~$10–16 |
| M2.5 — 1× H200 4-bit | $3.59 | ~35 | ~$28.49 | ~$57–95 |
| M2.5 — 2× H200 FP8 | $7.18 | ~70 | ~$28.49 | ~$57–95 |
| GLM-5 — 8× H100 FP8 | ~$21.50 | ~40 | ~$149 | ~$298–498 |
| — | — | — | — | — |
| **Claude API comparison** | | | | |
| Sonnet 4.5 (API) | — | — | $15/M output | — |
| Opus 4.6 (API) | — | — | $75/M output | — |
| Haiku 4.5 (API) | — | — | $1.25/M output | — |

> **Verdict:** Coder-Next on 2× L40S at $4.79–16/M output is cheaper than Haiku 4.5 API at heavy use. M2.5 at $28–95/M output sits between Sonnet and Opus API pricing. GLM-5 self-hosted is never economical.

### Concurrent Usage

> Assumes 20 tok/s per active stream; 70% of users idle at any moment.

| Setup | tok/s | Active streams | Effective users (70% idle) |
|---|---:|---:|---:|
| 1× A100 — Coder-Next 4-bit | ~50 | 2–3 | ~7–10 |
| 2× L40S — Coder-Next TP=2 | ~80 | 3–4 | ~10–13 |
| 1× H200 — M2.5 4-bit | ~35 | 1–2 | ~5–7 |
| 2× H200 — M2.5 FP8 | ~70 | 3–4 | ~10–13 |
| 8× H100 — GLM-5 FP8 | ~40 | 2 | ~7 |

---

## Recommendation: Qwen3-Coder-Next

**70.6% SWE-Bench Verified. Best coding value per dollar. Fits one GPU.**

- 80B total / 3B active (sparse MoE — throughput of a 3B, quality of 80B)
- 256K context (native, extendable)
- Fits on a single A100 80GB or 2× L40S at 4-bit — no multi-machine complexity
- Apache 2.0

### VRAM

| Precision | VRAM | Notes |
|---|---:|---|
| FP8 | ~80GB | Fits H100 80GB — no KV room. Not recommended. |
| 4-bit | ~40–46GB | Fits A100 80GB with ~34GB KV. **Use this.** |

### Hardware Options

**Option A — 1× A100 80GB (~$857–1,109/mo) ✅ Best value**

```
VRAM:        80GB total
4-bit weights: ~46GB
KV cache:    ~34GB free
Context:     comfortable 32K, up to ~64K with care
Concurrency: 1–2 requests, ~7–10 effective users
```

Cheapest serious option. Solo team or small squad doing sequential coding tasks. This is the default call.

**Option B — 2× L40S 48GB (~$1,000–1,066/mo) ✅ More throughput**

```
VRAM:        96GB total (2× 48GB)
4-bit weights: ~46GB split across 2 GPUs
KV cache:    ~50GB combined free
Context:     65K+ comfortable
Concurrency: 3–4 requests, ~10–13 effective users
```

~$150–200/mo more than a single A100. Worth it for teams with multiple concurrent users or longer context needs. Better tok/s (est. ~80) than a single A100 (~50).

**Option C — 1× H100 80GB (~$1,937–2,045/mo) ⚠️ Over budget for this model**

```
VRAM:        80GB total
FP8 weights: ~80GB (no KV room)
4-bit weights: ~46GB with ~34GB KV
```

Faster than A100 (HBM3), but ~2× the cost. The performance gain on Coder-Next doesn't justify it — you could upgrade to M2.5 instead for a comparable spend. Only pick this if H200s and A100s are both unavailable.

### Serving

**Option A — 1× A100 (single GPU):**

```bash
vllm serve Qwen/Qwen3-Coder-Next \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1
```

**Option B — 2× L40S (TP=2):**

```bash
vllm serve Qwen/Qwen3-Coder-Next \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 2
```

### LiteLLM Config

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

**OpenClaw integration:**
- `base_url`: `http(s)://<host>:8000/v1`
- `api_key`: LiteLLM `master_key` (if set)
- Model aliases: `haiku`, `sonnet`, `opus` → all route to Coder-Next

---

## Alternative: MiniMax M2.5

**80.2% SWE-Bench Verified — highest open-weight coding score. Also 76.3% BrowseComp (agentic). Pay $2.6–5.2k/mo for the quality jump.**

- 230B total / 10B active (MoE)
- 196K context
- Apache 2.0
- Notable: "spec-writing tendency" — model tends to plan/outline structure before writing code. Works well for greenfield features, less ideal for surgical patches.

### VRAM

| Precision | VRAM | Notes |
|---|---:|---|
| FP8 | ~230GB | 2× H200 minimum (barely), 4× H100 for headroom |
| 4-bit | ~115GB | 1× H200 141GB (fits) |
| Q3_K | ~101GB | 1× H200 141GB (fits, heavy quant) |

> Community reports: peak 728GB observed on 8× RTX Pro 6000 at full 200K context FP8. Plan accordingly.

### Hardware Options

**Option A — 1× H200 141GB 4-bit (~$2,585/mo) ✅ Entry point**

```
VRAM:        141GB total
4-bit weights: ~115GB
KV cache:    ~26GB free
Context:     tight — keep max_model_len ≤ 32K for reliable concurrency
Concurrency: 1–2 requests, ~5–7 effective users
Throughput:  ~30–40 tok/s (estimated)
```

Fits. KV cache is tight. Fine for teams with sequential workloads. Don't push 196K context here — you'll OOM.

**Option B — 2× H200 141GB FP8 (~$5,170/mo) ✅ Recommended for M2.5**

```
VRAM:        282GB total
FP8 weights: ~230GB
KV cache:    ~52GB free
Context:     80K–100K comfortable
Concurrency: 3–4 requests, ~10–13 effective users
Throughput:  ~70 tok/s (community benchmark, FP8)
```

Full FP8 quality. Meaningfully better throughput than the 4-bit option. This is the right setup if you're choosing M2.5.

**Option C — 4× H100 80GB FP8 (~$7,700–8,200/mo) ❌ Skip**

More expensive than 2× H200 for effectively the same setup. H200 has larger VRAM per GPU and better bandwidth. Pass unless H200 isn't available.

### Serving

```bash
# Option A: 1× H200, 4-bit (use a quantized checkpoint)
vllm serve MiniMaxAI/MiniMax-M2.5 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1 \
  --tool-call-parser minimax_m2

# Option B: 2× H200, FP8
vllm serve MiniMaxAI/MiniMax-M2.5 \
  --max-model-len 98304 \
  --gpu-memory-utilization 0.88 \
  --tensor-parallel-size 2 \
  --tool-call-parser minimax_m2
```

> `--tool-call-parser minimax_m2` is required for correct function-calling behavior with M2.5.

### LiteLLM Config

```yaml
model_list:
  - model_name: haiku
    litellm_params:
      model: openai/minimax-m2.5
      api_base: http://localhost:8001/v1
      api_key: none

  - model_name: sonnet
    litellm_params:
      model: openai/minimax-m2.5
      api_base: http://localhost:8001/v1
      api_key: none

  - model_name: opus
    litellm_params:
      model: openai/minimax-m2.5
      api_base: http://localhost:8001/v1
      api_key: none

litellm_settings:
  drop_params: true
  set_verbose: false
```

---

## Reference: GLM-5

**77.8% SWE-Bench Verified. 744B/40B active. Don't self-host. Use the API.**

GLM-5 uses MLA + DeepSeek Sparse Attention (DSA) for efficient KV cache — smart architecture, but it doesn't change the economics. 744B total params means you need enormous GPU clusters.

### VRAM

| Precision | VRAM | Practical setup |
|---|---:|---|
| BF16 | ~1.5TB | 12× H100 80GB minimum |
| FP8 | ~800GB | 8× H100 80GB (barely) or 6× H200 |
| 4-bit | ~370GB | 3× H200 or 5× H100 |
| Q3_K | ~241GB | 2× H200 (~282GB — barely fits) |

### Self-Hosting Costs (all painful)

| Setup | $/mo | Quality | Verdict |
|---|---:|---|---|
| 8× H100 FP8 | ~$15,000–16,000 | Full | ❌ Way over budget |
| 2× H200 Q3_K | ~$5,170 | Degraded | ❌ Heavy quant, marginal fit, questionable quality |
| 6× H200 FP8 | ~$15,510 | Full | ❌ Ridiculous |

At $5.2k/mo you could run M2.5 FP8 (2× H200) and get 80.2% SWE-Bench with full quality. GLM-5 Q3_K at the same cost is worse quality, barely fits, and scores 77.8%. There's no case for self-hosting GLM-5 under $15k/mo.

### API Access (use this instead)

| Provider | Notes |
|---|---|
| **OpenRouter** | Available under `thudm/glm-z1-32b` and GLM-5 variants |
| **Vertex AI** | GLM-5 via Model Garden |
| **BigModel** | First-party ZhipuAI platform — best SLA for GLM models |

Use GLM-5 via API when you need its specific capabilities (long-context, DSA efficiency) without the hosting overhead.

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

## Cloud Provider Comparison

> On-demand pricing, Feb 2026. Goal: find the cheapest reliable option for **1–2 GPU inference**.

### Per-GPU/hr Pricing (On-Demand)

| GPU | **Vast.ai** | **RunPod** | **Lambda** | **AWS** | **GCP** | **Azure** |
|---|---|---|---|---|---|---|
| **H200 141GB** | $2.19/hr | $3.59/hr | — | ~$4.50–5.00/hr* | — | — |
| **H100 80GB** | $1.60–1.65/hr | $1.99–2.84/hr | $2.49/hr | $3.90/hr** | $3.00/hr | $6.98/hr |
| **A100 80GB** | $0.52–0.80/hr | $1.19–1.54/hr | $1.50/hr | $4.10/hr** | — | — |
| **L40S 48GB** | ~$0.40–0.50/hr | $0.69–0.74/hr | — | — | — | — |
| **RTX 4090 24GB** | ~$0.20–0.30/hr | $0.68–0.74/hr | — | — | — | — |

\*AWS p5e.48xlarge is 8× H200 only (~$36–40/hr total), per-GPU estimated
\*\*AWS sells 8-GPU nodes only: p5.48xlarge = $31.21/hr (8× H100), p4d.24xlarge = $32.77/hr (8× A100)

**Notes:**
- **Vast.ai** is a marketplace — prices fluctuate with supply/demand. Cheapest listings shown; quality and uptime vary.
- **RunPod** has "community cloud" (cheaper) and "secure cloud" (more reliable) tiers. Prices above are community cloud.
- **Lambda** pricing is straightforward with zero egress fees, but GPU availability is limited — H100s sell out fast.
- **AWS/GCP/Azure** require an 8-GPU minimum for H100/H200 — you cannot rent a single GPU.
- **Thunder Compute** offers H100 at $1.38/hr (emerging provider, worth watching).
- **VERDA** offers H100 at $0.80/hr and A100 at $0.45/hr (lowest found; less established, caveat emptor).

### Monthly Cost for Our Setups

| Setup | **Vast.ai** | **RunPod** | **Lambda** | **AWS** |
|---|---|---|---|---|
| 1× A100 80GB (Coder-Next) | ~$374–576 | ~$857–1,109 | ~$1,080 | ~$2,952*** |
| 1× H100 80GB (Coder-Next FP8) | ~$1,152–1,188 | ~$1,433–2,045 | ~$1,793 | ~$22,464*** |
| 1× H200 141GB (M2.5 4-bit) | ~$1,577 | ~$2,585 | — | ~$26–29k*** |
| 2× H200 141GB (M2.5 FP8) | ~$3,154 | ~$5,170 | — | ~$26–29k*** |

\*\*\*AWS forces 8-GPU nodes — you pay for all 8 to use 1–2. Prices above are the **full node cost** (p5.48xlarge, p5e.48xlarge, p4d.24xlarge). This is not a typo.

### Why Hyperscalers Don't Make Sense for This

#### 1. 8-GPU minimum

AWS, GCP, and Azure sell H100/H200 as full 8-GPU nodes:
- AWS: `p5.48xlarge` (8× H100 SXM, $31.21/hr), `p5e.48xlarge` (8× H200, ~$36–40/hr), `p4d.24xlarge` (8× A100, $32.77/hr)
- GCP: `a3-highgpu-8g` (8× H100)
- Azure: `ND H100 v5` series

If you need 1–2 GPUs, you're paying for 6–7 idle GPUs. At $31/hr for 8× H100, renting "one H100" on AWS costs $3.90/hr equivalent — but you're actually paying for the whole node.

#### 2. Hidden costs

| Cost | AWS | RunPod | Vast.ai |
|---|---|---|---|
| **Egress** | $0.09/GB (adds 50–100% to bill at model weight downloads + inference traffic) | Free or minimal | Free or minimal |
| **Storage IOPS** | Premium for provisioned IOPS (model weights need fast NVMe) | Included in pod pricing | Included |
| **NAT gateway** | Per-hour + per-GB toll | N/A | N/A |
| **Minimum billing** | Per-hour minimum | Per-second | Per-second |

A single H200 model download (115GB for M2.5 4-bit) costs $10.35 in AWS egress on the way out. On RunPod or Vast.ai: $0.

#### 3. When hyperscalers make sense

- Enterprise compliance requirements (SOC2, HIPAA, FedRAMP)
- 8+ GPU training clusters where you fill the whole node
- Reserved/spot instances at scale with existing AWS credits burning
- Tight integration with existing AWS infrastructure (VPCs, IAM, etc.)

For **1–2 GPU inference on open-weight models**: none of these apply.

### Recommendation

For our use case (1–2 GPU inference):

- **Best price — Vast.ai**: marketplace rates are 40–60% cheaper than RunPod. Tradeoff: variable quality, machines can vanish, spot-like availability. Use for dev/eval, not production.
- **Best reliability — RunPod**: consistent pricing, good API (`runpodctl`), secure cloud option for compliance-sensitive use. Default choice for production.
- **Best for experimenting — Lambda**: clean UX, zero egress fees, straightforward pricing. Downside: GPU availability is tight; H100s frequently unavailable on-demand.
- **Avoid — AWS/GCP/Azure**: wrong product for this workload. Built for enterprises running full 8-GPU clusters. You'll pay 3–8× more per effective GPU and deal with substantially more billing complexity.

---

## VRAM Reality Check

<details>
<summary><strong>How vLLM/SGLang use GPU memory (expand)</strong></summary>

GPU memory = **weights + KV cache + activations + overhead**. Quantization only shrinks weights.

### Weight memory (rule of thumb)

| Precision | Bytes/param | Coder-Next (80B) | M2.5 (230B) | GLM-5 (744B) |
|---|---:|---:|---:|---:|
| BF16 | 2 | ~160GB | ~460GB | ~1,488GB |
| FP8 | 1 | ~80GB | ~230GB | ~744GB |
| 4-bit | 0.5 | ~40GB | ~115GB | ~372GB |
| Q3_K | ~0.375 | ~30GB | ~86GB | ~279GB |

> For MoE models: only active params run computation, but **all expert weights must be loaded into VRAM**. You load the full model, compute with a fraction. No shortcut.

### KV cache grows with context × concurrency

```
KV cache ≈ 2 × layers × heads × head_dim × seq_len × batch_size × dtype_bytes
```

A rough rule: assume 0.5–1GB per 1K tokens per concurrent sequence for large MoE models. At 32K context, 4 concurrent sequences ≈ 64–128GB KV cache.

### Validate before deploying

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Watch memory during load
watch -n 2 nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
```

**Sources:**
- vLLM memory: https://docs.vllm.ai/en/latest/configuration/conserving_memory/
- SGLang docs: https://docs.sglang.ai/
- Qwen3-Coder-Next model card: https://huggingface.co/Qwen/Qwen3-Coder-Next
- MiniMax M2.5 model card: https://huggingface.co/MiniMaxAI/MiniMax-M2.5

</details>

---

## Caveats

**SWE-Bench Verified scores:** M2.5 at 80.2% leads, but run your own evals on your actual codebase. Benchmark composition matters — M2.5 may over-index on greenfield vs. patch tasks.

**MiniMax M2.5 "spec-writing tendency":** The model plans features and outlines structure before writing code. Great for architecture and greenfield. Less ideal if you need surgical single-function patches — it'll write you a 500-line refactor plan first.

**GLM-5 API pricing:** Varies by provider. Check BigModel (ZhipuAI) for first-party rates; OpenRouter for comparison. Don't self-host unless you have an existing 8× H100 cluster you're already paying for.

**Throughput numbers:** Community benchmarks vary significantly by batch size, context length, prompt structure, and GPU state. Treat all tok/s estimates as order-of-magnitude guidance, not SLAs.

**Multi-GPU ops:** Tensor-parallel across multiple GPUs requires a multi-GPU pod (not separate single-GPU pods) — inter-GPU NVLink bandwidth is critical. Verify NVLink availability on RunPod pods before ordering.

**4-bit quality:** AWQ/GGUF 4-bit is measurably lower quality than FP8. For M2.5 specifically, prefer FP8 (2× H200) if the budget allows — the quality gap on a 230B model is more pronounced than on an 80B model.

**Benchmark dates:** Numbers from model cards at release. This landscape moves fast.
