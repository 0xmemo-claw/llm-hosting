# LLM Hosting — MiniMax M2.5 Single-Machine Deployment

## TL;DR

**MiniMax M2.5 on 1× H200 141GB** (~$1,577/mo Vast.ai, ~$2,585/mo RunPod). 80.2% SWE-Bench Verified — #1 open-weight coding score. Interleaved thinking. Single GPU. Apache 2.0. Done.

---

## Why MiniMax M2.5

### Specs

| | **MiniMax M2.5** |
|---|---|
| **Total params** | 230B |
| **Active params** | 10B |
| **Architecture** | MoE |
| **Context** | 196K |
| **License** | Apache 2.0 |
| **VRAM (FP8)** | ~230GB |
| **VRAM (4-bit)** | ~115GB |
| **VRAM (Q3_K)** | ~101GB |
| **Min GPU (practical)** | 1× H200 141GB (4-bit) |
| **Min $/mo (self-hosted)** | ~$1,577 (Vast.ai) |

### Benchmarks

| Benchmark | **MiniMax M2.5** | Claude Opus 4.6 | GPT-5.2 | Gemini-3 Pro |
|---|---|---|---|---|
| **SWE-Bench Verified** | **80.2% 🥇** | ~72% | ~70% | ~68% |
| **BrowseComp** | **76.3%** | — | — | — |
| **Multi-SWE-Bench** | **51.3%** | — | — | — |

> SWE-Bench Verified: M2.5 holds the #1 open-weight position. 37% faster than M2.1 on SWE-bench, matching Claude Opus 4.6 speed.

### Why Thinking Mode Matters

M2.5 supports **interleaved thinking** — reasoning is emitted as a separate `reasoning_details` field alongside the main response. Unlike chain-of-thought prompting:

- Thinking state is **maintained across tool calls** — the model remembers its reasoning across an agentic loop
- **Task decomposition is built-in** — the model plans, executes, checks, and revises without prompting
- Reasoning details don't pollute the response stream — clean separation of thinking vs. output
- Works natively with vLLM's `--reasoning-parser deepseek_r1` flag

This matters for coding agents. A model that plans before patching and reasons across multi-step tool use is categorically different from a model that just completes tokens. That's why M2.5 hits 80.2% on SWE-bench.

---

## Hardware

### Recommended: 1× H200 141GB

```
VRAM:          141GB total
4-bit weights: ~115GB
KV cache:      ~26GB free
Context:       comfortable 32-64K, up to 196K with careful tuning
Throughput:    ~30-40 tok/s single stream, ~50-60 tok/s (2 streams)
Monthly cost:  ~$1,545/mo (Vast.ai Type #29607811, US) · ~$2,585/mo (RunPod)
```

**Vast.ai listing:** Type #29607811 (US, 1× H200, $2.146/hr, 99.63% reliability, 18 days max). Best value at 230.2 DLP/$/hr. For production, use Type #30018972 (Iceland, $2.456/hr, 99.904% reliability, 1mo+ max) for higher uptime and longer duration.

Fits the model cleanly at 4-bit. KV cache is tight — keep `max_model_len` at 32-65K for reliable concurrency. Don't push 196K context on a single H200; you'll OOM. Best for teams with sequential workloads or up to ~5-7 effective concurrent users.

---

## Throughput & Economics

### Tokens per Second

| Setup | Precision | tok/s (single stream) | tok/s (multi-stream) | Source |
|---|---|---:|---:|---|
| 1× H200 141GB | 4-bit | ~30-40 | ~50-60 (2 streams) | estimated |
| 1× H200 141GB | Q3_K | ~25 | ~40 (2 streams) | community benchmark |

> Single H200 141GB is the only recommended deployment for M2.5.

### Cost per 1M Output Tokens

| Setup | Provider | $/hr | tok/s | $/M @ 100% | $/M real-world (30-50%) |
|---|---|---:|---:|---:|---:|
| 1× H200 4-bit | Vast.ai (Type #29607811) | $2.15 | ~35 | ~$17.06 | ~$34-57 |
| 1× H200 4-bit | Vast.ai (Type #30018972) | $2.46 | ~35 | ~$19.52 | ~$39-65 |
| 1× H200 4-bit | RunPod | $3.59 | ~35 | ~$28.49 | ~$57-95 |

**Claude API comparison:**

| Tier | $/M output |
|---|---:|
| Opus 4.6 | $75 |
| Sonnet 4.5 | $15 |
| Haiku 4.5 | $1.25 |

> **Verdict:** M2.5 at $35-95/M output (real-world) sits between Sonnet and Opus API pricing — but beats both on SWE-Bench Verified. At heavy sustained use, self-hosting pays off fast.

### Concurrent Users

> Assumes 20 tok/s per active stream; 70% of users idle at any moment.

| Setup | tok/s | Active streams | Effective users (70% idle) |
|---|---:|---:|---:|
| 1× H200 4-bit | ~35 | 1-2 | ~5-7 |
| 1× H200 Q3_K | ~25 | 1 | ~3-5 |

---

## Serving Setup

### vLLM (recommended)

```bash
vllm serve MiniMaxAI/MiniMax-M2.5 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --tool-call-parser minimax_m2 \
  --reasoning-parser deepseek_r1
```

> `--tool-call-parser minimax_m2` is required for correct function-calling behavior.
> `--reasoning-parser deepseek_r1` enables the `reasoning_details` field (interleaved thinking).

### SGLang

```bash
python -m sglang.launch_server \
  --model-path MiniMaxAI/MiniMax-M2.5 \
  --tp-size 1 \
  --mem-fraction-static 0.85 \
  --context-length 65536
```

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

```bash
litellm --config litellm_config.yaml --port 8000
```

**OpenClaw integration:**
- `base_url`: `http(s)://<host>:8000/v1`
- `api_key`: LiteLLM `master_key` (if set)
- Model aliases: `haiku`, `sonnet`, `opus` → all route to M2.5

---

## Alternatives (Reference Only)

### Qwen3-Coder-Next (80B/3B active)

- ❌ No thinking mode
- ❌ No vision
- ✅ 70.6% SWE-Bench Verified — solid second place
- ✅ Fits 1× A100 80GB at 4-bit (~$374-1,109/mo)
- ✅ Best value/$ for pure coding

Pick this if you **only** need coding and want to minimize cost. You'll run at ~$857-1,109/mo (RunPod) vs. $2,585/mo for M2.5, but you give up thinking mode and ~10 SWE-bench points.

```bash
vllm serve Qwen/Qwen3-Coder-Next \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1
```

### Qwen3-235B-A22B (230B/22B active)

- ✅ Thinking mode (reasoning_details field)
- ❌ No vision
- ✅ Slightly lower coding quality than M2.5 (~72-74% SWE-bench)
- ✅ Fits 1× H200 141GB at 4-bit (~118GB) — **single-machine compatible**
- Good fallback if M2.5 has availability issues on Vast.ai

### GLM-5 (744B/40B active) — API Only

- ✅ Thinking + Vision
- ✅ 77.8% SWE-Bench Verified
- ❌ Needs 8× H100 minimum (~$15k+/mo to self-host)
- ❌ **Not single-machine compatible** — requires 8-GPU cluster
- Use via API only at our budget (OpenRouter, Vertex AI, BigModel)

---

## Cloud Provider Comparison

> On-demand pricing, Feb 2026. Goal: find the cheapest reliable option for **1–2 GPU inference**.

### Per-GPU/hr Pricing (On-Demand)

| GPU | **Vast.ai** | **RunPod** | **Lambda** | **AWS** | **GCP** | **Azure** |
|---|---|---|---|---|---|---|
| **H200 141GB** | $2.15/hr (Type #29607811, US) | $3.59/hr | — | ~$4.50–5.00/hr* | — | — |
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

## VRAM Reality Check

<details>
<summary><strong>M2.5 memory breakdown (expand)</strong></summary>

GPU memory = **weights + KV cache + activations + overhead**. Quantization only shrinks weights.

### Weight memory at different precisions

| Precision | Bytes/param | M2.5 (230B) |
|---|---:|---:|
| BF16 | 2 | ~460GB |
| FP8 | 1 | ~230GB |
| 4-bit | 0.5 | ~115GB |
| Q3_K | ~0.375 | ~86-101GB |

### The MoE trap

M2.5 has 230B total parameters but only 10B active per forward pass (MoE routing). This does **not** mean you only load 10B into VRAM. You load all 230B expert weights into GPU memory; MoE routing selects which 10B to activate for each token. No VRAM shortcut.

At 4-bit:
- Weights: ~115GB
- Activations: ~2-3GB (only 10B active, low activation memory)
- KV cache: depends on context × concurrency (see below)
- Overhead: ~3-5GB

On a 141GB H200: 115 + 3 + ~20 (KV) + 4 = **~142GB** — barely fits at 32K context. Push to 65K and you need tight `gpu-memory-utilization` tuning.

### KV cache grows with context × concurrency

```
KV cache ≈ 2 × layers × heads × head_dim × seq_len × batch_size × dtype_bytes
```

Rough rule for M2.5: ~0.5-1GB per 1K tokens per concurrent sequence. At 32K context, 1 concurrent sequence ≈ 16-32GB KV. The remaining ~26GB on a 141GB H200 (after 115GB weights) is tight for anything beyond 1-2 sequences.

### Practical guidance for 1× H200

```bash
# Safe starting point — 32K context, high GPU util
vllm serve MiniMaxAI/MiniMax-M2.5 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90

# More context (65K) — less KV headroom, 1 concurrent sequence
vllm serve MiniMaxAI/MiniMax-M2.5 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90

# Monitor while loading:
watch -n 2 nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
```

**Do not** attempt 196K context on a single H200 at 4-bit. You will OOM.

### Sources
- vLLM memory docs: https://docs.vllm.ai/en/latest/configuration/conserving_memory/
- MiniMax M2.5 model card: https://huggingface.co/MiniMaxAI/MiniMax-M2.5
- Community VRAM benchmarks: peak 728GB observed on 8× RTX Pro 6000 at full 200K context FP8

</details>

---

## Caveats

**M2.5 "spec-writing tendency":** The model plans features and outlines structure before writing code. Works well for greenfield and architecture tasks. Less ideal for surgical single-function patches — it'll draft a refactor plan first. Feature, not bug, for agents.

**37% faster than M2.1:** M2.5 matches Claude Opus 4.6 speed on SWE-bench iterations. Interleaved thinking adds per-token latency but improves overall task success rate on multi-step reasoning.

**Interleaved thinking latency:** Expect higher TTFT (time-to-first-token) when thinking is active. The model reasons before responding. Budget for this in latency-sensitive applications.

**Vast.ai pricing fluctuates:** H200 availability is supply-driven. Listings disappear. Use Vast.ai for dev/eval; use RunPod for production where price predictability matters.

**No vision:** M2.5 is text/code only. If you need vision later, add Qwen2.5-VL-72B on a separate A100 (~$857-1,109/mo) as a sidecar. LiteLLM can route vision requests to it transparently.

**4-bit quality:** AWQ/GGUF 4-bit is measurably lower quality than FP8. On a 230B model the gap is more pronounced than on 80B models. The ~$1,577/mo (Vast.ai) / ~$2,585/mo (RunPod) pricing reflects 4-bit deployment.

**Benchmark dates:** Numbers from model cards at release. This landscape moves fast. Run your own evals on your actual codebase before committing.

**Throughput numbers:** Community benchmarks vary by batch size, context length, prompt structure, and GPU state. Treat all tok/s estimates as order-of-magnitude guidance, not SLAs.
