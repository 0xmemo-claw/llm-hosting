# LLM Hosting (self-hosted)

Host your **own 3‑tier LLM stack** for personal or company agents — always‑on, predictable, and under your control.

This repo focuses on:
- **One machine** that serves **opus / sonnet / haiku** tiers
- **vLLM** per model for throughput
- **LiteLLM** as a router so apps hit one OpenAI‑compatible endpoint

---

## Performance vs Pricing (quick eval)

> Costs below use **Serverless Active $/hr** as an **upper‑bound reference**. Pods/instances are often cheaper.

| Config | Target use | Tiers hosted + layout | VRAM / GPU count | Cost (Active $/hr upper‑bound) | Expected throughput / latency (notes) | Reliability | Rec.
|---|---|---|---|---:|---|---|---|
| **1× H200 141GB (On‑Demand pod)** | Always‑on agents, heavy reasoning | **opus+sonnet+haiku** on **single vLLM** (shared KV cache) | 141GB / 1 GPU | **$4.46/hr** | **High / Med** — best single‑GPU headroom; long‑context viable with KV cache tuning | **On‑demand recommended** for stability | ⭐⭐⭐⭐⭐ |
| **1× H100 80GB** | Always‑on, balanced cost | **opus+sonnet+haiku** on **single vLLM** (tight KV cache) | 80GB / 1 GPU | **$3.35/hr** | **Med‑High / Med** — works **only** with aggressive quant + shorter contexts for 70B‑class | **On‑demand recommended** | ⭐⭐⭐⭐ |
| **1× A100 80GB (budget)** | Dev / light prod | **sonnet+haiku** + **opus** w/ smaller model or heavy quant | 80GB / 1 GPU | **$2.16/hr** | **Med / Med‑Low** — value pick; expect shorter contexts + stricter concurrency caps | **Spot‑friendly** for non‑critical | ⭐⭐⭐ |
| **1× H100 80GB + 2× L40S 48GB** | Always‑on agents, higher throughput | **opus** on H100, **sonnet+haiku** on **per‑tier GPUs** | 176GB / 3 GPUs | **$6.01/hr** | **High / Low** — best isolation; parallel queues, fewer KV cache clashes | **On‑demand recommended** | ⭐⭐⭐⭐⭐ |

**Notes:** Real throughput depends on model choice, context length, quantization, and batching. Benchmark your setup with **`vllm-bench`** (tokens/sec + p95 latency) before locking infra.

---

## 3 tiers + routing (practical)

There is no perfect 1:1 OSS match to Claude Opus/Sonnet/Haiku. We treat them as **quality bands** and route by tier name:

- **opus** → best open‑source quality you can afford
- **sonnet** → strong mid‑tier for coding + reasoning
- **haiku** → cheap + fast for tools/agents

**Recommended OSS bands**:
- **Haiku:** Qwen2.5‑7B, Llama‑3.1/3.2‑8B, Mistral‑7B
- **Sonnet:** Qwen2.5‑32B, Llama‑3.1‑70B (4‑bit), Mixtral‑8x22B
- **Opus:** Qwen2.5‑72B / Llama‑3.1‑70B (FP16 if you can), 405B only if multi‑GPU

Routing flow:
1) Each model served by **vLLM** on its own port
2) **LiteLLM** routes `opus|sonnet|haiku` to the right backend

---

## Single‑machine setups (opinionated)

### Pods vs Serverless (RunPod)
- **Pods (VMs)** = full machine with GPU(s), billed while running. Variants:
  - **On‑Demand** (your VM stays up; stable availability)
  - **Spot/Community** (cheaper, can be preempted)
  - **Secure Cloud** (isolated hardware + compliance)
- **Serverless** = pay‑per‑second workers; scale‑to‑zero (Flex) or always‑on (Active).

### Best tier option (one big machine)
**Recommended default:** **On‑Demand Pod, H200 141GB**
- Can host **opus + sonnet + haiku** on one GPU **if you cap context/concurrency and tune KV cache** (quantization helps a lot).
- Lowest ops complexity; no cross‑GPU routing.

**Cheaper multi‑GPU pod (more headroom):** **1× H100 80GB + 2× L40S 48GB**
- **H100** for opus, **L40S** for sonnet + haiku.
- Better throughput and isolation per tier; still one box.

**When to use spot/community:** non‑critical workloads, batch inference, or can tolerate preemption. **On‑demand** for always‑on agents and latency‑sensitive apps.

---

## VRAM reality check (vLLM)

vLLM GPU memory is **not just weights**. It includes **model weights + KV cache + activation memory + system overhead** ([vLLM GPU memory components](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/gpu_memory_utilization/)). The **KV cache grows linearly with context length and concurrent sequences**, so vLLM explicitly recommends lowering `max_model_len` and `max_num_seqs` to reduce memory usage ([vLLM conserving memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)).

**Quantization helps for weights only:** 8‑bit **halves** weight memory; 4‑bit is roughly **¼** (rule of thumb), but KV cache is still usually FP16/BF16 unless you enable KV‑cache quantization ([HF bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)).

**Weight memory rule of thumb (params × bytes/param):**
- **FP16/BF16** ≈ 2 bytes/param → **7B ≈ 14GB**, **32B ≈ 64GB**, **70B ≈ 140GB**
- **INT8** ≈ 1 byte/param → ~½ the above
- **4‑bit** ≈ 0.5 bytes/param → ~¼ the above

**Conservative single‑GPU guidance** (FP16/BF16 weights, FP16 KV cache, ~10–15% overhead, **1–2 concurrent** sequences):

| Model | 8k context | 16k context | 32k context | Notes |
|---|---:|---:|---:|---|
| **7B** | **20–24GB** | **24–32GB** | **32–48GB** | 4–8 concurrent? add ~1× extra KV cache (≈ +4/8/16GB). |
| **32B** | **80–96GB** | **96–120GB** | **140–180GB** | 4‑bit weights can drop ~48GB, but KV cache dominates at 16k+. |
| **70B/72B** | **160–200GB** | **200–260GB** | **280–360GB** | **H100 80GB** works only for **4‑bit + short context**. 16k+ needs H200/B200 or multi‑GPU. |

**Practical take:** if you want **long context + concurrency**, you size for **KV cache**, not just weights.

### How to validate on your box
```bash
# 1) What you actually have
nvidia-smi --query-gpu=name,memory.total --format=csv

# 2) Start with realistic caps (lower first, then relax)
# vLLM example flags:
#   --max-model-len 8192
#   --gpu-memory-utilization 0.85

# 3) Benchmark after it loads (tokens/sec + p95 latency)
# vllm-bench --model <your-model> --num-prompts 200 --max-model-len 8192
```

---

## RunPod pricing (Serverless, per‑second) — **upper bound for Pod costs**

Sources (checked 2026‑02‑17 UTC):
- Docs (per‑second): https://docs.runpod.io/serverless/pricing
- Pricing page (per‑hour toggle): https://www.runpod.io/pricing

**Formula:**
- per hour = per second × 3600
- per day = per hour × 24
- per 30‑day month = per day × 30

> Pods/instances are often cheaper than serverless. Treat these as **upper‑bound** estimates.

**Formula:**
- per hour = per second × 3600
- per day = per hour × 24
- per 30‑day month = per day × 30

> Pods/instances usually cost **less** than serverless; treat these as **upper‑bound** estimates until we pull pod pricing.

### Flex pricing
| GPU | $/s | $/hr | $/day | $/30d |
|---|---:|---:|---:|---:|
| **B200 180GB** | $0.00240 | $8.64 | $207.36 | $6,220.80 |
| **H200 141GB** | $0.00155 | $5.58 | $133.92 | $4,017.60 |
| **H100 80GB** | $0.00116 | $4.18 | $100.22 | $3,006.72 |
| **A100 80GB** | $0.00076 | $2.74 | $65.66 | $1,969.92 |
| **48GB (L40/L40S/6000 Ada)** | $0.00053 | $1.91 | $45.79 | $1,373.76 |
| **48GB (A6000/A40)** | $0.00034 | $1.22 | $29.38 | $881.28 |
| **24GB (4090)** | $0.00031 | $1.12 | $26.78 | $803.52 |
| **24GB (L4/A5000/3090)** | $0.00019 | $0.68 | $16.42 | $492.48 |
| **16GB (A4000/A4500/RTX 4000)** | $0.00016 | $0.58 | $13.82 | $414.72 |

### Active pricing
| GPU | $/s | $/hr | $/day | $/30d |
|---|---:|---:|---:|---:|
| **B200 180GB** | $0.00190 | $6.84 | $164.16 | $4,924.80 |
| **H200 141GB** | $0.00124 | $4.46 | $107.14 | $3,214.08 |
| **H100 80GB** | $0.00093 | $3.35 | $80.35 | $2,410.56 |
| **A100 80GB** | $0.00060 | $2.16 | $51.84 | $1,555.20 |
| **48GB (L40/L40S/6000 Ada)** | $0.00037 | $1.33 | $31.97 | $959.04 |
| **48GB (A6000/A40)** | $0.00024 | $0.86 | $20.74 | $622.08 |
| **24GB (4090)** | $0.00021 | $0.76 | $18.14 | $544.32 |
| **24GB (L4/A5000/3090)** | $0.00013 | $0.47 | $11.23 | $336.96 |
| **16GB (A4000/A4500/RTX 4000)** | $0.00011 | $0.40 | $9.50 | $285.12 |


## Single‑machine management (ops)

**OS / services**
- **NVIDIA driver + CUDA toolkit** pinned to a known‑good version
- **Docker Compose** (or systemd) runs: `vllm-opus`, `vllm-sonnet`, `vllm-haiku`, `litellm`
- **systemd units** for each container/service with auto‑restart and healthchecks
- Monitoring: **nvidia-smi**, **DCGM exporter**, Prometheus + Grafana (GPU mem, power, tokens/sec)

**Network / “channels”**
- Each tier on its **own port** (e.g., 8001/8002/8003) behind **internal Docker network**
- Public entrypoint via **Caddy/Nginx** with TLS, auth (API keys), and rate limits
- Centralized logs (stdout → Loki/ELK) + request tracing

**Routing / QoS**
- **LiteLLM** routes `opus|sonnet|haiku` to each vLLM backend
- Per‑tier **concurrency limits**, **timeouts**, and **max tokens**
- Optional: priority queues (opus > sonnet > haiku) for latency control

---

## Router setup
See `deploy/`.

- Each model served via **vLLM** exposing OpenAI‑compatible endpoints.
- **LiteLLM** provides a single `OPENAI_BASE_URL` for apps, and routes by model name.

## Integrate with your agent stack

Point your agent system (OpenClaw or any OpenAI‑compatible client) at LiteLLM:
- `base_url`: `http(s)://<host>:8000/v1`
- `api_key`: LiteLLM `master_key` (if enabled)

Then use logical model names: `opus`, `sonnet`, `haiku`.
