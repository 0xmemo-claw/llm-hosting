# LLM Hosting (self-hosted)

## 30‑second summary

Host a **3‑tier LLM stack (opus / sonnet / haiku)** on your own GPU box so your agents are **always‑on, predictable, and under your control**.

- **vLLM per model** for throughput
- **LiteLLM router** so apps hit one OpenAI‑compatible endpoint
- Designed for **teams** who want **stable latency** and **fixed monthly costs**

---

## Pick a setup (quick decision tree)

- **Need lowest ops complexity + best quality?** → **1× H200 141GB**
- **Need lower cost but still all tiers?** → **1× H100 80GB**
- **Need more throughput + isolation per tier?** → **H100 + L40S (single box)**
- **Want predictable latency + blast‑radius isolation?** → **2 machines (Opus box + Mid‑tier box)**
- **Dev / light prod only?** → **1× A100 80GB**

> **Recommended default:** **1× H200 141GB on‑demand pod** with opus+sonnet+haiku on a single GPU, **tight context + concurrency caps**.

---

## Performance vs Pricing (quick eval)

> Costs below use **Pods (on‑demand)** typical ranges (min–median). **Serverless Active** is an upper‑bound reference.

| Config | Best for | Tiers layout | VRAM | Est. $/hr<br>(Pod) | Est. $/mo<br>(30d) | Simul users<br>(exclusive tier) | Score |
|---|---|---|---|---:|---:|---|---|
| **1× H200 141GB** | Always‑on agents<br>long context | opus+sonnet+haiku<br>single vLLM | 141GB<br>1×GPU | **$3.6–$3.6** | **$2.6k–$2.6k** | Haiku‑only: 15–35<br>Sonnet‑only: 5–12<br>Opus‑only: 2–5 | ⭐⭐⭐⭐⭐ |
| **1× H100 80GB** | Always‑on<br>balanced cost | opus+sonnet+haiku<br>single vLLM | 80GB<br>1×GPU | **$2.7–$2.8** | **$1.9k–$2.0k** | Haiku‑only: 8–18<br>Sonnet‑only: 3–6<br>Opus‑only: 1–3 | ⭐⭐⭐⭐ |
| **1× A100 80GB** | Dev / light prod | sonnet+haiku<br>opus = smaller/quant | 80GB<br>1×GPU | **$1.2–$1.5** | **$0.86k–$1.11k** | Haiku‑only: 5–12<br>Sonnet‑only: 2–4<br>Opus‑only: 1–2 | ⭐⭐⭐ |
| **1× H100 80GB + 2× L40S 48GB** | Always‑on<br>higher throughput | opus on H100<br>sonnet+haiku on L40S | 176GB<br>3×GPU | **$4.1–$4.3** | **$2.9k–$3.1k** | Haiku‑only: 12–25<br>Sonnet‑only: 4–8<br>Opus‑only: 2–5 | ⭐⭐⭐⭐⭐ |
| **2× machines: H100/H200 + L40S** | Always‑on<br>predictable latency | **Machine A:** opus (70B/72B)<br>**Machine B:** sonnet+haiku | 80/141GB + 48GB | **$3.4–$4.3** | **$2.4k–$3.1k** | Haiku‑only: 12–25<br>Sonnet‑only: 4–8<br>Opus‑only: 2–5 | ⭐⭐⭐⭐⭐ |

**Notes (short):**
- **Exclusive‑tier assumption:** the box is mostly serving **one tier at a time** (e.g., team using Opus now).
- **If mixed‑tier concurrent usage happens**, use the **previous multi‑tenant model** and expect **lower numbers**.
- “No slowdown” = p95 latency stays acceptable (<2s prefill, stable decode) at ~10–20 tok/s, ~8k context.
- Longer context, tool calls, or higher max_tokens reduce concurrency (KV cache + overhead).

**How to measure:** vLLM `/metrics` + a load test (vllm‑bench/Locust) for tokens/sec + p95 latency.

---

## 3 tiers + routing (practical)

There’s no perfect 1:1 OSS match to Claude tiers. We treat them as **quality bands**:

- **opus** → best open‑source quality you can afford
- **sonnet** → strong mid‑tier for coding + reasoning
- **haiku** → cheap + fast for tools/agents

**Recommended OSS bands**
- **Haiku:** Qwen2.5‑7B, Llama‑3.1/3.2‑8B, Mistral‑7B
- **Sonnet:** Qwen2.5‑32B, Llama‑3.1‑70B (4‑bit), Mixtral‑8x22B
- **Opus:** Qwen2.5‑72B / Llama‑3.1‑70B (FP16 if you can), 405B only if multi‑GPU

**Routing flow**
1) Each model served by **vLLM** on its own port
2) **LiteLLM** routes `opus|sonnet|haiku` to the right backend

---

## Setups (opinionated)

### One big machine (simplest)
- **Default:** **H200 141GB** on‑demand pod
- All tiers on one GPU (cap context + concurrency, quantize if needed)

### One box, more headroom
- **H100 80GB + 2× L40S 48GB**
- H100 for opus, L40S for sonnet+haiku

### Two‑machine split (best isolation)
- **Machine A (big):** opus‑tier 70B/72B on **H100 80GB** or **H200 141GB**
- **Machine B (mid):** sonnet + haiku on **L40S 48GB** or **A6000 48GB**

---

<details>
<summary><strong>VRAM reality check (vLLM)</strong></summary>

vLLM GPU memory is **not just weights**. It includes **weights + KV cache + activations + overhead**.
KV cache grows with **context length** and **concurrent sequences**, so vLLM recommends lowering
`max_model_len` and `max_num_seqs` to reduce memory usage.

**Quantization helps weights only:** 8‑bit ≈ ½ weights, 4‑bit ≈ ¼, but KV cache is still FP16/BF16 unless you enable KV‑cache quantization.

**Weight memory rule of thumb:**
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

**How to validate on your box**
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
# Start with realistic caps
# --max-model-len 8192 --gpu-memory-utilization 0.85
# Benchmark after load
# vllm-bench --model <your-model> --num-prompts 200 --max-model-len 8192
```

Sources:
- vLLM GPU memory components: https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/gpu_memory_utilization/
- vLLM conserving memory: https://docs.vllm.ai/en/latest/configuration/conserving_memory/
- HF bitsandbytes: https://huggingface.co/docs/transformers/quantization/bitsandbytes

</details>

<details>
<summary><strong>RunPod pricing (Pods + Serverless)</strong></summary>

### Pods (on‑demand) — typical range

> Pods on‑demand typical range = **min–median** from official RunPod pricing + independent snapshots.

| GPU | Pods on‑demand typical range ($/hr) | Est. $/30d |
|---|---:|---:|
| **H200 141GB** | **$3.59–$3.59** | **$2,585–$2,585** |
| **H100 80GB** | **$2.69–$2.84** | **$1,937–$2,045** |
| **A100 80GB** | **$1.19–$1.54** | **$857–$1,109** |
| **L40S 48GB** | **$0.69–$0.74** | **$497–$533** |

> **Always‑on 3 tiers**: expect **~$1.9k–$3.1k/mo** on on‑demand pods (30‑day month), depending on GPU mix and market.

**Sources (checked 2026‑02‑17 UTC):**
- RunPod GPU Pricing (on‑demand pods): https://www.runpod.io/gpu-pricing
- Hivenet pricing guide (H100/A100 snapshots): https://compute.hivenet.com/post/runpod-pricing-complete-guide-to-gpu-cloud-costs-in-2025
- Sentisight L40/L40S snapshot: https://www.sentisight.ai/runpod-ai-cloud-platform-full-stack-ai-apps/

**Note:** H200 currently only appears on RunPod’s official pricing page, so its range reflects that single source.

### Serverless (per‑second) — upper bound for Pod costs

Sources (checked 2026‑02‑17 UTC):
- Docs (per‑second): https://docs.runpod.io/serverless/pricing
- Pricing page (per‑hour toggle): https://www.runpod.io/pricing

**Formula:** per hour = per second × 3600 → per day × 24 → per 30‑day month × 30

> Pods/instances are often cheaper than serverless. Treat these as **upper‑bound** estimates.

#### Flex pricing
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

#### Active pricing
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

</details>

<details>
<summary><strong>Single‑machine management (ops)</strong></summary>

**OS / services**
- NVIDIA driver + CUDA pinned to a known‑good version
- Docker Compose (or systemd) runs: `vllm-opus`, `vllm-sonnet`, `vllm-haiku`, `litellm`
- systemd units with auto‑restart + health checks
- Monitoring: `nvidia-smi`, DCGM exporter, Prometheus + Grafana

**Network / “channels”**
- Each tier on its own port (8001/8002/8003) behind internal Docker network
- Public entrypoint via Caddy/Nginx with TLS, auth, rate limits
- Centralized logs (stdout → Loki/ELK) + request tracing

**Routing / QoS**
- LiteLLM routes `opus|sonnet|haiku` to each vLLM backend
- Per‑tier concurrency limits, timeouts, and max tokens
- Optional priority queues (opus > sonnet > haiku)

</details>

<details>
<summary><strong>Router setup</strong></summary>

### Option A (recommended): vLLM per model + LiteLLM router
- Run **1 vLLM server per model**
- Route with **LiteLLM** so apps hit one OpenAI‑compatible endpoint

**Pros:** simple mental model, per‑model autoscaling, easy model swaps

**Cons:** higher cost if you keep 3 GPUs always‑on

### Option B: single GPU, multiple models (not recommended)
Possible but painful: memory fragmentation, unload/reload overhead, bad latency.

### OpenClaw integration
Point your OpenClaw OpenAI‑compatible provider at LiteLLM:

- `base_url`: `http(s)://<host>:8000/v1`
- `api_key`: LiteLLM `master_key` (if enabled)

Then use logical model aliases: `haiku`, `sonnet`, `opus`.

</details>
