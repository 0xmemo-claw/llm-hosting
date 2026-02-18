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

> **Pods (on‑demand)** = typical range (min–median). **Flex Pod** = **RunPod Serverless Flex** rates as an **upper‑bound** proxy (RunPod doesn’t publish “Flex Pod” pricing).

| Config | Best for | Layout | VRAM | Pod on‑demand<br>$/hr | Pod Flex<br>$/hr | Pod on‑demand<br>$/30d | Pod Flex<br>$/30d | Simul users<br>(exclusive tier) | Score |
|---|---|---|---|---:|---:|---:|---:|---|---|
| **1× H200 141GB** | Always‑on<br>long context | opus+sonnet+haiku<br>1× vLLM | 141GB<br>1×GPU | **$3.6** | **$5.6** | **$2.6k** | **$4.0k** | Haiku: 15–35<br>Sonnet: 5–12<br>Opus: 2–5 | ⭐⭐⭐⭐⭐ |
| **1× H100 80GB** | Always‑on<br>balanced cost | opus+sonnet+haiku<br>1× vLLM | 80GB<br>1×GPU | **$2.7–$2.8** | **$4.2** | **$1.9k–$2.0k** | **$3.0k** | Haiku: 8–18<br>Sonnet: 3–6<br>Opus: 1–3 | ⭐⭐⭐⭐ |
| **1× A100 80GB** | Dev / light prod | sonnet+haiku<br>opus = smaller/quant | 80GB<br>1×GPU | **$1.2–$1.5** | **$2.7** | **$0.86k–$1.11k** | **$2.0k** | Haiku: 5–12<br>Sonnet: 2–4<br>Opus: 1–2 | ⭐⭐⭐ |
| **1× H100 + 2× L40S** | Always‑on<br>more throughput | opus on H100<br>sonnet+haiku on L40S | 176GB<br>3×GPU | **$4.1–$4.3** | **$8.0** | **$2.9k–$3.1k** | **$5.8k** | Haiku: 12–25<br>Sonnet: 4–8<br>Opus: 2–5 | ⭐⭐⭐⭐⭐ |
| **2× machines: H100/H200 + L40S** | Always‑on<br>best isolation | **A:** opus<br>**B:** sonnet+haiku | 80/141GB<br>+ 48GB | **$3.4–$4.3** | **$6.1–$7.5** | **$2.4k–$3.1k** | **$4.4k–$5.4k** | Haiku: 12–25<br>Sonnet: 4–8<br>Opus: 2–5 | ⭐⭐⭐⭐⭐ |
| **Opus 5–10: 2× H100 replicas** | Opus‑only<br>scale‑out | opus only<br>2× vLLM | 160GB<br>2×GPU | **$5.4–$5.7** | **$8.4** | **$3.9k–$4.1k** | **$6.0k** | Opus: 5–7 | ⭐⭐⭐⭐ |
| **Opus 5–10: 2× H200 (TP)** | Opus‑only<br>single box | opus only<br>1× vLLM (TP) | 282GB<br>2×GPU | **$7.2** | **$11.2** | **$5.2k** | **$8.0k** | Opus: 5–10 | ⭐⭐⭐⭐ |

**Notes (short):**
- **Exclusive‑tier assumption:** the box is mostly serving **one tier at a time** (e.g., team using Opus now).
- **Mixed‑tier concurrent usage** lowers these numbers (shared KV cache).
- **Flex Pod definition:** RunPod doesn’t publish “Flex Pod” pricing; we use **Serverless Flex** rates as an **upper‑bound proxy** for spot/interruptible economics (docs: https://docs.runpod.io/serverless/pricing, pricing page: https://www.runpod.io/pricing).
- “No slowdown” = acceptable p95 (<2s prefill) at ~10–20 tok/s, ~8k context.

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

## Open-Weight Alternatives (Feb 2026)

> **TL;DR:** GLM-4.7-Flash or Devstral Small 2 for haiku, Qwen3-Coder-Next 80B for sonnet, DeepSeek-V3.2 for opus. Multi-GPU required for the full-fat flagship models.

### Model overview

| Model | Params (total / active) | Architecture | Context | SWE-Bench* | Min VRAM (quant) | Best GPU config | Tier match | Notes |
|---|---|---|---|---|---|---|---|---|
| **Devstral Small 2** | 24B / 24B (dense) | Dense transformer | 256K | ~68% Verified | ~13GB (4-bit) | 1× RTX 3090/4090 | **Haiku** | Best single-24GB coding model; ships w/ Mistral Vibe CLI |
| **GLM-4.7-Flash** | 30B / 3B active | MoE | 128K | 59.2% Verified | ~6–8GB (4-bit) | 1× RTX 3090/4090 | **Haiku** | Blazing fast (60–80+ tok/s); excels at tool-calling + UI gen |
| **Qwen3-Coder-30B-A3B** | 30B / 3B active | MoE | 256K | — | ~8GB (4-bit) | 1× 24GB GPU | **Haiku** | Smallest Qwen coder; fits a single consumer GPU |
| **Qwen3-Coder-Next (80B-A3B)** | 80B / 3B active | MoE | 256K | — | ~46GB (4-bit GGUF) | 2× L40S or 1× H200 | **Sonnet** | Practical Qwen coder tier; strong coding/agent perf |
| **GLM-4.7 (full)** | 355–384B / 32B active | MoE | 128K | 59.2% Verified | ~135GB (2-bit) | 2× H100 or 1× H200 | **Sonnet–Opus** | Full model; not single-GPU practical |
| **Qwen3-Coder 480B** | 480B / 35B active | MoE | 256K (→1M) | ~39% Pro Public | ~240GB (4-bit) | 4× H200 or 8× H100 | **Opus** | Flagship; needs serious multi-GPU; coding-agent-first design |
| **DeepSeek-V3.2** | 671B / 37B active | MoE | 128K | — (top-tier) | ~170GB (FP8) | 4–5× H200 or 8× H100 | **Opus** | Best OSS general-purpose rival to Claude Opus |

> \* SWE-Bench numbers are from mixed sources (Verified vs Pro vs Public subsets). See [benchmark caveats](#benchmark-caveats) below.

### Practical tier mapping

<details>
<summary><strong>Haiku band — single 24GB GPU (~$0.47–$1.12/hr Serverless Active)</strong></summary>

**Candidates:** Devstral Small 2, GLM-4.7-Flash, Qwen3-Coder-30B-A3B

| Model | GPU | Pod on-demand $/hr | Serverless Active $/hr | Recommended? |
|---|---|---:|---:|---|
| Devstral Small 2 (4-bit) | 1× RTX 4090 | ~$0.74 | $0.76 | ✅ Best coding accuracy |
| GLM-4.7-Flash (4-bit) | 1× RTX 3090/4090 | ~$0.68–$0.74 | $0.47–$0.76 | ✅ Best speed + tool-calling |
| Qwen3-Coder-30B-A3B (4-bit) | 1× 24GB GPU | ~$0.68–$0.74 | $0.47–$0.76 | ⚠️ Decent; weaker than above two |

**Verdict:** Deploy **Devstral Small 2** if coding accuracy matters. Deploy **GLM-4.7-Flash** if you need raw throughput or heavy tool-calling. Both fit a single 24GB GPU and cost ~$350–$550/mo on always-on pods.

</details>

<details>
<summary><strong>Sonnet band — 48–96GB GPU (~$1.33–$4.46/hr Serverless Active)</strong></summary>

**Candidates:** Qwen3-Coder-Next 80B, GLM-4.7-Flash (overkill on 48GB), GLM-4.7 full (if you have H200)

| Model | GPU | Pod on-demand $/hr | Serverless Active $/hr | Recommended? |
|---|---|---:|---:|---|
| Qwen3-Coder-Next 80B (4-bit GGUF) | 2× L40S 48GB | ~$1.38–$1.48 | $2.66 | ✅ Best fit; ~46GB GGUF |
| Qwen3-Coder-Next 80B (4-bit) | 1× H200 141GB | $3.59 | $4.46 | ⚠️ Overkill GPU; run with haiku too |
| GLM-4.7-Flash (4-bit) | 1× L40S 48GB | $0.69–$0.74 | $1.33 | ✅ If latency > accuracy |

**Verdict:** **2× L40S + Qwen3-Coder-Next 80B** is the sweet spot (~$1k–$1.1k/mo). If you already have an H200 for opus, throw GLM-4.7-Flash on a dedicated L40S for sonnet/haiku.

</details>

<details>
<summary><strong>Opus band — multi-GPU required</strong></summary>

**Candidates:** DeepSeek-V3.2 (671B), Qwen3-Coder 480B

| Model | GPU config | Est. Pod $/hr | Serverless Active $/hr | Notes |
|---|---|---:|---:|---|
| DeepSeek-V3.2 (FP8) | 4–5× H200 141GB | ~$14.4–$18.0 | ~$17.8–$22.3 | 4× H200 NVLink pod preferred |
| DeepSeek-V3.2 (FP8) | 8× H100 80GB | ~$21.5–$22.7 | ~$26.8 | Higher cost; widely available |
| Qwen3-Coder 480B (4-bit) | 4× H200 141GB | ~$14.4 | ~$17.8 | ~240GB needed; coding-focused |
| Qwen3-Coder 480B (4-bit) | 8× H100 80GB | ~$21.5–$22.7 | ~$26.8 | Fallback if no H200 |

> **RunPod pricing estimate:** multi-GPU pods aren't always available. Budget **$10k–$16k/mo** for always-on opus-class open-weight model. Often cheaper to keep using the Claude API at this tier unless volume is very high.

**Verdict:** **DeepSeek-V3.2 on 4× H200** if you're committed. Otherwise: use the Claude API for opus and self-host open-weight for haiku/sonnet only.

</details>

### What we'd actually deploy

```
Haiku:  GLM-4.7-Flash (4-bit)        1× RTX 4090    ~$0.47/hr active
        OR Devstral Small 2 (4-bit)   1× RTX 4090    ~$0.76/hr active
        
Sonnet: Qwen3-Coder-Next 80B (4-bit) 2× L40S 48GB   ~$2.66/hr active
        OR GLM-4.7-Flash              1× L40S 48GB   ~$1.33/hr active (fast/cheap)

Opus:   DeepSeek-V3.2 (FP8)          4× H200        ~$17.8/hr active
        OR keep using Claude API — genuinely cheaper unless you have massive volume
```

**Monthly ballpark (haiku + sonnet self-hosted, opus = Claude API):**
- 1× RTX 4090 (haiku) + 2× L40S (sonnet): ~**$650–$900/mo** always-on pods
- Opus via Claude API at modest usage: often cheaper than ~$10k+/mo multi-GPU pod

### Benchmark caveats

<a name="benchmark-caveats"></a>

> ⚠️ **SWE-bench numbers are unreliable comparators.** Here's why:
>
> - **Verified vs Pro vs Public** subsets differ wildly. A model scoring 59% on Verified might score ~35% on Pro Public (harder tasks, different agents).
> - **Contamination:** flagship models (esp. DeepSeek) have been trained on or near benchmark data. Scores on public subsets are optimistic.
> - **Agent scaffolding matters:** the same base model gets 20–40% improvement with better scaffolding. Scores don't transfer directly to your agent stack.
> - **Treat as relative signal, not absolute truth.** Use evals on your own tasks/codebase before committing.
>
> Sources: [SEAL SWE-bench Pro leaderboard](https://scale.com/leaderboard/swe_bench_pro), [SWE-bench Verified](https://www.swebench.com/), [GLM-4.7 HF](https://huggingface.co/THUDM/GLM-4.7), [Devstral Small 2 blog](https://mistral.ai/news/devstral), [DeepSeek-V3.2 report](https://github.com/deepseek-ai/DeepSeek-V3), [Qwen3-Coder HF](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct)

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
>
> **Note:** RunPod’s **Flex** label here refers to **Serverless Flex workers**, which we use as the **upper‑bound proxy** for “Flex Pod” estimates in the quick‑eval table.

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
