# LLM Hosting (self-hosted)

Host your **own 3‑tier LLM stack** for personal or company agents — always‑on, predictable, and under your control.

This repo focuses on:
- **One machine** that serves **opus / sonnet / haiku** tiers
- **vLLM** per model for throughput
- **LiteLLM** as a router so apps hit one OpenAI‑compatible endpoint

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

### Option A — **1× H200 141GB** (best single‑GPU box)
- Can host **opus‑class 70B + sonnet + haiku** with multi‑model or separate instances
- Best chance to keep everything on a single GPU without constant unload/reload

### Option B — **1× H100 80GB** (cheaper single‑GPU)
- **Opus** likely needs 4‑bit; **sonnet + haiku** can share vLLM but tradeoffs: lower throughput + cache pressure
- Good if you accept tight memory + some latency spikes

### Option C — **multi‑GPU pod**: **2× 48GB + 1× 80GB**
- 80GB for **opus**, 48GBs for **sonnet + haiku**
- If you truly want “one machine” with multiple GPUs, **cost ≈ sum of GPUs** (serverless is an upper bound). Pods are often cheaper — we need real pod pricing to confirm.

---

## RunPod pricing (serverless upper bound)

Source: https://www.runpod.io/pricing

**Formula:**
- per hour = per second × 3600
- per day = per hour × 24
- per 30‑day month = per day × 30

> Pods/instances usually cost **less** than serverless; treat these as **upper‑bound** estimates until we pull pod pricing.

### Flex pricing
| GPU | $/s | $/hr | $/day | $/30d |
|---|---:|---:|---:|---:|
| **B200 180GB** | $8.64 | $31,104.00 | $746,496.00 | $22,394,880.00 |
| **H200 141GB** | $5.58 | $20,088.00 | $482,112.00 | $14,463,360.00 |
| **H100 80GB** | $4.18 | $15,048.00 | $361,152.00 | $10,834,560.00 |
| **A100 80GB** | $2.72 | $9,792.00 | $235,008.00 | $7,050,240.00 |
| **48GB (L40/L40S/6000 Ada)** | $1.90 | $6,840.00 | $164,160.00 | $4,924,800.00 |
| **48GB (A6000/A40)** | $1.22 | $4,392.00 | $105,408.00 | $3,162,240.00 |
| **24GB (4090)** | $1.10 | $3,960.00 | $95,040.00 | $2,851,200.00 |
| **24GB (L4/A5000/3090)** | $0.69 | $2,484.00 | $59,616.00 | $1,788,480.00 |

### Active pricing
| GPU | $/s | $/hr | $/day | $/30d |
|---|---:|---:|---:|---:|
| **B200 180GB** | $6.84 | $24,624.00 | $590,976.00 | $17,729,280.00 |
| **H200 141GB** | $4.46 | $16,056.00 | $385,344.00 | $11,560,320.00 |
| **H100 80GB** | $3.35 | $12,060.00 | $289,440.00 | $8,683,200.00 |
| **A100 80GB** | $2.17 | $7,812.00 | $187,488.00 | $5,624,640.00 |
| **48GB (L40/L40S/6000 Ada)** | $1.33 | $4,788.00 | $114,912.00 | $3,447,360.00 |
| **48GB (A6000/A40)** | $0.85 | $3,060.00 | $73,440.00 | $2,203,200.00 |
| **24GB (4090)** | $0.77 | $2,772.00 | $66,528.00 | $1,995,840.00 |
| **24GB (L4/A5000/3090)** | $0.48 | $1,728.00 | $41,472.00 | $1,244,160.00 |

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
