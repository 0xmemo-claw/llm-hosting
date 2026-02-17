# Claw Launcher — Self-hosted Model Stack (RunPod)

Goal: host an on-demand machine (RunPod pod) that can serve **three capability tiers** roughly matching:
- **Opus-class** (highest quality)
- **Sonnet-class** (strong mid-tier)
- **Haiku-class** (cheap/fast)

This repo focuses on **open-source** model equivalents + a practical **router setup**.

## Quick takeaways

- There is no perfect 1:1 open-source equivalent to Claude Opus/Sonnet/Haiku. The best you can do is **quality bands**.
- For production you want:
  1) **vLLM** (OpenAI-compatible endpoints, best throughput)
  2) **one model per GPU** (simplest + most stable)
  3) **LiteLLM** as a router (OpenAI-compatible) for `opus|sonnet|haiku` logical model names.

## RunPod pricing (from RunPod pricing page)
Source: https://www.runpod.io/pricing

Serverless (per second):
- **B200 180GB**: $8.64/s flex, $6.84/s active
- **H200 141GB**: $5.58/s flex, $4.46/s active
- **H100 80GB**: $4.18/s flex, $3.35/s active
- **A100 80GB**: $2.72/s flex, $2.17/s active
- **L40/L40S/6000 Ada 48GB**: $1.90/s flex, $1.33/s active
- **A6000/A40 48GB**: $1.22/s flex, $0.85/s active
- **5090 32GB**: $1.58/s flex, $1.11/s active
- **4090 24GB**: $1.10/s flex, $0.77/s active
- **L4/A5000/3090 24GB**: $0.69/s flex, $0.48/s active
- **A4000/A4500/etc 16GB**: $0.58/s flex, $0.40/s active

(For pods/instances, RunPod pricing varies by region/availability; treat serverless as an upper bound for always-on inference.)

## Open-source model equivalents (practical bands)

### Haiku-class (fast/cheap)
Targets: low latency, small cost, good enough reasoning for tools/agents.

Good options:
- **Qwen2.5-7B-Instruct** (strong for size)
- **Llama 3.1 8B Instruct** / **Llama 3.2 3B/8B** (if you want smaller)
- **Mistral 7B Instruct v0.3**

Recommended GPU bracket:
- **24GB** (L4/3090/4090) is plenty for 7B/8B at FP16 or 4-bit.

### Sonnet-class (mid-tier)
Targets: strong general quality, coding/reasoning, still deployable on 1 GPU if quantized.

Good options:
- **Qwen2.5-32B-Instruct** (excellent tradeoff)
- **Llama 3.1 70B Instruct** (better quality, heavier)
- **Mixtral 8x22B Instruct** (heavy but strong)

Recommended GPU bracket:
- **48GB** for 32B at FP16 / 70B at 4-bit.
- **80GB** if you want 70B at FP16 or bigger context + throughput.

### Opus-class (top-tier)
Targets: best open-source quality you can realistically self-host.

Good options:
- **Llama 3.1 405B Instruct** (requires multi-GPU, expensive)
- **Llama 3.1 70B Instruct** (best “single node” practical top-tier)
- **Qwen2.5-72B-Instruct** (strong competitor to 70B class)

Recommended GPU bracket:
- **80GB** minimum for 70B/72B at reasonable precision.
- **141–180GB** or multi-GPU for 405B-class.

## Router setup
See `deploy/`.

- Each model served via **vLLM** exposing OpenAI-compatible endpoints.
- **LiteLLM** provides a single `OPENAI_BASE_URL` for apps, and routes by model name.

## What you plug into OpenClaw
In your `openclaw.json`, set provider endpoints to the LiteLLM router, then define logical model aliases (`opus`, `sonnet`, `haiku`) mapping to the router.

---

Next steps:
- Add diagnostics around X signup (separate workstream)
- Add per-GPU sizing notes for context length + kv-cache
