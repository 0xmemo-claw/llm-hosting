# LLM Hosting — MiniMax M2.5 on a Single H200

**MiniMax M2.5** (230B/10B active MoE) on 1× H200 141GB for ~$1,545/mo. 80.2% SWE-Bench Verified — #1 open-weight coding model. Interleaved thinking, tool calling, Apache 2.0.

---

## Why M2.5

| | MiniMax M2.5 | Claude Opus 4.6 | GPT-5.2 | Gemini-3 Pro |
|---|---|---|---|---|
| **SWE-Bench Verified** | **80.2%** | ~72% | ~70% | ~68% |
| **BrowseComp** | **76.3%** | — | — | — |
| **Multi-SWE-Bench** | **51.3%** | — | — | — |

230B params, 10B active per token (MoE). Interleaved thinking maintains reasoning state across tool calls — the model plans, executes, and revises like an architect. 196K max context. Apache 2.0 license.

---

## Deployment Options

Two tested approaches for a single H200 141GB, plus a multi-GPU reference path:

| | **AWQ 4-bit** | **Unsloth UD-Q3_K_XL** | **FP8 Native** |
|---|---|---|---|
| **Guide** | [`deploy.md`](deploy.md) | [`unsloth.md`](unsloth.md) | [Official docs](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/docs/vllm_deploy_guide.md) |
| **Model** | `QuantTrio/MiniMax-M2.5-AWQ` | `unsloth/MiniMax-M2.5-GGUF` | `MiniMaxAI/MiniMax-M2.5` |
| **Weights on GPU** | ~112 GiB (with CPU offload) | ~101 GiB | ~220 GiB |
| **Max Context** | **65K** | **128K+** | 65K (4 GPU) |
| **Engine** | vLLM | llama.cpp | vLLM |
| **GPUs** | 1× H200 | 1× H200 | 4× H200/H100 |
| **Batched serving** | Yes (up to 4 concurrent) | No (single stream) | Yes |
| **Tool calling** | Native (vLLM parser) | Manual prompt template | Native (vLLM parser) |
| **Quality** | INT4 — slight degradation | ~3-bit dynamic — moderate | Full FP8 — best |
| **Throughput** | ~25–40 tok/s | ~25–40 tok/s | ~30–40 tok/s |
| **Monthly (Vast.ai)** | ~$1,545 | ~$1,545 | ~$6,190 |

### When to pick which

- **AWQ (`deploy.md`)** — Best all-rounder. vLLM gives you batched serving, native tool calling, and an OpenAI-compatible API with proper reasoning parsing. 65K context handles most workloads. Pick this for multi-user or agentic setups.

- **Unsloth (`unsloth.md`)** — Best for long context. 128K+ tokens on a single GPU without CPU offloading tricks. llama.cpp is simpler to set up but single-stream only. Pick this for document analysis, large codebases, or single-user workloads where context > throughput.

- **FP8 multi-GPU** — Best quality. Full precision, no quantization artifacts. Requires 4+ GPUs and 4× the cost. Pick this for production where quality per token matters most.

---

## Unsloth Quantization Variants

[unsloth/MiniMax-M2.5-GGUF](https://huggingface.co/unsloth/MiniMax-M2.5-GGUF) publishes Dynamic 2.0 quantizations that selectively keep critical layers at higher precision while compressing less sensitive layers. The `UD-` prefix means Unsloth Dynamic; `XL` means more layers kept at high precision.

| Variant | Size | H200 Headroom | Max Context | Quality |
|---------|------|---------------|-------------|---------|
| UD-Q3_K_XL | ~101 GiB | ~40 GiB | **128K+** | Good — best for 3-bit |
| Q3_K_M (standard) | ~86 GiB | ~55 GiB | **196K** | Moderate |
| UD-Q2_K_XL | ~72 GiB | ~69 GiB | **196K** | Reduced |
| UD-Q4_K_XL | ~120 GiB | ~20 GiB | 65K | Higher — close to AWQ |
| Q4_K_M (standard) | ~120 GiB | ~20 GiB | 65K | Good |

**Recommended: UD-Q3_K_XL** — the sweet spot of quality, size, and context window on a single H200.

---

## Cost

| Setup | Provider | $/hr | $/month | Context |
|-------|----------|------|---------|---------|
| 1× H200 (AWQ or Unsloth) | Vast.ai | $2.15 | ~$1,545 | 65–128K |
| 1× H200 (AWQ or Unsloth) | RunPod | $3.59 | ~$2,585 | 65–128K |
| 4× H200 (FP8 native) | Vast.ai | ~$8.60 | ~$6,190 | 32–65K |

Use **Vast.ai** for dev/eval (cheapest, variable quality). Use **RunPod** for production (predictable, secure cloud option). Avoid AWS/GCP/Azure for single-GPU inference — they sell 8-GPU nodes minimum ($22k+/mo for one H100 equivalent).

For comparison: Claude Opus API costs $75/M output tokens. M2.5 self-hosted runs ~$35–95/M output at real-world utilization.

---

## Quick Start

### AWQ / vLLM (65K context)

```bash
# On your Vast.ai H200 instance (vastai/vllm template)
pkill -f 'vllm serve'

export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve QuantTrio/MiniMax-M2.5-AWQ \
  --max-model-len 65536 --gpu-memory-utilization 0.98 \
  --trust-remote-code --enforce-eager \
  --cpu-offload-gb 50 --max-num-seqs 4 \
  --enable-auto-tool-choice --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --download-dir /workspace/models --host 0.0.0.0 --port 8001 \
  --api-key $M25_API_KEY
```

### Unsloth / llama.cpp (128K context)

```bash
# Build llama.cpp, download model, then:
./llama.cpp/build/bin/llama-server \
  -m /workspace/models/UD-Q3_K_XL/MiniMax-M2.5-UD-Q3_K_XL-00001-of-00004.gguf \
  --port 8000 --host 0.0.0.0 \
  -c 131072 -n 8192 --n-gpu-layers 999 --flash-attn
```

Both serve an OpenAI-compatible API. vLLM runs on port 8001 (not 8000 — Vast.ai's Caddy proxy intercepts 8000).

---

## Connect (from any device)

vLLM is exposed directly via a Cloudflare Tunnel with `--api-key` auth. No LiteLLM proxy needed on the remote — single hop.

Copy `.env.example` to `.env` and fill in the values. Ask the deployer for the current tunnel URL and API key.

```bash
cp .env.example .env
```

```
Base URL:  $M25_BASE_URL
API Key:   $M25_API_KEY
Model:     QuantTrio/MiniMax-M2.5-AWQ
```

> Tunnel URL changes on instance restart. See [`endpoints.yaml`](endpoints.yaml) for the current URL and restart instructions.

### Cursor / Continue / Claw

| Setting | Value |
|---------|-------|
| Base URL | `$M25_BASE_URL` |
| API Key | `$M25_API_KEY` |
| Model | `QuantTrio/MiniMax-M2.5-AWQ` |

### Python

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["M25_BASE_URL"],
    api_key=os.environ["M25_API_KEY"]
)

r = client.chat.completions.create(
    model="QuantTrio/MiniMax-M2.5-AWQ",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=500
)
print(r.choices[0].message.content)
```

### curl

```bash
source .env
curl $M25_BASE_URL/chat/completions \
  -H "Authorization: Bearer $M25_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"QuantTrio/MiniMax-M2.5-AWQ","messages":[{"role":"user","content":"Hello"}],"max_tokens":500}'
```

### How it works

```
Client (any device)
  → Cloudflare Tunnel (public HTTPS)
    → vLLM :8001 (--api-key auth, inference)
```

Vast.ai's Caddy proxy intercepts port 8000 with cookie-based auth that breaks SDK clients. We run vLLM on port 8001 and expose it with our own `cloudflared` tunnel. See [`endpoints.yaml`](endpoints.yaml) for restart instructions.

---

## Files

| File | Purpose |
|------|---------|
| [`deploy.md`](deploy.md) | Full AWQ/vLLM deployment guide (tested, 65K context) |
| [`unsloth.md`](unsloth.md) | Unsloth GGUF / llama.cpp guide (128K+ context) |
| [`endpoints.yaml`](endpoints.yaml) | Public tunnel URLs and client config examples |
| [`.env.example`](.env.example) | Template for API credentials (copy to `.env`) |

---

## Caveats

- **No vision.** M2.5 is text/code only.
- **Thinking adds latency.** The model reasons before answering — budget 2–5s TTFT on complex prompts.
- **Quantization quality.** AWQ 4-bit and Unsloth 3-bit both degrade quality vs FP8. For most coding tasks the difference is small. For complex multi-step reasoning, FP8 multi-GPU is measurably better.
- **Vast.ai is a marketplace.** Machines vanish. Use for dev; RunPod for production.
- **All throughput numbers are estimates.** Varies by context length, batch size, and prompt structure.
