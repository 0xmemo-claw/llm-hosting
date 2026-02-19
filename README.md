# LLM Hosting — Self-Hosted Coding Models on Vast.ai

Open-weight coding models on single-GPU Vast.ai instances. OpenAI-compatible API, tool calling, accessible from any device via Cloudflare Tunnel.

---

## Model Comparison

| | **Qwen3-Coder-Next** | **MiniMax M2.5** |
|---|---|---|
| **SWE-Bench Verified** | 70.6% | **80.2%** |
| **Architecture** | 80B / 3B active (MoE) | 230B / 10B active (MoE) |
| **Context** | **256K** | 196K |
| **Thinking mode** | No | Yes (interleaved) |
| **Tool calling** | `qwen3_coder` | `minimax_m2` |
| **4-bit VRAM** | **~46 GiB** | ~115 GiB |
| **Min GPU** | **L40S 48GB** | H200 141GB |
| **Throughput** | **~60–85 tok/s** | ~25–40 tok/s |
| **Concurrent users** | **10–15** (A100) | 4 (H200, tight) |
| **Monthly (Vast.ai)** | **~$450–576** | ~$1,545 |
| **License** | Apache 2.0 | Apache 2.0 |

**Verdict:** Qwen3-Coder-Next is the better team model. 3× cheaper, 2× faster, 3× more concurrent users, 256K context. You trade ~10 SWE-bench points and thinking mode — but for multi-user coding agent workflows, the throughput and cost win.

---

## Recommended: Qwen3-Coder-Next

### Why

- **3B active params** — only 3B of 80B fire per token, so inference is fast and KV cache is small
- **256K native context** — full repo-scale understanding without tricks
- **~46 GiB at 4-bit** — fits on an A100 80GB with **34 GiB free** for KV cache and concurrency
- **~60–85 tok/s** — 2–3× faster than M2.5, feels instant for coding tasks
- **Agentic training** — trained on 800K verifiable coding tasks with execution feedback
- **Tool calling** — native `qwen3_coder` parser in vLLM, works with Claude Code, Cline, Continue

### GPU Options

| GPU | VRAM | After 4-bit weights | Context | Concurrent users | $/mo (Vast.ai) |
|-----|------|---------------------|---------|------------------|----------------|
| **A100 80GB** (recommended) | 80 GiB | **34 GiB free** | **256K** | **10–15** | **~$450–576** |
| L40S 48GB | 48 GiB | ~2 GiB free | 32–64K | 3–5 | ~$360–430 |
| 2× RTX 4090 | 2× 24 GiB | ~2 GiB free | 32K | 2–3 | ~$300–430 |

**Pick the A100 80GB.** It's the sweet spot — full 256K context, 10+ concurrent users, $450–576/mo. The L40S works but is tight on headroom.

### Quick Start

```bash
# On Vast.ai A100 80GB instance (vastai/vllm template)
pkill -f 'vllm serve'

vllm serve Qwen/Qwen3-Coder-Next \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --download-dir /workspace/models \
  --host 0.0.0.0 --port 8001 \
  --api-key $API_KEY
```

AWQ 4-bit variant for tighter GPUs:

```bash
vllm serve cyankiwi/Qwen3-Coder-Next-AWQ-4bit \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --download-dir /workspace/models \
  --host 0.0.0.0 --port 8001 \
  --api-key $API_KEY
```

---

## Alternative: MiniMax M2.5

Higher quality per token, but expensive and limited concurrency. Best for single-user setups where thinking mode matters.

See [`m2-5-deploy.md`](m2-5-deploy.md) for the full deployment guide (AWQ 4-bit on 1× H200, 65K context, 4 concurrent users, ~$1,545/mo).

Not recommended for team usage — $1,545/mo for 4 concurrent users vs $500/mo for 10+ with Qwen3-Coder-Next.

---

## Connect (from any device)

vLLM is exposed directly via a Cloudflare Tunnel with `--api-key` auth. Single hop, no proxy.

Copy `.env.example` to `.env` and get the values from the deployer.

```bash
cp .env.example .env
```

```
Base URL:  $M25_BASE_URL
API Key:   $M25_API_KEY
Model:     Qwen/Qwen3-Coder-Next  (or QuantTrio/MiniMax-M2.5-AWQ)
```

### Cursor / Continue / Claw

| Setting | Value |
|---------|-------|
| Base URL | `$M25_BASE_URL` |
| API Key | `$M25_API_KEY` |
| Model | `Qwen/Qwen3-Coder-Next` |

### Python

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["M25_BASE_URL"],
    api_key=os.environ["M25_API_KEY"]
)

r = client.chat.completions.create(
    model="Qwen/Qwen3-Coder-Next",
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
  -d '{"model":"Qwen/Qwen3-Coder-Next","messages":[{"role":"user","content":"Hello"}],"max_tokens":500}'
```

### How it works

```
Client (any device)
  → Cloudflare Tunnel (public HTTPS)
    → vLLM :8001 (--api-key auth, inference)
```

Vast.ai's Caddy proxy intercepts port 8000. We use port 8001 and our own `cloudflared` tunnel. See [`endpoints.yaml`](endpoints.yaml) for restart instructions.

---

## Cost

| Setup | GPU | $/mo (Vast.ai) | Context | Users | SWE-Bench |
|-------|-----|----------------|---------|-------|-----------|
| **Qwen3-Coder-Next** | 1× A100 80GB | **~$450–576** | **256K** | **10–15** | 70.6% |
| Qwen3-Coder-Next | 1× L40S 48GB | ~$360–430 | 32–64K | 3–5 | 70.6% |
| MiniMax M2.5 AWQ | 1× H200 141GB | ~$1,545 | 65K | 4 | 80.2% |

For comparison: Claude Opus API is $75/M output tokens. Qwen3-Coder-Next self-hosted runs ~$5–15/M output.

---

## Files

| File | Purpose |
|------|---------|
| [`env-setup.md`](env-setup.md) | Shared Vast.ai environment setup (SSH, tunnels, clients) |
| [`m2-5-deploy.md`](m2-5-deploy.md) | MiniMax M2.5 AWQ/vLLM deployment guide |
| [`endpoints.yaml`](endpoints.yaml) | Public tunnel URLs and client config examples |
| [`.env.example`](.env.example) | Template for API credentials (copy to `.env`) |

---

## Caveats

- **Qwen3-Coder-Next has no thinking mode.** It doesn't reason before answering. For complex multi-step planning, M2.5 is measurably better — but for straightforward coding tasks the difference is small.
- **No vision.** Both models are text/code only.
- **Vast.ai is a marketplace.** Machines vanish. Use for dev; RunPod for production.
- **All throughput numbers are estimates.** Varies by context length, batch size, and prompt structure.
