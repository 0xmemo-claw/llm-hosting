# MiniMax M2.5 — Vast.ai Deployment Guide

Based on [0xmemo-claw/llm-hosting](https://github.com/0xmemo-claw/llm-hosting) — corrected with real-world deployment findings.

---

## Overview

Deploy **MiniMax M2.5** (230B params, 10B active MoE) on Vast.ai using vLLM. This gives you an OpenAI-compatible API with interleaved thinking, tool calling, and 80.2% SWE-Bench Verified performance.

### Critical: Model Sizing

The base FP8 model (`MiniMaxAI/MiniMax-M2.5`) requires **~220 GB VRAM** for weights alone. The official MiniMax documentation specifies a **minimum of 4× H200/H100 GPUs** with `--tensor-parallel-size 4`.

**A single H200 (141 GB) cannot run the base FP8 model.** This guide covers two deployment paths:

| Path | Model | GPUs | VRAM Needed | Context |
|------|-------|------|-------------|---------|
| **A — Single GPU (AWQ 4-bit)** | `QuantTrio/MiniMax-M2.5-AWQ` | 1× H200 | ~112 GB (with CPU offload) | 65K |
| **B — Multi-GPU (FP8 native)** | `MiniMaxAI/MiniMax-M2.5` | 4× H200/H100 | ~220 GB | 32–65K |

Path A uses `--cpu-offload-gb` to move model weights to system RAM, freeing GPU for KV cache. It trades some quality (INT4 vs FP8) and inference speed for dramatically lower cost. Path B is the official recommended deployment.

---

## Step 1: Rent GPU(s) on Vast.ai

### Pick the Right Template

From the Vast.ai template list, select:

> **vLLM** — `vastai/vllm` (Cuda 12.9, SSH, Jupyter)

This image has vLLM pre-installed (v0.15.x+ with MiniMax M2.5 support).

### Pick the Right Machine

**For Path A (single GPU, AWQ):**

| Requirement  | Value                                   |
| ------------ | --------------------------------------- |
| GPU          | 1× H200 141GB                           |
| Disk         | ≥250 GB (model weights + overhead)      |
| RAM          | ≥64 GB system RAM                       |
| Reliability  | ≥99.5%                                  |
| Max Duration | ≥7 days (longer = better for stability) |

**For Path B (multi-GPU, FP8):**

| Requirement  | Value                                   |
| ------------ | --------------------------------------- |
| GPU          | 4× H200 141GB or 4× H100 80GB          |
| Disk         | ≥500 GB                                 |
| RAM          | ≥128 GB system RAM                      |

### How to Search on Vast.ai

1. Go to [console.vast.ai](https://console.vast.ai)
2. Select the **vLLM** template (`vastai/vllm`)
3. Filter: GPU Type → **H200**, GPU Count → **1** (Path A) or **4** (Path B), Disk Space → **≥250 GB**
4. Sort by **$/hr** or **DLP/$/hr** (value metric)
5. Rent the instance

---

## Step 2: Connect to Your Instance

Once the instance is running, Vast.ai provides SSH connection details on the instance page.

```bash
ssh -p <PORT> root@<HOST> -i <PRIVATE_KEY> -L 8001:localhost:8001
```

The `-L` flag tunnels vLLM (port 8001) to your local machine. We use port 8001 instead of 8000 because Vast.ai's Caddy proxy intercepts port 8000 and adds cookie-based auth that breaks API clients.

Alternatively, open the **Jupyter** interface from the Vast.ai dashboard for a web terminal.

---

## Step 3: Stop the Default Model

The `vastai/vllm` image auto-starts a small default model (e.g. DeepSeek-R1-Distill-Llama-8B). Kill it first:

```bash
pkill -f 'vllm serve'
sleep 3

# Confirm GPU is free
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
# Should show: 0 MiB, 143157 MiB
```

---

## Step 4: Serve the Model with vLLM

### Path A: Single H200 — AWQ 4-bit (Tested & Working)

Uses [QuantTrio/MiniMax-M2.5-AWQ](https://huggingface.co/QuantTrio/MiniMax-M2.5-AWQ), a data-free INT4 AWQ quantization (~122 GiB download).

The key to getting high context on a single GPU is `--cpu-offload-gb`, which moves model weights from GPU to system RAM, freeing VRAM for KV cache. The instance needs ≥64 GB system RAM (most Vast.ai H200 instances have 500 GB–2 TiB).

**65K context (recommended):**

```bash
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve QuantTrio/MiniMax-M2.5-AWQ \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.98 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --download-dir /workspace/models \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --enforce-eager \
  --swap-space 16 \
  --cpu-offload-gb 50 \
  --max-num-seqs 4 \
  --api-key <YOUR_API_KEY>
```

**32K context (lower latency, no CPU offload overhead):**

```bash
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve QuantTrio/MiniMax-M2.5-AWQ \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.97 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --download-dir /workspace/models \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --enforce-eager \
  --swap-space 16 \
  --cpu-offload-gb 30 \
  --api-key <YOUR_API_KEY>
```

**What each flag does:**

| Flag | Purpose |
| ---- | ------- |
| `--max-model-len 65536` | Max context window (65K tokens). Use `32768` for lower latency. |
| `--gpu-memory-utilization 0.98` | Use 98% of VRAM. Required for 65K — lower values won't fit the KV cache. Use 0.97 for 32K. Do NOT go to 0.99 — OOMs during sampler warmup. |
| `--cpu-offload-gb 50` | Offloads ~9 GiB of model weights to CPU RAM, freeing GPU for KV cache. This is the key to fitting 65K context. Use 30 for 32K context. Adds slight inference latency. |
| `--max-num-seqs 4` | Limits concurrent sequences to 4. Required at 0.98 util to avoid OOM during warmup. |
| `--enforce-eager` | Disables CUDA graph capture, saving ~2 GiB VRAM. Required for this memory-tight deployment. |
| `--tool-call-parser minimax_m2` | Required for correct function-calling behavior. |
| `--reasoning-parser minimax_m2_append_think` | Correct parser for M2.5 interleaved thinking. Do NOT use `deepseek_r1` — it produces empty `content` fields. |
| `--trust-remote-code` | Required — M2.5 uses custom model code from HuggingFace. |
| `--swap-space 16` | 16 GB CPU swap space for overflow. |
| `--port 8001` | Use 8001 instead of 8000. Vast.ai's Caddy proxy intercepts port 8000 with cookie auth that breaks API clients. |
| `--api-key <KEY>` | Enables Bearer token auth on the vLLM server. Clients must send `Authorization: Bearer <KEY>`. |
| `VLLM_USE_DEEP_GEMM=0` | Disables DeepGEMM kernel (recommended by QuantTrio for AWQ). |
| `VLLM_USE_FLASHINFER_MOE_FP16=1` | Uses FlashInfer for FP16 MoE operations. |

**First run downloads ~122 GiB of model weights from HuggingFace.** Subsequent starts load from `/workspace/models` cache in ~80 seconds.

Monitor loading with:

```bash
watch -n 2 nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
```

### Path B: 4× GPUs — FP8 Native (Official Recommended)

Uses the base model [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5). This is the official deployment from the [MiniMax vLLM guide](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/docs/vllm_deploy_guide.md).

**4-GPU deployment (TP=4):**

```bash
SAFETENSORS_FAST_GPU=1 vllm serve \
  MiniMaxAI/MiniMax-M2.5 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think
```

**8-GPU deployment (TP=8, expert parallel):**

```bash
SAFETENSORS_FAST_GPU=1 vllm serve \
  MiniMaxAI/MiniMax-M2.5 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think
```

Multi-GPU gives full 32–65K context, higher throughput, and full FP8 quality.

---

## Step 5: Verify the API is Running

Once loading completes, vLLM serves an OpenAI-compatible API on port `8001`.

```bash
# List models
curl -H "Authorization: Bearer <YOUR_API_KEY>" http://localhost:8001/v1/models

# Test a completion
curl http://localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "QuantTrio/MiniMax-M2.5-AWQ",
    "messages": [{"role": "user", "content": "Hello, what model are you?"}],
    "max_tokens": 500
  }'
```

For Path B, replace the model name with `MiniMaxAI/MiniMax-M2.5`.

The response includes `<think>...</think>` reasoning blocks followed by the final answer. Budget 500+ max_tokens — the model reasons extensively before answering.

---

## Step 6: Expose the API

### Option A: Cloudflare Tunnel (Recommended — works from any device)

Start a `cloudflared` quick tunnel pointing at vLLM on port 8001. This bypasses Vast.ai's built-in tunnels (which add cookie auth that breaks API clients).

```bash
nohup /opt/instance-tools/bin/cloudflared tunnel --url http://localhost:8001 \
  > /var/log/cloudflared-vllm.log 2>&1 &

# Get the public URL
grep 'trycloudflare.com' /var/log/cloudflared-vllm.log
```

The tunnel URL (e.g. `https://some-random-words.trycloudflare.com`) is your public API endpoint. Auth is handled by vLLM's `--api-key` flag — no nginx or additional proxy needed.

> Tunnel URLs change on restart. For a stable URL, set up a named Cloudflare Tunnel with a custom domain.

### Option B: SSH Tunnel

From your local machine:

```bash
ssh -p <VAST_PORT> root@<VAST_HOST> -i <PRIVATE_KEY> \
  -L 8001:localhost:8001
```

Then use `http://localhost:8001/v1` from your apps.

---

## Step 7: Connect Clients

| Setting | Value |
|---------|-------|
| Base URL | `https://<tunnel-url>.trycloudflare.com/v1` or `http://localhost:8001/v1` (SSH) |
| API Key | The key you set with `--api-key` |
| Model | `QuantTrio/MiniMax-M2.5-AWQ` |

### Any OpenAI-Compatible Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<tunnel-url>.trycloudflare.com/v1",
    api_key="<YOUR_API_KEY>"
)

response = client.chat.completions.create(
    model="QuantTrio/MiniMax-M2.5-AWQ",
    messages=[{"role": "user", "content": "Write a Python quicksort."}],
    max_tokens=2048
)
print(response.choices[0].message.content)
```

---

## VRAM & Context Reference

### Path A: Single H200 with AWQ 4-bit + CPU Offload

**65K config (`--cpu-offload-gb 50`, `--gpu-memory-utilization 0.98`):**

| Item | Memory |
|------|--------|
| AWQ model weights on GPU | ~112 GiB (rest offloaded to CPU) |
| H200 total VRAM | 141 GiB (139.8 GiB usable) |
| Available for KV cache | ~22.7 GiB |
| KV cache capacity | 95,872 tokens |
| Safe max context | 65K tokens |
| Max concurrent sequences | 4 (`--max-num-seqs 4`) |

**32K config (`--cpu-offload-gb 30`, `--gpu-memory-utilization 0.97`):**

| Item | Memory |
|------|--------|
| AWQ model weights on GPU | ~113 GiB |
| Available for KV cache | ~13.9 GiB |
| KV cache capacity | 58,800 tokens |
| Safe max context | 32K tokens |
| Max concurrent sequences | default |

### Path B: 4× H200 with FP8 Native

| Item | Memory |
|------|--------|
| FP8 model weights | ~220 GiB (split across GPUs) |
| Total VRAM (4×) | 564 GiB |
| Available for KV cache | ~340 GiB |
| Safe max context | 32–65K tokens |
| KV cache capacity | ~400K tokens total |

### Official Memory Requirements (from MiniMax)

- **Weights:** 220 GB
- **KV cache:** 240 GB per 1M context tokens
- **96 GB × 4 GPU:** ~400K total KV cache tokens
- **144 GB × 8 GPU:** up to 3M total KV cache tokens
- **Max per-sequence context:** 196K tokens (regardless of hardware)

---

## Troubleshooting

### "No available memory for the cache blocks"

The model weights leave no room for KV cache at the given `--gpu-memory-utilization`.

- **Path A fix:** Add `--cpu-offload-gb 50` to move weights to CPU RAM. Use `--gpu-memory-utilization 0.98` and `--enforce-eager`. If 65K still won't fit, reduce `--max-model-len 32768` and use `--cpu-offload-gb 30` with `--gpu-memory-utilization 0.97`.
- **Path B fix:** This shouldn't happen with 4+ GPUs. If it does, reduce `--max-model-len`.

### "CUDA out of memory occurred when warming up sampler"

`--gpu-memory-utilization` is too high (e.g. 0.99). Lower to 0.98, or also add `--max-num-seqs 4` to reduce warmup memory. Do NOT set utilization above 0.98 for single-GPU AWQ deployments.

### CUDA out of memory during model loading

The FP8 base model (`MiniMaxAI/MiniMax-M2.5`) requires ~141 GiB just for weight tensor initialization — more than a single H200 has. Use the AWQ model for single-GPU, or add more GPUs.

### "assert self.kv_cache_dtype in {'fp8', 'fp8_e4m3'}"

Do NOT use `--kv-cache-dtype fp8_e5m2` with the AWQ model. The attention backend only accepts `fp8` or `fp8_e4m3`. However, FP8 KV cache with AWQ FP16 inputs causes dtype mismatches — omit the `--kv-cache-dtype` flag entirely for Path A.

### "trust_remote_code" error

Add `--trust-remote-code` to the vLLM command. M2.5 uses custom model code hosted on HuggingFace.

### Empty `content` field with reasoning in response

You're using the wrong reasoning parser. Use `--reasoning-parser minimax_m2_append_think` (NOT `deepseek_r1`). The `deepseek_r1` parser separates reasoning into `reasoning_content` and can leave `content` null.

### Model download hangs or fails

```bash
df -h                                    # check disk space
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx  # set HuggingFace token if rate-limited
```

### vLLM version issues

```bash
pip install --upgrade vllm
# M2.5 support requires vLLM ≥0.13.0
# The vastai/vllm image ships v0.15.x which works
```

### Slow first response

Normal. Interleaved thinking adds TTFT latency — the model reasons before generating. Budget 2–5s for first token on complex prompts. With AWQ, expect slightly slower inference than FP8.

### Instance disappeared (Vast.ai)

Spot/interruptible instances can be reclaimed. For stability, use longer-duration listings or switch to RunPod.

### Default model keeps restarting

The `vastai/vllm` image runs a supervisor script that auto-starts a default model. After `pkill -f 'vllm serve'`, the supervisor may restart it. Kill the supervisor too if needed:

```bash
pkill -f 'vllm.sh'
sleep 2
pkill -f 'vllm serve'
```

---

## Quick Reference: Full Boot Sequence (Single H200)

```bash
# 1. SSH into your Vast.ai H200 instance
ssh -p <PORT> root@<HOST> -i <PRIVATE_KEY>

# 2. Kill the default model
pkill -f 'vllm serve'
sleep 3

# 3. Set environment variables
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# 4. Start vLLM with AWQ model (65K context, port 8001, API key auth)
nohup vllm serve QuantTrio/MiniMax-M2.5-AWQ \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.98 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --download-dir /workspace/models \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --enforce-eager \
  --swap-space 16 \
  --cpu-offload-gb 50 \
  --max-num-seqs 4 \
  --api-key <YOUR_API_KEY> \
  > /var/log/vllm-minimax.log 2>&1 &

# 5. Wait for model to load
#    First run: ~5 min (downloads 122 GiB)
#    Subsequent runs: ~80 seconds
tail -f /var/log/vllm-minimax.log  # watch for "Application startup complete"

# 6. Verify
curl -H "Authorization: Bearer <YOUR_API_KEY>" http://localhost:8001/v1/models

# 7. Start Cloudflare Tunnel for public access
nohup /opt/instance-tools/bin/cloudflared tunnel --url http://localhost:8001 \
  > /var/log/cloudflared-vllm.log 2>&1 &
grep 'trycloudflare.com' /var/log/cloudflared-vllm.log

# 8. Test
curl http://localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model":"QuantTrio/MiniMax-M2.5-AWQ","messages":[{"role":"user","content":"Hello!"}],"max_tokens":500}'
```

---

## Cost Summary

| Setup | GPUs | $/hr | $/month | Context | Quality |
| ----- | ---- | ---- | ------- | ------- | ------- |
| **Vast.ai 1× H200 (AWQ)** | 1× H200 | ~$2.15 | ~$1,545 | 65K | INT4 quantized, CPU offload |
| **Vast.ai 4× H200 (FP8)** | 4× H200 | ~$8.60 | ~$6,190 | 32–65K | Full FP8 |
| **Vast.ai 4× H100 (FP8)** | 4× H100 | ~$7.00 | ~$5,040 | 32K | Full FP8 |
| **RunPod 1× H200 (AWQ)** | 1× H200 | $3.59 | ~$2,585 | 65K | INT4 quantized, CPU offload |

For comparison, Claude Opus API costs $75/M output tokens. At sustained use, self-hosting M2.5 breaks even quickly.

---

## References

- [MiniMax M2.5 on HuggingFace](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [Official vLLM Deploy Guide](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/docs/vllm_deploy_guide.md)
- [QuantTrio AWQ Model](https://huggingface.co/QuantTrio/MiniMax-M2.5-AWQ)
- [vLLM MiniMax Recipe](https://docs.vllm.ai/projects/recipes/en/latest/MiniMax/MiniMax-M2.html)
