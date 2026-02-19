# MiniMax M2.5 — Vast.ai Deployment Guide

Based on [0xmemo-claw/llm-hosting](https://github.com/0xmemo-claw/llm-hosting) — corrected with real-world deployment findings.

> **Prerequisites:** Complete steps 1–3 in [env-setup.md](env-setup.md) first.

---

## Overview

Deploy **MiniMax M2.5** (230B params, 10B active MoE) on Vast.ai using vLLM. This gives you an OpenAI-compatible API with interleaved thinking, tool calling, and 80.2% SWE-Bench Verified performance.

### Model Sizing

The base FP8 model (`MiniMaxAI/MiniMax-M2.5`) requires **~220 GB VRAM** for weights alone — a **minimum of 4× H200/H100 GPUs**. This guide covers two paths:

| Path | Model | GPUs | VRAM Needed | Context |
|------|-------|------|-------------|---------|
| **A — Single GPU (AWQ 4-bit)** | `QuantTrio/MiniMax-M2.5-AWQ` | 1× H200 | ~112 GB (with CPU offload) | 65K |
| **B — Multi-GPU (FP8 native)** | `MiniMaxAI/MiniMax-M2.5` | 4× H200/H100 | ~220 GB | 32–65K |

Path A uses `--cpu-offload-gb` to move model weights to system RAM, freeing GPU for KV cache. Path B is the official recommended deployment.

---

## Hardware Requirements

**Path A (single GPU, AWQ):**

| Requirement  | Value                                   |
| ------------ | --------------------------------------- |
| GPU          | 1× H200 141GB                           |
| Disk         | ≥250 GB (model weights + overhead)      |
| RAM          | ≥64 GB system RAM                       |
| Reliability  | ≥99.5%                                  |
| Max Duration | ≥7 days (longer = better for stability) |

**Path B (multi-GPU, FP8):**

| Requirement  | Value                                   |
| ------------ | --------------------------------------- |
| GPU          | 4× H200 141GB or 4× H100 80GB          |
| Disk         | ≥500 GB                                 |
| RAM          | ≥128 GB system RAM                      |

When searching on Vast.ai, filter: GPU Type → **H200**, GPU Count → **1** (Path A) or **4** (Path B), Disk Space → **≥250 GB**.

---

## Serve the Model

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

**Flag reference:**

| Flag | Purpose |
| ---- | ------- |
| `--max-model-len 65536` | Max context window (65K tokens). Use `32768` for lower latency. |
| `--gpu-memory-utilization 0.98` | Use 98% of VRAM. Required for 65K — lower values won't fit KV cache. Use 0.97 for 32K. Do NOT go to 0.99 — OOMs during sampler warmup. |
| `--cpu-offload-gb 50` | Offloads ~9 GiB of model weights to CPU RAM, freeing GPU for KV cache. Key to fitting 65K context. Use 30 for 32K. Adds slight latency. |
| `--max-num-seqs 4` | Limits concurrent sequences to 4. Required at 0.98 util to avoid OOM during warmup. |
| `--enforce-eager` | Disables CUDA graph capture, saving ~2 GiB VRAM. Required for this memory-tight deployment. |
| `--tool-call-parser minimax_m2` | Required for correct function-calling behavior. |
| `--reasoning-parser minimax_m2_append_think` | Correct parser for M2.5 interleaved thinking. Do NOT use `deepseek_r1` — it produces empty `content` fields. |
| `--trust-remote-code` | Required — M2.5 uses custom model code from HuggingFace. |
| `--swap-space 16` | 16 GB CPU swap space for overflow. |
| `--port 8001` | Use 8001 not 8000. See [env-setup.md](env-setup.md) for why. |
| `--api-key <KEY>` | Enables Bearer token auth. Clients must send `Authorization: Bearer <KEY>`. |
| `VLLM_USE_DEEP_GEMM=0` | Disables DeepGEMM kernel (recommended by QuantTrio for AWQ). |
| `VLLM_USE_FLASHINFER_MOE_FP16=1` | Uses FlashInfer for FP16 MoE operations. |

**First run downloads ~122 GiB of model weights.** Subsequent starts load from `/workspace/models` cache in ~80 seconds.

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

## After Serving

Once vLLM prints "Application startup complete", continue with steps 4–5 in [env-setup.md](env-setup.md) to expose the API and connect clients. Use `QuantTrio/MiniMax-M2.5-AWQ` (Path A) or `MiniMaxAI/MiniMax-M2.5` (Path B) as the model name.

The response includes `<think>...</think>` reasoning blocks followed by the final answer. Budget 500+ max_tokens — the model reasons extensively before answering.

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

## Troubleshooting (M2.5-Specific)

### "No available memory for the cache blocks"

The model weights leave no room for KV cache at the given `--gpu-memory-utilization`.

- **Path A fix:** Add `--cpu-offload-gb 50` to move weights to CPU RAM. Use `--gpu-memory-utilization 0.98` and `--enforce-eager`. If 65K still won't fit, reduce `--max-model-len 32768` and use `--cpu-offload-gb 30` with `--gpu-memory-utilization 0.97`.
- **Path B fix:** This shouldn't happen with 4+ GPUs. If it does, reduce `--max-model-len`.

### "CUDA out of memory occurred when warming up sampler"

`--gpu-memory-utilization` is too high (e.g. 0.99). Lower to 0.98, or also add `--max-num-seqs 4` to reduce warmup memory. Do NOT set utilization above 0.98 for single-GPU AWQ deployments.

### CUDA out of memory during model loading

The FP8 base model requires ~141 GiB just for weight tensor initialization — more than a single H200 has. Use the AWQ model for single-GPU, or add more GPUs.

### "assert self.kv_cache_dtype in {'fp8', 'fp8_e4m3'}"

Do NOT use `--kv-cache-dtype fp8_e5m2` with the AWQ model. Omit the `--kv-cache-dtype` flag entirely for Path A.

### Empty `content` field with reasoning in response

You're using the wrong reasoning parser. Use `--reasoning-parser minimax_m2_append_think` (NOT `deepseek_r1`). The `deepseek_r1` parser separates reasoning into `reasoning_content` and can leave `content` null.

### Slow first response

Normal. Interleaved thinking adds TTFT latency — the model reasons before generating. Budget 2–5s for first token on complex prompts. With AWQ, expect slightly slower inference than FP8.

> For general troubleshooting (downloads, vLLM version, Vast.ai issues), see [env-setup.md](env-setup.md#troubleshooting-general).

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

---

## References

- [MiniMax M2.5 on HuggingFace](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [Official vLLM Deploy Guide](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/docs/vllm_deploy_guide.md)
- [QuantTrio AWQ Model](https://huggingface.co/QuantTrio/MiniMax-M2.5-AWQ)
- [vLLM MiniMax Recipe](https://docs.vllm.ai/projects/recipes/en/latest/MiniMax/MiniMax-M2.html)
