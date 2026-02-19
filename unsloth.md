# MiniMax M2.5 — Unsloth GGUF Deployment (High Context)

Alternative to the AWQ/vLLM approach in `deploy.md`. Uses Unsloth's Dynamic 3-bit quantization with llama.cpp for significantly higher context windows.

---

## Why Unsloth

The AWQ approach (`deploy.md`) loads the model at ~121 GiB on a single H200, leaving ~18–22 GiB for KV cache — enough for 65K context with CPU offloading. Unsloth's Dynamic 3-bit (UD-Q3_K_XL) is only **~101 GiB**, leaving **~40 GiB** for KV cache — enough for **128K+ context** without any CPU offloading tricks.

| Approach | Weights | KV Cache Room | Max Context | Inference Engine |
|----------|---------|---------------|-------------|------------------|
| AWQ 4-bit (deploy.md) | ~121 GiB | ~22 GiB (with CPU offload) | 65K | vLLM |
| **Unsloth UD-Q3_K_XL** | **~101 GiB** | **~40 GiB** | **128K+** | llama.cpp |
| Unsloth Q3_K_M (standard) | ~86 GiB | ~55 GiB | 196K | llama.cpp |

The tradeoff: Unsloth uses llama.cpp instead of vLLM. llama.cpp provides an OpenAI-compatible API via `llama-server`, but lacks vLLM's batched serving optimizations. Better for single-user / low-concurrency workloads where context length matters more than throughput.

---

## How Dynamic 2.0 Quantization Works

Standard quantization applies the same bit-width to every layer. Unsloth's Dynamic 2.0 quantization analyzes each layer's sensitivity and applies different quantization levels:

- **Critical layers** (attention projections, first/last layers): kept at 8 or 16-bit
- **Less sensitive layers** (most MoE expert weights): quantized to 3-bit
- **Result**: overall ~3-bit average with quality closer to 4-bit

This is why the `UD-` (Unsloth Dynamic) variants outperform standard `Q3_K_M` at similar sizes. The "XL" suffix means extra-large — more layers are kept at higher precision.

---

## Available Variants

From [unsloth/MiniMax-M2.5-GGUF](https://huggingface.co/unsloth/MiniMax-M2.5-GGUF):

| Variant | Size | Shards | Quality | Fits H200? | Context Headroom |
|---------|------|--------|---------|------------|------------------|
| **UD-Q3_K_XL** | ~101 GiB | 4 | Best for size | Yes | ~40 GiB → 128K+ |
| UD-Q4_K_XL | ~120 GiB | 4 | Higher quality | Tight | ~20 GiB → 65K |
| Q3_K_M (standard) | ~86 GiB | 4 | Moderate | Yes | ~55 GiB → 196K |
| Q3_K_S (standard) | ~78 GiB | 3 | Lower | Yes | ~63 GiB → 196K |
| UD-Q2_K_XL | ~72 GiB | 3 | Reduced | Yes | ~69 GiB → 196K |
| Q4_K_M (standard) | ~120 GiB | 4 | Good | Tight | ~20 GiB → 65K |

**Recommended for H200: UD-Q3_K_XL** — best quality-to-size ratio, leaves enough room for 128K context.

---

## Deployment on Vast.ai H200

### Step 1: Build llama.cpp with CUDA

```bash
apt-get update && apt-get install -y cmake git
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
cd ..
```

### Step 2: Download the Model

```bash
pip install huggingface-hub

huggingface-cli download unsloth/MiniMax-M2.5-GGUF \
  --include "UD-Q3_K_XL/*" \
  --local-dir /workspace/models/minimax-m2.5-ud-q3
```

This downloads ~101 GiB in 4 shards. On Vast.ai H200 instances with fast networking, expect 10–30 minutes.

### Step 3: Start the Server

```bash
./llama.cpp/build/bin/llama-server \
  -m /workspace/models/minimax-m2.5-ud-q3/UD-Q3_K_XL/MiniMax-M2.5-UD-Q3_K_XL-00001-of-00004.gguf \
  --port 8000 \
  --host 0.0.0.0 \
  -c 131072 \
  -n 8192 \
  --n-gpu-layers 999 \
  --flash-attn \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 40 \
  --min-p 0.01
```

| Flag | Purpose |
|------|---------|
| `-c 131072` | Context window — 128K tokens. Can push to 196608 with Q3_K_M/Q3_K_S. |
| `-n 8192` | Max tokens per response. |
| `--n-gpu-layers 999` | Offload all layers to GPU. |
| `--flash-attn` | Enable flash attention for faster inference. |
| `--temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01` | Recommended sampling parameters from Unsloth/MiniMax. |

### Step 4: Verify

`llama-server` provides an OpenAI-compatible API:

```bash
# Health check
curl http://localhost:8000/health

# Test completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m2.5",
    "messages": [{"role": "user", "content": "Hello, what model are you?"}],
    "max_tokens": 500
  }'
```

---

## Context Length Comparison on H200

With UD-Q3_K_XL (~101 GiB weights) on a 141 GiB H200:

```
Total VRAM:        141 GiB
Model weights:     ~101 GiB
Available for KV:  ~40 GiB
```

| Context | KV Cache | Fits? |
|---------|----------|-------|
| 32K | ~8 GiB | Yes, comfortably |
| 65K | ~16 GiB | Yes |
| 128K | ~32 GiB | Yes |
| 196K | ~50 GiB | Tight — may need partial CPU offload with `--n-gpu-layers` tuning |

With Q3_K_M (~86 GiB weights), 196K context should fit entirely in GPU memory.

---

## Quality Comparison

Unsloth Dynamic 3-bit is **not as accurate** as AWQ 4-bit or FP8. Expect:

- Slightly more reasoning errors on complex multi-step tasks
- Occasional garbled output on long generations (rare with UD-Q3_K_XL, more common with standard Q3_K)
- SWE-Bench scores likely ~2–5% lower than FP8/AWQ deployments (no published benchmarks at Q3)

**When to pick Unsloth over AWQ:**
- You need >65K context consistently (document analysis, long codebases)
- Single-user workload where context matters more than throughput
- You want simpler deployment (llama.cpp has fewer configuration pitfalls than vLLM)

**When to stick with AWQ (deploy.md):**
- Multi-user serving with batching (vLLM is much better at this)
- 65K context is sufficient
- You need higher quality per token

---

## Throughput Expectations

| Hardware | Quant | tok/s (single stream) |
|----------|-------|----------------------|
| 1× H200 141GB | UD-Q3_K_XL | ~25–40 |
| 1× H100 80GB + CPU offload | UD-Q3_K_XL | ~15–25 |
| M4 Max 128GB unified | UD-Q3_K_XL | ~20–40 |

llama.cpp is single-stream — it processes one request at a time. For concurrent users, either run multiple instances or use the AWQ/vLLM approach from `deploy.md`.

---

## References

- [unsloth/MiniMax-M2.5-GGUF](https://huggingface.co/unsloth/MiniMax-M2.5-GGUF) — all quantization variants
- [Unsloth MiniMax-2.5 Guide](https://unsloth.ai/docs/models/minimax-2.5) — official Unsloth docs
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — inference engine
