# Vast.ai Environment Setup

Shared setup steps for any model deployment. Model-specific guides reference this doc.

---

## 1. Rent a GPU on Vast.ai

### Pick the Template

From the Vast.ai template list, select:

> **vLLM** — `vastai/vllm` (Cuda 12.9, SSH, Jupyter)

This image has vLLM pre-installed (v0.15.x+).

### How to Search

1. Go to [console.vast.ai](https://console.vast.ai)
2. Select the **vLLM** template (`vastai/vllm`)
3. Filter by GPU type, GPU count, and disk space (see your model guide for requirements)
4. Sort by **$/hr** or **DLP/$/hr** (value metric)
5. Rent the instance

---

## 2. Connect to Your Instance

Once the instance is running, Vast.ai provides SSH connection details on the instance page.

```bash
ssh -p <PORT> root@<HOST> -i <PRIVATE_KEY> -L 8001:localhost:8001
```

The `-L` flag tunnels vLLM (port 8001) to your local machine. We use port 8001 instead of 8000 because Vast.ai's Caddy proxy intercepts port 8000 and adds cookie-based auth that breaks API clients.

Alternatively, open the **Jupyter** interface from the Vast.ai dashboard for a web terminal.

---

## 3. Stop the Default Model

The `vastai/vllm` image auto-starts a small default model. Kill it first:

```bash
pkill -f 'vllm serve'
sleep 3

# Confirm GPU is free
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
```

If the default model keeps restarting, kill the supervisor script too:

```bash
pkill -f 'vllm.sh'
sleep 2
pkill -f 'vllm serve'
```

---

## 4. Expose the API

After vLLM is running (see your model guide), expose it publicly.

### Option A: Cloudflare Tunnel (Recommended)

Start a `cloudflared` quick tunnel pointing at vLLM on port 8001. This bypasses Vast.ai's built-in tunnels (which add cookie auth that breaks API clients).

```bash
nohup /opt/instance-tools/bin/cloudflared tunnel --url http://localhost:8001 \
  > /var/log/cloudflared-vllm.log 2>&1 &

# Get the public URL
grep 'trycloudflare.com' /var/log/cloudflared-vllm.log
```

The tunnel URL (e.g. `https://some-random-words.trycloudflare.com`) is your public API endpoint. Auth is handled by vLLM's `--api-key` flag.

> Tunnel URLs change on restart. For a stable URL, set up a named Cloudflare Tunnel with a custom domain.

### Option B: SSH Tunnel

From your local machine:

```bash
ssh -p <VAST_PORT> root@<VAST_HOST> -i <PRIVATE_KEY> \
  -L 8001:localhost:8001
```

Then use `http://localhost:8001/v1` from your apps.

---

## 5. Connect Clients

| Setting | Value |
|---------|-------|
| Base URL | `https://<tunnel-url>.trycloudflare.com/v1` or `http://localhost:8001/v1` (SSH) |
| API Key | The key you set with `--api-key` |
| Model | The full model name (e.g. `Qwen/Qwen3-Coder-Next`) |

### Python

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["M25_BASE_URL"],
    api_key=os.environ["M25_API_KEY"]
)

r = client.chat.completions.create(
    model="<MODEL_NAME>",
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
  -d '{"model":"<MODEL_NAME>","messages":[{"role":"user","content":"Hello"}],"max_tokens":500}'
```

### Verify

```bash
# List models
curl -H "Authorization: Bearer $M25_API_KEY" $M25_BASE_URL/models

# Health check (on instance)
curl http://localhost:8001/health
```

---

## Troubleshooting (General)

### Model download hangs or fails

```bash
df -h                                    # check disk space
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx  # set HuggingFace token if rate-limited
```

### vLLM version issues

```bash
pip install --upgrade vllm
# The vastai/vllm image ships v0.15.x
```

### Instance disappeared (Vast.ai)

Spot/interruptible instances can be reclaimed. For stability, use longer-duration listings or switch to RunPod.

### Monitor GPU during loading

```bash
watch -n 2 nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
```
