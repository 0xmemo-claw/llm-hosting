# Router setup options

## Option A (recommended): vLLM per model + LiteLLM router

- Run 1 vLLM server per model (best isolation).
- Route with LiteLLM so apps always hit one OpenAI-compatible endpoint.

Pros:
- simple mental model
- per-model autoscaling possible
- easy to swap models

Cons:
- costs more if you keep 3 GPUs always-on

## Option B: single GPU, multiple models (NOT recommended)

Possible but painful: memory fragmentation, unload/reload overhead, bad latency.

## OpenClaw integration

Point your OpenClaw OpenAI-compatible provider at LiteLLM:

- base_url: `http(s)://<host>:8000/v1`
- api_key: LiteLLM master_key (if enabled)

Then in OpenClaw config, use logical model aliases `haiku|sonnet|opus`.
