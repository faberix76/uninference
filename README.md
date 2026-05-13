# Uninference - High Customized and Optimized LLM Inference Server based on llama.cpp with CUDA Acceleration and CPU Offload for Mixture-of-Experts Models on Consumer Hardware

High-performance LLM inference server powered by [llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA GPU acceleration, designed for running models on consumer-grade hardware (8 GB VRAM / 20 GB RAM).

## What's Inside

- **`uninference/`** — Fork of `llama.cpp` with server binary (`llama-server`), CLI tooling, and shared libraries. Provides inference via an HTTP API compatible with the OpenAI chat completions format, plus built-in MCP (Model Context Protocol) proxy support.
- **`docker-files/`** — Single-stage Docker build for CUDA 12.8 + Ubuntu 24.04 with full GPU offload (`-ngl 999`, FA all quants). Produces a container image with `llama-server`, `llama-cli`, and `llama-bench`.
- **`server.py`** — Lightweight FastMCP server that runs alongside `llama-server`, exposing MCP tools, resources, and prompts with CORS support for browser-based clients.
- **`turboquant.sh`** — One-shot deployment script that launches the container with optimized flags: flash attention, turbo-quantized KV cache, large batch sizes, 128K context window, and MoE-friendly thread allocation.
- **`docs/models.md`** — Detailed analysis of MoE models compatible with the target hardware, memory budgeting, and router mode configuration.

## Quick Start

```bash
# 1. Place a GGUF model in models/
# 2. Build the Docker image
./uninference/build.sh

# 3. Deploy the server
./turboquant.sh
```

The server listens on `http://localhost:8080` with full OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/completions`, `/models`).

## Key Features

- **Hybrid architecture support** — Optimized for MoE models with linear/recurrent layers that minimise KV cache size (Qwen3.x 30B–35B, DeepSeek-Coder-V2-Lite).
- **Turbo-quantized KV cache** — K-cache at 4-bit, V-cache at 3-bit for maximum context within limited VRAM.
- **Router mode** — Serve multiple GGUF models from the same `models/` directory; clients select via the `model` field in requests.
- **MCP over HTTP** — Built-in `llama-server` MCP proxy plus a standalone FastMCP server for tool/resource/prompt definitions.

## Requirements

- NVIDIA GPU with 8+ GB VRAM (CUDA 12.8 compatible)
- 20+ GB system RAM
- Docker with NVIDIA container toolkit
