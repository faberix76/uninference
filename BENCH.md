# Turboquant Benchmarking Guide

This document describes how to benchmark inference performance of the turboquant
container using **llama-benchy**, a llama-bench-style benchmarking tool that works
with any OpenAI-compatible LLM endpoint.

## Prerequisites

- **turboquant container running** (start with `./turboquant.sh`)
- **llama-benchy** installed at `benchy/` (already done — see Installation)
- **`uv`** installed on the host

## Quick Start

```bash
# Static benchmark (pp2048, tg2048, depth 8192, 2 runs)
./static-bench.sh
```

## Installation

llama-benchy is already cloned into `benchy/` with its virtual environment
set up. To refresh or reinstall:

```bash
cd benchy
uv sync --all-extras --dev
```

## static-bench.sh Reference

`static-bench.sh` is a static benchmark configuration that runs llama-benchy
directly with a fixed set of parameters for the Qwen3.6-35B-A3B model.

The script runs from `benchy/` and passes all arguments to `llama-benchy`.
See [llama-benchy options](https://github.com/eugr/llama-benchy) for details.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TQ_BASE_URL` | Override default base URL | `http://localhost:8080/v1` |

## Output Metrics

| Metric | Description |
|--------|-------------|
| **t/s** | Tokens per second (pp = prefill speed, tg = decode speed) |
| **peak t/s** | Max tokens/sec in any 1-second window (tg only) |
| **ttfr (ms)** | Time To First Response chunk |
| **est_ppt (ms)** | Estimated prompt processing time (ttfr − latency) |
| **e2e_ttft (ms)** | End-to-end Time To First Token |

When `concurrency > 1`:
- **t/s (total)**: aggregate throughput across all clients
- **t/s (req)**: per-client throughput

When `--enable-prefix-caching` is used:
- **ctx_pp**: time to load and cache the context
- **pp @ depth**: prompt processing with cached prefix

## Further Analysis

Use `--format json --save-result bench.json` to export full data including
per-request timeseries. See `benchy/examples/benchmark_visualization.ipynb`
for Jupyter-based analysis.

## Latency Mode Recommendation

For production-like measurements, use `--latency-mode generation`. This measures
the time to generate a single token and subtracts it from TTFR to estimate
server-side prompt processing time more accurately.

The default `api` mode (fetching `/v1/models`) only eliminates network round-trip
and may overestimate processing time on short prompts.

## Notes

- The container runs at `/v1/chat/completions` only (no `/v1/completions`).
- llama-benchy automatically downloads a book from Project Gutenberg for
  realistic multi-turn text (no random tokens).
- If you get cache hits unexpectedly, add `--no-cache`.
- With very large context depths (e.g., 32768+), consider increasing timeout
  via `--runs 1` (single run per test) to reduce total benchmark time.
