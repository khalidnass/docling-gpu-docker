# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-optimized Docker image for [docling-serve](https://github.com/docling-project/docling-serve) (v1.12.0) with CUDA 12.8. Fixes 8 critical bugs in the official `docling-serve-cu128` image that cause GPU underperformance (ONNX on CPU, missing CUDA libs, RapidOCR ignoring GPU, torch.compile crashes, etc.).

## Build & Run

```bash
# Build (requires models/ directory with pre-downloaded models in build context)
DOCKER_BUILDKIT=1 docker build -t docling-serve-cu128-custom .

# Run
docker run --gpus all -p 5001:5001 docling-serve-cu128-custom

# Verify GPU support
docker run --gpus all --rm docling-serve-cu128-custom python3 -c "
import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers()
import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
"
```

## Diagnostics & Benchmarking

```bash
# Run diagnostics (inside container or locally with deps)
python3 test_docling.py --diag

# Benchmark standard pipeline (tasks 1-4) and VLM pipeline (tasks 5-8)
python3 test_docling.py --benchmark --base-url http://localhost:5001 --input-dir ./files
python3 test_docling.py --benchmark --base-url http://localhost:5001 --input-dir ./files --tasks 1 2 5 6

# Compare official vs custom image performance
./benchmark.sh

# Cloud (H100/H200) diagnostics - copy-paste into OpenShift pod
bash h20_diag.sh
```

### Debug Image (with full pipeline profiling)

```bash
# Build debug image with per-stage timing, per-crop OCR logging, VLM tok/s tracking
DOCKER_BUILDKIT=1 docker build -f Dockerfile.debug -t docling-serve-debug .

# Run and view profiling output
docker run --gpus all -p 5001:5001 docling-serve-debug
docker logs <id> 2>&1 | grep "\[PROFILE\]"

# Run standalone model benchmarks inside debug container
docker exec -it <id> python3 /opt/app-root/src/benchmark_models.py
docker exec -it <id> python3 /opt/app-root/src/benchmark_onnx_sessions.py
```

## Architecture

### Dockerfile (multi-stage build)

**Stage 1 - Builder** (`nvidia/cuda:12.8.0-devel-ubuntu24.04`):
- Python 3.12 via deadsnakes PPA, `uv` package manager
- Clones docling-serve v1.12.0, installs via two-pass `uv sync` (flash-attn uses prebuilt wheels for compute cap 8.0+)
- Swaps `onnxruntime` (CPU) → `onnxruntime-gpu` with build-time CUDAExecutionProvider verification
- Security upgrades (pillow, cryptography) done here so the COPY layer is Trivy-clean

**Stage 2 - Runtime** (`nvidia/cuda:12.8.0-runtime-ubuntu24.04`):
- Copies venv from builder; wires NVIDIA pip library `.so` files via `ldconfig`
- Pre-bakes 5 ML models (~706 MB): layout-heron, tableformer, DocumentFigureClassifier-v2.0, RapidOcr, EasyOcr
- Patches `rapid_ocr_model.py` via `sed` to set `EngineConfig.onnxruntime.use_cuda`
- Entrypoint shim ensures DocumentFigureClassifier v2.0 survives volume mounts
- Runs as non-root UID 1001

### Key Bug Fixes

| # | Bug | Fix |
|---|-----|-----|
| 1 | ONNX on CPU | Swap onnxruntime → onnxruntime-gpu |
| 2 | flash-attn can't compile | Multi-stage: compile in devel, prebuilt wheels for H100 |
| 3 | CUDA libs missing | ldconfig wiring for pip nvidia packages |
| 4 | cuDNN unavailable | ldconfig wires nvidia-cudnn-cu12 from pytorch |
| 5 | RapidOCR on CPU | sed patch to set EngineConfig.onnxruntime.use_cuda |
| 6 | Classifier v1/v2 conflict | Bake v2.0 + entrypoint copy fallback |
| 7 | torch.compile crashes | TORCH_COMPILE_DISABLE=1 |
| 8 | Security CVEs | Upgrade pillow, cryptography in builder stage |

### Dockerfile.debug (instrumented build)

Extends `Dockerfile` with:
- `print(..., flush=True)` profiling patches on TimeRecorder, RapidOCR (per-crop), layout, table structure, pipeline summary
- `sitecustomize.py` at `/usr/lib/python3.12/` enables `settings.debug.profile_pipeline_timings` automatically
- Includes `ibm-granite/granite-docling-258M` VLM model and standalone benchmark scripts
- All `[PROFILE]` output uses `print()` not `logging` (uvicorn workers don't inherit Python logging config)

### Pre-loaded Models (in `models/` directory, gitignored)

- `docling-project--docling-layout-heron` — RT-DETR v2 layout detection (164 MB)
- `docling-project--docling-models` — Tableformer table structure (342 MB)
- `docling-project--DocumentFigureClassifier-v2.0` — EfficientNet B0 (32 MB)
- `RapidOcr` — PP-OCRv4 ONNX (59 MB)
- `EasyOcr` — Fallback OCR (109 MB)
- VLM: `ibm-granite/granite-vision-3.3-2b` downloaded at runtime on CUDA
- VLM: `ibm-granite/granite-docling-258M` — 258M param single-pass VLM (debug image only)

### Key Environment Variables

- `TORCH_COMPILE_DISABLE=1` / `TORCHINDUCTOR_DISABLE=1` — prevents GPU crashes
- `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` — threading tuning
- `DOCLING_SERVE_ARTIFACTS_PATH` / `DOCLING_ARTIFACTS_PATH` — model cache paths
- `DOCLING_PICTURE_DESCRIPTION_VLM_DEVICE=cuda` — VLM runs on GPU
- `DOCLING_SERVE_ENG_LOC_NUM_WORKERS=2` — processing workers

## File Descriptions

- `Dockerfile` — Multi-stage GPU-optimized production build (the core of this project)
- `Dockerfile.debug` — Extended build with full pipeline profiling and VLM model
- `test_docling.py` — Diagnostics (10-check GPU report) and HTTP benchmark (8 task types: 4 standard + 4 VLM)
- `benchmark.sh` — Shell script comparing official vs custom image with automated container lifecycle
- `benchmark_models.py` — Standalone model benchmarks (RapidOCR, layout, table, VLM) bypassing docling
- `benchmark_onnx_sessions.py` — ONNX Runtime session options benchmark for RapidOCR tuning
- `vlm_bench_final.py` — Transformers BF16 vs vLLM BF16 comparison for granite-docling-258M
- `docling_debug_profiler.py` — Profiling monkey-patches for pipeline stages
- `debug_entrypoint.sh` — Enhanced entrypoint with startup diagnostics
- `INVESTIGATION.md` — Full pipeline investigation report with profiling data and optimization recommendations
- `h20_diag.sh` — Self-contained diagnostic for H100/H200 cloud pods
- `files/` — Test PDFs for benchmarking (gitignored)

## Performance Findings (A5000, CUDA 12.8)

### Standard Pipeline (via HTTP)
| File | Pages | std-default | std-rapidocr |
|------|-------|-------------|--------------|
| testcase1.pdf | 1 | 4.1s | 3.7s |
| testcase2.pdf | 5 | 11.5s | 11.7s |
| testcase3.pdf | 9 | 21.0s | 17.2s |
| testcase4.pdf | 21 | 46.2s | 47.1s |
| testcase5.pdf | 15 | 47.9s | 28.5s |
| testcase6.pdf | 15 | 14.0s | 7.7s |

### VLM Pipeline (granite-docling-258M)
- Transformers BF16: ~10.7 tok/s, ~420s per 15-page doc
- vLLM BF16: ~69.4 tok/s (6.8x faster than transformers)

### Key Bottleneck
OCR is the dominant cost (~80% of pipeline time) due to per-crop sequential processing. Layout and table structure are fast on GPU.
