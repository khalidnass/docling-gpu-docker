# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-optimized Docker image for [docling-serve](https://github.com/docling-project/docling-serve) (v1.14.3) with CUDA 12.8. Fixes 8+ critical bugs in the official `docling-serve-cu128` image, adds vLLM v0.17 for accelerated VLM inference, switches OCR to PP-OCRv5, and bakes all standard pipeline models for zero-network-I/O startup.

## Build & Run

```bash
# Build (requires models/ directory with pre-downloaded models in build context)
DOCKER_BUILDKIT=1 docker build -t docling-serve-cu128-custom:v1.14.3 .

# Run with VLM models mounted separately
docker run --gpus all -p 5001:5001 \
  -v /path/to/vlm-models:/opt/app-root/src/.cache/docling/models/vlm \
  docling-serve-cu128-custom:v1.14.3

# Run with Qwen VLM instead of granite
docker run --gpus all -p 5001:5001 \
  -v /path/to/vlm-models:/opt/app-root/src/.cache/docling/models/vlm \
  -e DOCLING_PICTURE_DESCRIPTION_VLM_MODEL_ID=/opt/app-root/src/.cache/docling/models/vlm/Qwen--Qwen2.5-VL-3B-Instruct \
  docling-serve-cu128-custom:v1.14.3

# Verify GPU support
docker run --gpus all --rm docling-serve-cu128-custom:v1.14.3 python3 -c "
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
DOCKER_BUILDKIT=1 docker build -f Dockerfile.debug -t docling-serve-debug:v1.14.3 .

# Run and view profiling output
docker run --gpus all -p 5001:5001 \
  -v /path/to/vlm-models:/opt/app-root/src/.cache/docling/models/vlm \
  docling-serve-debug:v1.14.3
docker logs <id> 2>&1 | grep "\[PROFILE\]"

# Run standalone model benchmarks inside debug container
docker exec -it <id> python3 /opt/app-root/src/benchmark_models.py
docker exec -it <id> python3 /opt/app-root/src/benchmark_onnx_sessions.py
```

## Architecture

### Dockerfile (multi-stage build)

**Stage 1 - Builder** (`nvidia/cuda:12.8.0-devel-ubuntu24.04`):
- Python 3.12 via deadsnakes PPA, `uv` package manager
- Clones docling-serve v1.14.3, installs via two-pass `uv sync` (flash-attn uses prebuilt wheels for compute cap 8.0+)
- Swaps `onnxruntime` (CPU) → `onnxruntime-gpu` with build-time CUDAExecutionProvider verification
- Upgrades RapidOCR to v3.7.0 (PP-OCRv5 support)
- Installs vLLM v0.17 for accelerated VLM inference (6.8x over transformers)
- Security upgrades (pillow, cryptography) done here so the COPY layer is Trivy-clean

**Stage 2 - Runtime** (`nvidia/cuda:12.8.0-runtime-ubuntu24.04`):
- Copies venv from builder; wires NVIDIA pip library `.so` files via `ldconfig`
- Pre-bakes 6 ML models (~1.3 GB): layout-heron, tableformer, DocumentFigureClassifier-v2.0, RapidOcr (v4+v5), EasyOcr, CodeFormulaV2
- Patches `rapid_ocr_model.py` to enable CUDA + switch to PP-OCRv5 server models
- VLM models mounted at `/opt/app-root/src/.cache/docling/models/vlm/` (separate from baked models)
- `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` prevents model downloads/updates
- Entrypoint shim ensures DocumentFigureClassifier v2.0 survives volume mounts
- Runs as non-root UID 1001

### Differences from Official `docling-serve-cu128` Image

| Feature | Official Image | This Custom Image |
|---------|---------------|-------------------|
| ONNX Runtime | CPU only | GPU (CUDAExecutionProvider) |
| flash-attn | Fails to compile | Prebuilt wheels (compute cap 8.0+) |
| CUDA libs | Not on LD_LIBRARY_PATH | Wired via ldconfig |
| cuDNN | Unavailable | Wired from pip nvidia-cudnn-cu12 |
| RapidOCR GPU | Ignores GPU (wrong config key) | Patched EngineConfig.onnxruntime.use_cuda |
| OCR models | PP-OCRv4 mobile | PP-OCRv5 server (more accurate) |
| torch.compile | Crashes on GPU | Disabled (TORCH_COMPILE_DISABLE=1) |
| Security CVEs | Present | Upgraded pillow, cryptography |
| vLLM | Not installed | v0.17 (6.8x faster VLM inference) |
| Models | Downloaded at runtime | 6 standard models baked, VLM mounted |
| Code/Formula | Model downloaded at runtime | CodeFormulaV2 baked |
| Network at startup | Required | Not required (HF_HUB_OFFLINE=1) |
| Classifier v2.0 | Can break with volume mounts | Baked + entrypoint fallback |

### Production vs Debug Image

| Layer | Production | Debug |
|-------|-----------|-------|
| Entrypoint | `container-entrypoint` | `debug-entrypoint` |
| Profiling patches | None | TimeRecorder, RapidOCR per-crop, pipeline summary |
| sitecustomize.py | None | dprint() + ENABLE_PROFILING |
| PYTHONUNBUFFERED | Not set | 1 |
| Debug scripts | None | benchmark_models.py, benchmark_onnx_sessions.py, test_docling.py |
| granite-docling-258M | Not baked | Baked |
| Everything else | Identical | Identical |

### Key Bug Fixes

| # | Bug | Fix |
|---|-----|-----|
| 1 | ONNX on CPU | Swap onnxruntime → onnxruntime-gpu |
| 2 | flash-attn can't compile | Multi-stage: compile in devel, prebuilt wheels for H100 |
| 3 | CUDA libs missing | ldconfig wiring for pip nvidia packages |
| 4 | cuDNN unavailable | ldconfig wires nvidia-cudnn-cu12 from pytorch |
| 5 | RapidOCR on CPU | Python str.replace patch for EngineConfig.onnxruntime.use_cuda |
| 6 | Classifier v1/v2 conflict | Bake v2.0 + entrypoint copy fallback |
| 7 | torch.compile crashes | TORCH_COMPILE_DISABLE=1 |
| 8 | Security CVEs | Upgrade pillow, cryptography in builder stage |

### Baked Models (in `models/` directory, gitignored)

| Model | Directory | Size | Purpose |
|-------|-----------|------|---------|
| Layout (Heron) | `docling-project--docling-layout-heron` | 164 MB | RT-DETR v2 layout detection |
| Tableformer | `docling-project--docling-models` | 342 MB | Table structure recognition |
| Figure Classifier | `docling-project--DocumentFigureClassifier-v2.0` | 32 MB | `do_picture_classification=true` |
| RapidOCR | `RapidOcr` | 59 MB | OCR (v4 + v5 models, v5 default) |
| EasyOCR | `EasyOcr` | 109 MB | Fallback OCR engine |
| CodeFormulaV2 | `docling-project--CodeFormulaV2` | 611 MB | `do_code_enrichment=true`, `do_formula_enrichment=true` |

### VLM Models (mounted at runtime)

Mount to `/opt/app-root/src/.cache/docling/models/vlm/`:

| Model | Directory | Size | Purpose |
|-------|-----------|------|---------|
| Granite Vision 3.3 2B | `ibm-granite--granite-vision-3.3-2b` | 5.6 GB | Default picture description VLM |
| Granite Docling 258M | `ibm-granite--granite-docling-258M` | 506 MB | Single-pass VLM pipeline |
| Qwen2.5-VL 3B | `Qwen--Qwen2.5-VL-3B-Instruct` | 7.1 GB | Alternative VLM |

### Configuration

- `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` (docling defaults)
- `DOCLING_SERVE_ENG_LOC_NUM_WORKERS=2` (docling default)
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (no network model downloads)
- PP-OCRv5 server models as default OCR (patched at build time)
- No pipeline batching ENV overrides
- No OCR scale override (stays 3 = 216 DPI)

Users can override any of these via `-e` flags at runtime.

### Key Environment Variables

- `TORCH_COMPILE_DISABLE=1` / `TORCHINDUCTOR_DISABLE=1` — prevents GPU crashes
- `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` — docling defaults
- `DOCLING_SERVE_ARTIFACTS_PATH` / `DOCLING_ARTIFACTS_PATH` — model cache paths
- `DOCLING_PICTURE_DESCRIPTION_VLM_MODEL_ID` — VLM model path (default: granite-vision-3.3-2b via mount)
- `DOCLING_PICTURE_DESCRIPTION_VLM_DEVICE=cuda` — VLM runs on GPU
- `DOCLING_SERVE_ENG_LOC_NUM_WORKERS=2` — docling default (2 workers)
- `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` — prevents model updates

## File Descriptions

- `Dockerfile` — Multi-stage GPU-optimized production build (the core of this project)
- `Dockerfile.debug` — Extended build with full pipeline profiling and VLM model
- `test_docling.py` — Diagnostics (10-check GPU report) and HTTP benchmark (8 task types: 4 standard + 4 VLM)
- `benchmark.sh` — Shell script comparing official vs custom image with automated container lifecycle
- `benchmark_rapidocr.py` — PP-OCRv4 vs v5 benchmark with GPU/CPU/batch comparisons
- `benchmark_models.py` — Standalone model benchmarks (RapidOCR, layout, table, VLM) bypassing docling
- `benchmark_onnx_sessions.py` — ONNX Runtime session options benchmark for RapidOCR tuning
- `vlm_bench_final.py` — Transformers BF16 vs vLLM BF16 comparison for granite-docling-258M
- `docling_debug_profiler.py` — Profiling monkey-patches for pipeline stages
- `debug_entrypoint.sh` — Enhanced entrypoint with startup diagnostics
- `INVESTIGATION.md` — Full pipeline investigation report with profiling data and optimization recommendations
- `h20_diag.sh` — Self-contained diagnostic for H100/H200 cloud pods
- `patches/ocr_page_batch_patch.py` — Build-time patch: batch OCR cls+rec across crops (Phase 3, Step 8)
- `patches/onnx_session_opt_patch.py` — Build-time patch: ONNX session memory/execution opts (Phase 3, Step 9)
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
