# docling-serve GPU Docker Image (CUDA 12.8)

GPU-optimized Docker image for [docling-serve](https://github.com/docling-project/docling-serve) that fixes critical GPU acceleration bugs in the official `docling-serve-cu128` image.

## Problem

The official `docling-serve-cu128` Docker image has several bugs that cause GPU underperformance:

| Issue | Root Cause | Fix Applied |
|-------|-----------|-------------|
| ONNX models run on CPU | Ships `onnxruntime` (CPU-only) instead of `onnxruntime-gpu` | Explicitly swap for `onnxruntime-gpu` |
| flash-attn fails to compile | Runtime base image lacks nvcc/CUDA headers | Use `nvidia/cuda:12.8.0-devel-ubuntu22.04` base |
| CUDA libraries not found | pip NVIDIA packages not on `LD_LIBRARY_PATH` | ldconfig wiring for all nvidia pip package lib dirs |
| cuDNN not available | devel image doesn't include cuDNN | PyTorch cu128 wheels bring `nvidia-cudnn-cu12`; wired via ldconfig |
| RapidOCR runs on CPU despite GPU | docling sets `Det.use_cuda` but RapidOCR reads `EngineConfig.onnxruntime.use_cuda` (defaults false) | Patch `rapid_ocr_model.py` to set `EngineConfig.onnxruntime.use_cuda` |

References:
- [docling#2528](https://github.com/docling-project/docling/issues/2528) - ONNX CPU-only bug
- [docling-serve#434](https://github.com/docling-project/docling-serve/issues/434) - GPU image ships CPU onnxruntime

## Build

```bash
docker build -t docling-serve-cu128-custom .
```

## Run

```bash
docker run --gpus all -p 5001:5001 docling-serve-cu128-custom
```

## Verify GPU Support

```bash
docker run --gpus all --rm docling-serve-cu128-custom python3 -c "
import onnxruntime as ort
print('ORT providers:', ort.get_available_providers())
assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'CUDA EP missing!'
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
print('ALL GOOD - GPU support confirmed')
"
```

## What's Included

- **Base**: `nvidia/cuda:12.8.0-devel-ubuntu22.04`
- **docling-serve**: v1.12.0
- **Python**: 3.12 (deadsnakes PPA)
- **GPU acceleration**: ONNX Runtime GPU, PyTorch cu128, flash-attn (compiled with nvcc)
- **Pre-downloaded models**: layout, tableformer, picture_classifier, rapidocr, easyocr
- **VLM picture description**: `ibm-granite/granite-vision-3.0-2b` (runs on CUDA)
- **OCR**: Tesseract + EasyOCR + RapidOCR (GPU-accelerated via ONNX)
- **Non-root**: Runs as UID 1001

## Configuration

Environment variables can be overridden at runtime:

```bash
docker run --gpus all -p 5001:5001 \
  -e OMP_NUM_THREADS=8 \
  -e DOCLING_SERVE_ENG_LOC_NUM_WORKERS=4 \
  docling-serve-cu128-custom
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OMP_NUM_THREADS` | 4 | OpenMP thread count |
| `DOCLING_SERVE_ENG_LOC_NUM_WORKERS` | 2 | Number of processing workers |
| `DOCLING_SERVE_MAX_SYNC_WAIT` | 1200 | Max sync wait time (seconds) |
| `DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT` | 1200 | Max document processing timeout |
| `DOCLING_PICTURE_DESCRIPTION_VLM_DEVICE` | cuda | Device for VLM model |
