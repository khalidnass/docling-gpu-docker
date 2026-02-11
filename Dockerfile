# syntax=docker/dockerfile:1
#
# GPU-optimized Dockerfile for docling-serve with CUDA 12.8
#
# Fixes critical bugs in the official docling-serve-cu128 image:
#   1. ONNX Runtime runs on CPU (ships onnxruntime instead of onnxruntime-gpu)
#   2. flash-attn cannot compile (runtime image lacks nvcc/CUDA headers)
#   3. CUDA pip libraries not on LD_LIBRARY_PATH
#   4. cuDNN not wired from pip-installed nvidia-cudnn-cu12
#
# Build:
#   docker build -t docling-serve-cu128-custom .
#
# Run:
#   docker run --gpus all -p 5001:5001 docling-serve-cu128-custom

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# ---------- System packages ----------
# Ubuntu 24.04 ships Python 3.12 natively (no PPA needed)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        # Build tools
        ninja-build g++ pkg-config \
        # OCR
        tesseract-ocr tesseract-ocr-eng libleptonica-dev \
        # OpenCV runtime deps
        libgl1 libglib2.0-0t64 libsm6 libxext6 libxrender1 \
        # Utilities
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------- uv package manager ----------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# ---------- Directory layout ----------
ENV APP_ROOT=/opt/app-root
ENV HOME=/opt/app-root/src

RUN mkdir -p \
        /opt/app-root/src/.local/bin \
        /opt/app-root/src/.cache/docling/models \
        /opt/app-root/src/.cache/huggingface/hub \
        /opt/app-root/src/.cache/torch \
        /opt/uv/python \
        /opt/uv/cache

WORKDIR /opt/app-root/src

# ---------- uv settings ----------
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/opt/app-root
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_CACHE_DIR=/opt/uv/cache

ENV PATH=/opt/app-root/src/.local/bin:/opt/app-root/src/bin:/opt/app-root/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# ---------- Create venv ----------
RUN uv venv --python /usr/bin/python3.12 --clear /opt/app-root

# ---------- Clone docling-serve ----------
ARG DOCLING_SERVE_REF=v1.12.0
RUN git clone --depth 1 --branch ${DOCLING_SERVE_REF} \
        https://github.com/docling-project/docling-serve.git /opt/docling-serve

# ---------- Install dependencies (two-pass for flash-attn) ----------
# Pass 1: everything except flash-attn (no CUDA compilation needed)
RUN cd /opt/docling-serve && \
    umask 002 && \
    uv sync --frozen --no-dev --all-extras \
        --no-group pypi --group cu128 \
        --no-extra flash-attn

# Pass 2: flash-attn with CUDA compilation
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
RUN cd /opt/docling-serve && \
    umask 002 && \
    uv sync --frozen --no-dev --all-extras \
        --no-group pypi --group cu128 \
        --no-build-isolation-package=flash-attn

# ---------- Fix ONNX Runtime: swap CPU for GPU (critical) ----------
RUN uv pip uninstall onnxruntime && \
    uv pip install --no-cache-dir onnxruntime-gpu

# ---------- Wire NVIDIA pip libraries into the dynamic linker ----------
RUN find /opt/app-root/lib/python3.12/site-packages/nvidia \
        -maxdepth 2 -type d -name lib -print \
        > /etc/ld.so.conf.d/pip-nvidia.conf && \
    ldconfig

# Backup: also set LD_LIBRARY_PATH explicitly
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64

# ---------- Build-time verification ----------
RUN /opt/app-root/bin/python3 -c "\
import onnxruntime as ort; \
providers = ort.get_available_providers(); \
print('ORT providers:', providers); \
assert 'CUDAExecutionProvider' in providers, \
    f'CUDAExecutionProvider missing! Got: {providers}'; \
print('ONNX Runtime GPU verification: PASSED'); \
"

# ---------- Environment variables ----------
# CUDA / GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Docling artifacts
ENV DOCLING_SERVE_ARTIFACTS_PATH=/opt/app-root/src/.cache/docling/models
ENV DOCLING_ARTIFACTS_PATH=/opt/app-root/src/.cache/docling/models

# Caches
ENV HF_HOME=/opt/app-root/src/.cache/huggingface
ENV TRANSFORMERS_CACHE=/opt/app-root/src/.cache/huggingface
ENV HF_HUB_CACHE=/opt/app-root/src/.cache/huggingface/hub
ENV TORCH_HOME=/opt/app-root/src/.cache/torch

# Performance
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# VLM picture description
ENV DOCLING_PICTURE_DESCRIPTION_MODEL_TYPE=vlm
ENV DOCLING_PICTURE_DESCRIPTION_VLM_MODEL_ID=ibm-granite/granite-vision-3.0-2b
ENV DOCLING_PICTURE_DESCRIPTION_VLM_DEVICE=cuda
ENV DOCLING_PICTURE_DESCRIPTION_BACKEND=vlm

# OCR
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

# Docling-serve runtime
ENV DOCLING_SERVE_ENGINE=DoclingParseV2DocumentBackend
ENV DOCLING_SERVE_MAX_SYNC_WAIT=1200
ENV DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT=1200
ENV DOCLING_SERVE_ENG_LOC_NUM_WORKERS=2

# ---------- Pre-download models for offline operation ----------
RUN HF_HUB_DOWNLOAD_TIMEOUT=90 HF_HUB_ETAG_TIMEOUT=90 \
    /opt/app-root/bin/docling-tools models download \
        -o "${DOCLING_SERVE_ARTIFACTS_PATH}" \
        layout tableformer picture_classifier rapidocr easyocr

# ---------- container-entrypoint shim ----------
RUN printf '%s\n' \
        '#!/usr/bin/env bash' \
        'set -e' \
        '' \
        '# Ensure cache dirs are writable for UID 1001' \
        '(chown -R 1001:0 /opt/app-root/src/.cache 2>/dev/null || true)' \
        '(chmod -R g+rwX /opt/app-root/src/.cache 2>/dev/null || true)' \
        '' \
        'exec "$@"' \
    > /usr/local/bin/container-entrypoint && \
    chmod +x /usr/local/bin/container-entrypoint

# ---------- Non-root user ----------
RUN chown -R 1001:0 /opt/app-root /tmp && \
    chmod -R g=u /opt/app-root /tmp

USER 1001

EXPOSE 5001 8080

ENTRYPOINT ["container-entrypoint"]
CMD ["docling-serve", "run"]
