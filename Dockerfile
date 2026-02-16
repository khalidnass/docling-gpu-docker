# syntax=docker/dockerfile:1
#
# GPU-optimized multi-stage Dockerfile for docling-serve with CUDA 12.8
#
# Fixes critical bugs in the official docling-serve-cu128 image:
#   1. ONNX Runtime runs on CPU (ships onnxruntime instead of onnxruntime-gpu)
#   2. flash-attn cannot compile (runtime image lacks nvcc/CUDA headers)
#   3. CUDA pip libraries not on LD_LIBRARY_PATH
#   4. cuDNN not wired from pip-installed nvidia-cudnn-cu12
#   5. RapidOCR ignores GPU (docling sets Det.use_cuda but ProviderConfig reads
#      EngineConfig.onnxruntime.use_cuda which defaults to false)
#
# Multi-stage build: compiles in devel image, runs in runtime image (~3-5GB smaller)
#
# flash-attn note: FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE installs prebuilt wheels
# with kernels for compute capability 8.0+ (H100/H200). Local GPUs with cap < 8
# won't use flash-attn, but it's required for production H100/H200 deployment.
#
# Build:
#   DOCKER_BUILDKIT=1 docker build -t docling-serve-cu128-custom .
#
# Run:
#   docker run --gpus all -p 5001:5001 docling-serve-cu128-custom

# ============================================================
# Stage 1: builder (devel image — has nvcc, CUDA headers)
# ============================================================
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        ninja-build g++ pkg-config \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------- uv package manager ----------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# ---------- Directory layout ----------
ENV APP_ROOT=/opt/app-root \
    HOME=/opt/app-root/src \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/app-root \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_CACHE_DIR=/opt/uv/cache

RUN mkdir -p \
        /opt/app-root/src/.local/bin \
        /opt/app-root/src/.cache/docling/models \
        /opt/app-root/src/.cache/huggingface/hub \
        /opt/app-root/src/.cache/torch \
        /opt/uv/python \
        /opt/uv/cache

ENV PATH=/opt/app-root/src/.local/bin:/opt/app-root/src/bin:/opt/app-root/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

WORKDIR /opt/app-root/src

# ---------- Create venv ----------
RUN uv venv --python /usr/bin/python3.12 --clear /opt/app-root

# ---------- Clone docling-serve ----------
ARG DOCLING_SERVE_REF=v1.12.0
RUN git clone --depth 1 --branch ${DOCLING_SERVE_REF} \
        https://github.com/docling-project/docling-serve.git /opt/docling-serve

# ---------- Install dependencies (two-pass for flash-attn) ----------
# Pass 1: everything except flash-attn (no CUDA compilation needed)
RUN --mount=type=cache,target=/opt/uv/cache \
    cd /opt/docling-serve && \
    umask 002 && \
    uv sync --frozen --no-dev --no-editable --all-extras \
        --no-group pypi --group cu128 \
        --no-extra flash-attn

# Pass 2: flash-attn — use prebuilt wheel (kernels for compute cap 8.0+, i.e. H100/H200)
# FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE avoids compiling from source
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
RUN --mount=type=cache,target=/opt/uv/cache \
    cd /opt/docling-serve && \
    umask 002 && \
    uv sync --frozen --no-dev --no-editable --all-extras \
        --no-group pypi --group cu128 \
        --no-build-isolation-package=flash-attn

# ---------- Fix ONNX Runtime: swap CPU for GPU (critical) ----------
RUN --mount=type=cache,target=/opt/uv/cache \
    uv pip uninstall onnxruntime && \
    uv pip install --no-cache-dir onnxruntime-gpu

# ---------- Build-time verification ----------
RUN /opt/app-root/bin/python3 -c "\
import onnxruntime as ort; \
providers = ort.get_available_providers(); \
print('ORT providers:', providers); \
assert 'CUDAExecutionProvider' in providers, \
    f'CUDAExecutionProvider missing! Got: {providers}'; \
print('ONNX Runtime GPU verification: PASSED'); \
"

# ---------- Cleanup & permission prep before COPY to runtime ----------
# Set group=user perms here so runtime COPY --chown doesn't need a separate chmod layer
RUN rm -rf /opt/docling-serve /opt/uv/cache \
    && find /opt/app-root -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && chmod -R g=u /opt/app-root

# ============================================================
# Stage 2: runtime (runtime image — much smaller, no nvcc/headers)
# ============================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Only runtime system deps — NO ninja, g++, git, python3.12-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
        tesseract-ocr tesseract-ocr-eng libleptonica-dev \
        libgl1 libglib2.0-0t64 libsm6 libxext6 libxrender1 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------- Copy venv + uv from builder ----------
# Use --chown to set ownership during COPY (avoids a huge duplicate layer from chown -R later)
COPY --chown=1001:0 --from=builder /opt/app-root /opt/app-root
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# ---------- Wire NVIDIA pip libraries into the dynamic linker ----------
# pip-installed nvidia packages (cudnn, nccl, etc.) bring their own .so libs
# inside site-packages — wire them via ldconfig so the runtime finds them
RUN find /opt/app-root/lib/python3.12/site-packages/nvidia \
        -maxdepth 2 -type d -name lib -print \
        > /etc/ld.so.conf.d/pip-nvidia.conf && \
    ldconfig

# ---------- Environment variables ----------
ENV APP_ROOT=/opt/app-root \
    HOME=/opt/app-root/src \
    PATH=/opt/app-root/src/.local/bin:/opt/app-root/src/bin:/opt/app-root/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64 \
    # CUDA / GPU
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # Docling artifacts
    DOCLING_SERVE_ARTIFACTS_PATH=/opt/app-root/src/.cache/docling/models \
    DOCLING_ARTIFACTS_PATH=/opt/app-root/src/.cache/docling/models \
    # Caches
    HF_HOME=/opt/app-root/src/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/app-root/src/.cache/huggingface \
    HF_HUB_CACHE=/opt/app-root/src/.cache/huggingface/hub \
    TORCH_HOME=/opt/app-root/src/.cache/torch \
    # Performance
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    # Disable torch.compile inductor (fails on GPUs with fewer SMs like A5000/T4)
    TORCHINDUCTOR_DISABLE=1 \
    # VLM picture description
    DOCLING_PICTURE_DESCRIPTION_MODEL_TYPE=vlm \
    DOCLING_PICTURE_DESCRIPTION_VLM_MODEL_ID=ibm-granite/granite-vision-3.3-2b \
    DOCLING_PICTURE_DESCRIPTION_VLM_DEVICE=cuda \
    DOCLING_PICTURE_DESCRIPTION_BACKEND=vlm \
    # OCR
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/ \
    # Docling-serve runtime
    DOCLING_SERVE_ENGINE=DoclingParseV2DocumentBackend \
    DOCLING_SERVE_MAX_SYNC_WAIT=1200 \
    DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT=1200 \
    DOCLING_SERVE_ENG_LOC_NUM_WORKERS=2

WORKDIR /opt/app-root/src

# ---------- Pre-download models for offline operation ----------
RUN HF_HUB_DOWNLOAD_TIMEOUT=90 HF_HUB_ETAG_TIMEOUT=90 \
    /opt/app-root/bin/docling-tools models download \
        -o "${DOCLING_SERVE_ARTIFACTS_PATH}" \
        layout tableformer picture_classifier rapidocr easyocr

# ---------- Fix DocumentFigureClassifier v2.0 path ----------
# docling v1.12+ looks for "DocumentFigureClassifier-v2.0" but the downloaded
# model is named "DocumentFigureClassifier". Create symlink so both names work.
RUN ln -sf "${DOCLING_SERVE_ARTIFACTS_PATH}/docling-project--DocumentFigureClassifier" \
           "${DOCLING_SERVE_ARTIFACTS_PATH}/docling-project--DocumentFigureClassifier-v2.0"

# ---------- container-entrypoint shim ----------
# Also creates the classifier symlink at runtime for volume-mounted models
RUN printf '%s\n' \
        '#!/usr/bin/env bash' \
        'set -e' \
        '' \
        '# Ensure cache dirs are writable for UID 1001' \
        '(chown -R 1001:0 /opt/app-root/src/.cache 2>/dev/null || true)' \
        '(chmod -R g+rwX /opt/app-root/src/.cache 2>/dev/null || true)' \
        '' \
        '# Fix DocumentFigureClassifier path for volume-mounted models' \
        'MODELS="${DOCLING_SERVE_ARTIFACTS_PATH:-/opt/app-root/src/.cache/docling/models}"' \
        'if [ -d "$MODELS/docling-project--DocumentFigureClassifier" ] && [ ! -e "$MODELS/docling-project--DocumentFigureClassifier-v2.0" ]; then' \
        '    ln -sf "$MODELS/docling-project--DocumentFigureClassifier" "$MODELS/docling-project--DocumentFigureClassifier-v2.0" 2>/dev/null || true' \
        'fi' \
        '' \
        'exec "$@"' \
    > /usr/local/bin/container-entrypoint && \
    chmod +x /usr/local/bin/container-entrypoint

# ---------- Fix security vulnerabilities (GHSA-cfh3-3jmp-rvhc, GHSA-r6ph-v2qm-q3c2) ----------
RUN uv pip install --no-cache-dir --python /opt/app-root/bin/python3 \
        "pillow>=12.1.1" \
        "cryptography>=46.0.5"

# ---------- Fix RapidOCR CUDA: patch docling to set EngineConfig.onnxruntime.use_cuda ----------
# Bug: docling sets Det.use_cuda=True but RapidOCR's ProviderConfig reads
# EngineConfig.onnxruntime.use_cuda (defaults to false), so OCR runs on CPU.
# This sed adds the missing onnxruntime engine config entries to rapid_ocr_model.py.
RUN RAPID_OCR_MODEL=/opt/app-root/lib/python3.12/site-packages/docling/models/stages/ocr/rapid_ocr_model.py && \
    sed -i 's|"EngineConfig.paddle.use_cuda": use_cuda,|"EngineConfig.onnxruntime.use_cuda": use_cuda,\n                "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": gpu_id,\n                "EngineConfig.paddle.use_cuda": use_cuda,|' \
        "$RAPID_OCR_MODEL" && \
    grep -q 'EngineConfig.onnxruntime.use_cuda' "$RAPID_OCR_MODEL" || \
        { echo "FATAL: RapidOCR CUDA patch failed"; exit 1; }

# ---------- Non-root user ----------
# /opt/app-root already owned by 1001:0 (COPY --chown) with g=u perms (set in builder)
# Create passwd entry for UID 1001 — PyTorch's getpass.getuser() needs it
RUN echo "docling:x:1001:0:docling:/opt/app-root/src:/bin/bash" >> /etc/passwd && \
    chown -R 1001:0 /tmp && \
    chmod -R g=u /tmp

USER 1001

EXPOSE 5001 8080

ENTRYPOINT ["container-entrypoint"]
CMD ["docling-serve", "run"]
