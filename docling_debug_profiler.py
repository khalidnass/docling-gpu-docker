#!/usr/bin/env python3
"""
Docling Debug Profiler — Monkey-patches for comprehensive pipeline timing.

Designed to be imported at startup in the debug Docker image. Patches docling's
existing TimeRecorder to log to stdout, and adds profiling for steps that don't
have TimeRecorder coverage yet.

Usage (in debug_entrypoint.sh):
    python3 -c "import docling_debug_profiler; docling_debug_profiler.install()" && exec docling-serve run ...

Or as a sitecustomize-style import:
    PYTHONSTARTUP=/path/to/docling_debug_profiler.py docling-serve run
"""

import functools
import logging
import time
from typing import Any

_log = logging.getLogger("docling.profiler")

_INSTALLED = False


def install():
    """Apply all profiling monkey-patches. Safe to call multiple times."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    _log.setLevel(logging.INFO)
    if not _log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        _log.addHandler(handler)

    _log.info("[PROFILE] Installing docling debug profiler patches...")

    _patch_time_recorder()
    _patch_rapid_ocr_model()
    _patch_layout_model()
    _patch_table_structure_model()
    _patch_base_pipeline_execute()
    _patch_hf_transformers_vlm()
    _patch_vllm_vlm()

    _log.info("[PROFILE] All profiler patches installed.")


# ---------------------------------------------------------------------------
# Patch 1: TimeRecorder.__exit__ — log every timing to stdout
# ---------------------------------------------------------------------------
def _patch_time_recorder():
    """Make TimeRecorder log elapsed time to stdout on every __exit__."""
    try:
        from docling.utils.profiling import TimeRecorder
        from docling.datamodel.settings import settings

        original_exit = TimeRecorder.__exit__

        def patched_exit(self, *args):
            original_exit(self, *args)
            # Log regardless of settings — the debug image always wants output
            if hasattr(self, "start") and hasattr(self, "key"):
                elapsed = time.monotonic() - self.start
                _log.info(f"[PROFILE] {self.key}: {elapsed:.4f}s")

        TimeRecorder.__exit__ = patched_exit

        # Also force profiling to be enabled
        settings.debug.profile_pipeline_timings = True

        _log.info("[PROFILE]   Patched TimeRecorder.__exit__ (logging enabled)")
    except Exception as e:
        _log.warning(f"[PROFILE]   Failed to patch TimeRecorder: {e}")


# ---------------------------------------------------------------------------
# Patch 2: RapidOcrModel — per-crop timing + dimensions + config logging
# ---------------------------------------------------------------------------
def _patch_rapid_ocr_model():
    """Add per-crop OCR timing and dimension logging."""
    try:
        from docling.models.rapid_ocr_model import RapidOcrModel

        original_init = RapidOcrModel.__init__
        original_call = RapidOcrModel.__call__

        @functools.wraps(original_init)
        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if self.enabled:
                _log.info(
                    f"[PROFILE] RapidOCR init: backend={self.options.backend}, "
                    f"scale={self.scale}"
                )

        @functools.wraps(original_call)
        def patched_call(self, conv_res, page_batch):
            import numpy
            from docling.utils.profiling import TimeRecorder

            if not self.enabled:
                yield from page_batch
                return

            for page in page_batch:
                assert page._backend is not None
                if not page._backend.is_valid():
                    yield page
                else:
                    with TimeRecorder(conv_res, "ocr"):
                        ocr_rects = self.get_ocr_rects(page)
                        _log.info(
                            f"[PROFILE] OCR page {page.page_no}: "
                            f"{len(ocr_rects)} regions detected"
                        )

                        all_ocr_cells = []
                        for i, ocr_rect in enumerate(ocr_rects):
                            if ocr_rect.area() == 0:
                                continue
                            high_res_image = page._backend.get_page_image(
                                scale=self.scale, cropbox=ocr_rect
                            )
                            im = numpy.array(high_res_image)
                            _log.info(
                                f"[PROFILE] OCR crop #{i} page {page.page_no}: "
                                f"{im.shape[1]}x{im.shape[0]}px"
                            )
                            t0 = time.perf_counter()
                            result = self.reader(
                                im,
                                use_det=self.options.use_det,
                                use_cls=self.options.use_cls,
                                use_rec=self.options.use_rec,
                            )
                            crop_elapsed = time.perf_counter() - t0
                            n_lines = 0
                            if result is not None and result.boxes is not None:
                                n_lines = len(result.boxes)
                            _log.info(
                                f"[PROFILE] OCR crop #{i} page {page.page_no}: "
                                f"{crop_elapsed:.4f}s, {n_lines} lines"
                            )

                            if result is None or result.boxes is None:
                                _log.warning("RapidOCR returned empty result!")
                                continue
                            result = list(
                                zip(
                                    result.boxes.tolist(),
                                    result.txts,
                                    result.scores,
                                )
                            )

                            del high_res_image
                            del im

                            if result is not None:
                                from docling_core.types.doc import (
                                    BoundingBox,
                                    CoordOrigin,
                                )
                                from docling_core.types.doc.page import (
                                    BoundingRectangle,
                                    TextCell,
                                )

                                cells = [
                                    TextCell(
                                        index=ix,
                                        text=line[1],
                                        orig=line[1],
                                        confidence=line[2],
                                        from_ocr=True,
                                        rect=BoundingRectangle.from_bounding_box(
                                            BoundingBox.from_tuple(
                                                coord=(
                                                    (line[0][0][0] / self.scale)
                                                    + ocr_rect.l,
                                                    (line[0][0][1] / self.scale)
                                                    + ocr_rect.t,
                                                    (line[0][2][0] / self.scale)
                                                    + ocr_rect.l,
                                                    (line[0][2][1] / self.scale)
                                                    + ocr_rect.t,
                                                ),
                                                origin=CoordOrigin.TOPLEFT,
                                            )
                                        ),
                                    )
                                    for ix, line in enumerate(result)
                                ]
                                all_ocr_cells.extend(cells)

                        self.post_process_cells(all_ocr_cells, page)

                    from docling.datamodel.settings import settings

                    if settings.debug.visualize_ocr:
                        self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                    yield page

        RapidOcrModel.__init__ = patched_init
        RapidOcrModel.__call__ = patched_call
        _log.info("[PROFILE]   Patched RapidOcrModel (per-crop timing)")
    except Exception as e:
        _log.warning(f"[PROFILE]   Failed to patch RapidOcrModel: {e}")


# ---------------------------------------------------------------------------
# Patch 3: LayoutModel — per-batch timing + page count logging
# ---------------------------------------------------------------------------
def _patch_layout_model():
    """Add per-batch timing and page count logging to layout model."""
    try:
        from docling.models.layout_model import LayoutModel

        original_call = LayoutModel.__call__

        @functools.wraps(original_call)
        def patched_call(self, conv_res, page_batch):
            page_list = list(page_batch)
            n_pages = len(page_list)
            _log.info(f"[PROFILE] Layout model: processing batch of {n_pages} pages")
            t0 = time.perf_counter()
            result = original_call(self, conv_res, iter(page_list))
            # Must exhaust the generator
            result_list = list(result)
            elapsed = time.perf_counter() - t0
            _log.info(
                f"[PROFILE] Layout model: batch of {n_pages} pages completed in {elapsed:.4f}s "
                f"({elapsed/max(n_pages,1):.4f}s/page)"
            )
            yield from result_list

        LayoutModel.__call__ = patched_call
        _log.info("[PROFILE]   Patched LayoutModel (batch timing)")
    except Exception as e:
        _log.warning(f"[PROFILE]   Failed to patch LayoutModel: {e}")


# ---------------------------------------------------------------------------
# Patch 4: TableStructureModel — per-table timing
# ---------------------------------------------------------------------------
def _patch_table_structure_model():
    """Add per-table timing logging."""
    try:
        from docling.models.table_structure_model import TableStructureModel

        original_call = TableStructureModel.__call__

        @functools.wraps(original_call)
        def patched_call(self, conv_res, page_batch):
            page_list = list(page_batch)
            _log.info(
                f"[PROFILE] TableStructure: processing {len(page_list)} pages"
            )
            t0 = time.perf_counter()
            result = original_call(self, conv_res, iter(page_list))
            result_list = list(result)
            elapsed = time.perf_counter() - t0
            _log.info(
                f"[PROFILE] TableStructure: {len(page_list)} pages in {elapsed:.4f}s"
            )
            yield from result_list

        TableStructureModel.__call__ = patched_call
        _log.info("[PROFILE]   Patched TableStructureModel (per-table timing)")
    except Exception as e:
        _log.warning(f"[PROFILE]   Failed to patch TableStructureModel: {e}")


# ---------------------------------------------------------------------------
# Patch 5: BasePipeline.execute — print timing summary after conversion
# ---------------------------------------------------------------------------
def _patch_base_pipeline_execute():
    """Print timing summary from conv_res.timings after execute()."""
    try:
        from docling.pipeline.base_pipeline import BasePipeline

        original_execute = BasePipeline.execute

        @functools.wraps(original_execute)
        def patched_execute(self, in_doc, raises_on_error):
            _log.info(
                f"[PROFILE] ===== PIPELINE START: {in_doc.file.name} "
                f"(pipeline={self.__class__.__name__}) ====="
            )
            t0 = time.perf_counter()
            conv_res = original_execute(self, in_doc, raises_on_error)
            total = time.perf_counter() - t0
            _log.info(f"[PROFILE] ===== PIPELINE TIMING SUMMARY =====")
            _log.info(f"[PROFILE]   pipeline: {self.__class__.__name__}")
            _log.info(f"[PROFILE]   document: {in_doc.file.name}")
            _log.info(f"[PROFILE]   total_wall: {total:.4f}s")
            _log.info(f"[PROFILE]   status: {conv_res.status}")
            if conv_res.timings:
                for key, item in conv_res.timings.items():
                    if item.times:
                        avg = sum(item.times) / len(item.times)
                        total_t = sum(item.times)
                        _log.info(
                            f"[PROFILE]   {key}: count={item.count}, "
                            f"total={total_t:.4f}s, avg={avg:.4f}s, "
                            f"scope={item.scope}"
                        )
            else:
                _log.info("[PROFILE]   (no timings recorded)")
            _log.info(f"[PROFILE] =====================================")
            return conv_res

        BasePipeline.execute = patched_execute
        _log.info("[PROFILE]   Patched BasePipeline.execute (timing summary)")
    except Exception as e:
        _log.warning(f"[PROFILE]   Failed to patch BasePipeline.execute: {e}")


# ---------------------------------------------------------------------------
# Patch 6: HuggingFace Transformers VLM — model load + inference timing
# ---------------------------------------------------------------------------
def _patch_hf_transformers_vlm():
    """Add model load time and per-inference token speed logging."""
    try:
        from docling.models.vlm_models_inline.hf_transformers_model import (
            HuggingFaceTransformersVlmModel,
        )

        original_init = HuggingFaceTransformersVlmModel.__init__

        @functools.wraps(original_init)
        def patched_init(self, *args, **kwargs):
            t0 = time.perf_counter()
            original_init(self, *args, **kwargs)
            elapsed = time.perf_counter() - t0
            _log.info(f"[PROFILE] HF Transformers VLM init: {elapsed:.2f}s")
            if hasattr(self, "vlm_options"):
                _log.info(
                    f"[PROFILE] HF Transformers VLM config: "
                    f"model={getattr(self.vlm_options, 'repo_id', 'unknown')}, "
                    f"scale={getattr(self.vlm_options, 'scale', 'unknown')}"
                )

        HuggingFaceTransformersVlmModel.__init__ = patched_init

        # Patch process_images if it exists
        if hasattr(HuggingFaceTransformersVlmModel, "process_images"):
            original_process = HuggingFaceTransformersVlmModel.process_images

            @functools.wraps(original_process)
            def patched_process(self, image_batch, prompt):
                images = list(image_batch) if not isinstance(image_batch, list) else image_batch
                _log.info(
                    f"[PROFILE] HF VLM inference: {len(images)} images"
                )
                t0 = time.perf_counter()
                results = list(original_process(self, images, prompt))
                elapsed = time.perf_counter() - t0
                total_tokens = sum(
                    getattr(r, "num_tokens", 0) for r in results
                )
                tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                _log.info(
                    f"[PROFILE] HF VLM inference: {elapsed:.4f}s, "
                    f"{total_tokens} tokens, {tok_per_sec:.1f} tok/s"
                )
                yield from results

            HuggingFaceTransformersVlmModel.process_images = patched_process

        _log.info("[PROFILE]   Patched HuggingFaceTransformersVlmModel")
    except Exception as e:
        _log.warning(f"[PROFILE]   Failed to patch HF Transformers VLM: {e}")


# ---------------------------------------------------------------------------
# Patch 7: vLLM VLM — init + inference timing
# ---------------------------------------------------------------------------
def _patch_vllm_vlm():
    """Add vLLM model load time and inference speed logging."""
    try:
        from docling.models.vlm_models_inline.vllm_model import VllmVlmModel

        original_init = VllmVlmModel.__init__

        @functools.wraps(original_init)
        def patched_init(self, *args, **kwargs):
            t0 = time.perf_counter()
            original_init(self, *args, **kwargs)
            elapsed = time.perf_counter() - t0
            _log.info(f"[PROFILE] vLLM VLM init: {elapsed:.2f}s")
            if hasattr(self, "vlm_options"):
                _log.info(
                    f"[PROFILE] vLLM VLM config: "
                    f"model={getattr(self.vlm_options, 'repo_id', 'unknown')}"
                )

        VllmVlmModel.__init__ = patched_init

        if hasattr(VllmVlmModel, "process_images"):
            original_process = VllmVlmModel.process_images

            @functools.wraps(original_process)
            def patched_process(self, image_batch, prompt):
                images = list(image_batch) if not isinstance(image_batch, list) else image_batch
                _log.info(f"[PROFILE] vLLM inference: {len(images)} images")
                t0 = time.perf_counter()
                results = list(original_process(self, images, prompt))
                elapsed = time.perf_counter() - t0
                total_tokens = sum(
                    getattr(r, "num_tokens", 0) for r in results
                )
                tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                _log.info(
                    f"[PROFILE] vLLM inference: {elapsed:.4f}s, "
                    f"{total_tokens} tokens, {tok_per_sec:.1f} tok/s"
                )
                yield from results

            VllmVlmModel.process_images = patched_process

        _log.info("[PROFILE]   Patched VllmVlmModel")
    except Exception as e:
        _log.warning(f"[PROFILE]   Failed to patch vLLM VLM: {e}")


# ---------------------------------------------------------------------------
# Startup diagnostics — print GPU/model summary
# ---------------------------------------------------------------------------
def print_startup_diagnostics():
    """Print GPU, ONNX, and model information at startup."""
    import os

    _log.info("[DIAG] ===== STARTUP DIAGNOSTICS =====")

    # GPU info
    try:
        import torch

        if torch.cuda.is_available():
            _log.info(f"[DIAG] GPU: {torch.cuda.get_device_name(0)}")
            _log.info(
                f"[DIAG] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
            _log.info(f"[DIAG] Compute cap: {torch.cuda.get_device_capability(0)}")
            _log.info(f"[DIAG] PyTorch CUDA: {torch.version.cuda}")
        else:
            _log.info("[DIAG] GPU: NOT AVAILABLE")
    except ImportError:
        _log.info("[DIAG] torch not installed")

    # ONNX Runtime
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        _log.info(f"[DIAG] ONNX RT: {ort.__version__}, providers={providers}")
        if "CUDAExecutionProvider" not in providers:
            _log.warning("[DIAG] WARNING: CUDAExecutionProvider NOT available!")
    except ImportError:
        _log.info("[DIAG] onnxruntime not installed")

    # Environment
    env_keys = [
        "DOCLING_SERVE_ARTIFACTS_PATH",
        "DOCLING_SERVE_ENG_LOC_NUM_WORKERS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TORCH_COMPILE_DISABLE",
        "DOCLING_PICTURE_DESCRIPTION_VLM_MODEL_ID",
    ]
    for k in env_keys:
        v = os.environ.get(k, "not set")
        _log.info(f"[DIAG] {k}={v}")

    # Models
    artifacts = os.environ.get(
        "DOCLING_SERVE_ARTIFACTS_PATH",
        "/opt/app-root/src/.cache/docling/models",
    )
    from pathlib import Path

    models_path = Path(artifacts)
    if models_path.exists():
        for item in sorted(models_path.iterdir()):
            if item.is_dir():
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                _log.info(f"[DIAG] Model: {item.name} ({size/1e6:.1f} MB)")
    else:
        _log.info(f"[DIAG] Models path not found: {artifacts}")

    _log.info("[DIAG] ===================================")


# Allow running as: python3 -c "import docling_debug_profiler; docling_debug_profiler.install()"
if __name__ == "__main__":
    install()
    print_startup_diagnostics()
