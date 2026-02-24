#!/usr/bin/env python3
"""
Standalone Model Benchmarks — Tests each docling model independently.

Bypasses docling's pipeline to isolate model performance from framework overhead.
Run inside the Docker container or locally with the same dependencies.

Usage:
    python3 benchmark_models.py                    # Run all benchmarks
    python3 benchmark_models.py --model rapidocr   # Run specific model
    python3 benchmark_models.py --model layout      # Layout model only
    python3 benchmark_models.py --model vlm         # VLM models only
    python3 benchmark_models.py --list              # List available benchmarks

Models tested:
    rapidocr     — PP-OCRv4 via RapidOCR (det+cls+rec)
    layout       — RT-DETR v2 layout detection
    tableformer  — TableFormer table structure
    vlm-granite  — granite-docling-258M via transformers
    vlm-vllm     — granite-docling-258M via vLLM
    vlm-vision   — granite-vision-3.3-2b via transformers
    vlm-smol     — SmolDocling-256M-preview via transformers
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
_log = logging.getLogger("benchmark")

ARTIFACTS_PATH = Path(
    os.environ.get(
        "DOCLING_SERVE_ARTIFACTS_PATH",
        os.environ.get(
            "DOCLING_ARTIFACTS_PATH",
            os.path.expanduser("~/.cache/docling/models"),
        ),
    )
)

HF_CACHE = Path(
    os.environ.get(
        "HF_HOME",
        os.path.expanduser("~/.cache/huggingface"),
    )
)

RESULTS: dict = {}


def _make_test_image(width: int, height: int) -> np.ndarray:
    """Create a synthetic test image with text-like patterns."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    # Add some dark horizontal bars to simulate text lines
    for y in range(30, height - 30, 40):
        h = np.random.randint(8, 16)
        x_start = np.random.randint(20, 60)
        x_end = width - np.random.randint(20, 60)
        img[y : y + h, x_start:x_end] = np.random.randint(10, 60, (h, x_end - x_start, 3))
    return img


def _make_test_pdf_page(width: int = 612, height: int = 792) -> np.ndarray:
    """Create a full-page test image at typical PDF dimensions."""
    return _make_test_image(width, height)


def _timer(label: str):
    """Simple context manager timer."""

    class Timer:
        def __init__(self):
            self.elapsed = 0.0

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
            _log.info(f"[BENCH] {label}: {self.elapsed:.4f}s")

    return Timer()


# =========================================================================
# Benchmark 1: RapidOCR Standalone
# =========================================================================
def bench_rapidocr():
    """Test RapidOCR with different configs and image sizes."""
    _log.info("=" * 60)
    _log.info("BENCHMARK: RapidOCR Standalone")
    _log.info("=" * 60)

    results = {}

    try:
        from rapidocr import RapidOCR, EngineType
    except ImportError:
        _log.error("rapidocr not installed")
        return results

    # Check ONNX Runtime providers
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        has_cuda = "CUDAExecutionProvider" in providers
        _log.info(f"ONNX RT providers: {providers}")
    except ImportError:
        has_cuda = False
        _log.warning("onnxruntime not installed")

    # Test image sizes
    test_images = {
        "small_crop_100x30": _make_test_image(100, 30),
        "medium_region_600x800": _make_test_image(600, 800),
        "full_page_2400x3200": _make_test_image(2400, 3200),
        "hi_res_page_1800x2400": _make_test_image(1800, 2400),
    }

    # Model paths
    det_path = ARTIFACTS_PATH / "RapidOcr" / "onnx" / "PP-OCRv4" / "det" / "ch_PP-OCRv4_det_infer.onnx"
    cls_path = ARTIFACTS_PATH / "RapidOcr" / "onnx" / "PP-OCRv4" / "cls" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"
    rec_path = ARTIFACTS_PATH / "RapidOcr" / "onnx" / "PP-OCRv4" / "rec" / "ch_PP-OCRv4_rec_infer.onnx"
    rec_keys = ARTIFACTS_PATH / "RapidOcr" / "paddle" / "PP-OCRv4" / "rec" / "ch_PP-OCRv4_rec_infer" / "ppocr_keys_v1.txt"

    if not det_path.exists():
        _log.warning(f"RapidOCR models not found at {ARTIFACTS_PATH / 'RapidOcr'}")
        # Try without explicit paths
        det_path = cls_path = rec_path = rec_keys = None

    # Configs to test
    configs = [
        {
            "name": "cpu_default",
            "params": {
                "Det.use_cuda": False,
                "Cls.use_cuda": False,
                "Rec.use_cuda": False,
            },
        },
    ]

    if has_cuda:
        configs.extend(
            [
                {
                    "name": "cuda_default",
                    "params": {
                        "Det.use_cuda": True,
                        "Cls.use_cuda": True,
                        "Rec.use_cuda": True,
                        "EngineConfig.onnxruntime.use_cuda": True,
                        "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 0,
                    },
                },
            ]
        )

    for config in configs:
        _log.info(f"\n--- Config: {config['name']} ---")
        config_results = {}

        params = {
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Cls.engine_type": EngineType.ONNXRUNTIME,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
        }
        if det_path:
            params.update(
                {
                    "Det.model_path": str(det_path),
                    "Cls.model_path": str(cls_path),
                    "Rec.model_path": str(rec_path),
                    "Rec.rec_keys_path": str(rec_keys),
                }
            )
        params.update(config["params"])

        try:
            with _timer(f"RapidOCR init ({config['name']})") as t:
                reader = RapidOCR(params=params)
            config_results["init_time"] = t.elapsed

            for img_name, img in test_images.items():
                # Warmup
                _ = reader(img)

                # Timed runs
                times = []
                n_runs = 3
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    result = reader(img)
                    times.append(time.perf_counter() - t0)

                n_lines = 0
                if result is not None and result.boxes is not None:
                    n_lines = len(result.boxes)

                avg = sum(times) / len(times)
                config_results[img_name] = {
                    "avg_s": round(avg, 4),
                    "min_s": round(min(times), 4),
                    "max_s": round(max(times), 4),
                    "lines_found": n_lines,
                    "img_shape": list(img.shape),
                }
                _log.info(
                    f"  {img_name}: avg={avg:.4f}s, min={min(times):.4f}s, "
                    f"max={max(times):.4f}s, lines={n_lines}"
                )

            del reader
        except Exception as e:
            _log.error(f"  Config {config['name']} failed: {e}")
            config_results["error"] = str(e)

        results[config["name"]] = config_results

    RESULTS["rapidocr"] = results
    return results


# =========================================================================
# Benchmark 2: Layout Model (RT-DETR) Standalone
# =========================================================================
def bench_layout():
    """Test RT-DETR layout model with different batch sizes and DPIs."""
    _log.info("=" * 60)
    _log.info("BENCHMARK: Layout Model (RT-DETR) Standalone")
    _log.info("=" * 60)

    results = {}

    try:
        from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
    except ImportError:
        _log.error("docling_ibm_models not installed")
        return results

    model_path = ARTIFACTS_PATH / "docling-project--docling-layout-heron"
    if not model_path.exists():
        _log.warning(f"Layout model not found: {model_path}")
        return results

    # DPI scales: 72dpi base * scale
    dpi_configs = {
        "72dpi": 1.0,
        "144dpi": 2.0,
        "216dpi": 3.0,
    }

    # Test with CUDA and CPU
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
    except ImportError:
        pass

    for device in devices:
        _log.info(f"\n--- Device: {device} ---")
        device_results = {}

        try:
            with _timer(f"LayoutPredictor init ({device})") as t:
                predictor = LayoutPredictor(
                    artifact_path=str(model_path),
                    device=device,
                )
            device_results["init_time"] = t.elapsed

            for dpi_name, scale in dpi_configs.items():
                w = int(612 * scale)
                h = int(792 * scale)
                from PIL import Image

                test_img = Image.fromarray(_make_test_image(w, h))

                # Warmup
                _ = list(predictor.predict([test_img]))

                for batch_size in [1, 4, 8]:
                    batch = [test_img] * batch_size

                    times = []
                    n_runs = 3
                    for _ in range(n_runs):
                        t0 = time.perf_counter()
                        preds = list(predictor.predict(batch))
                        times.append(time.perf_counter() - t0)

                    avg = sum(times) / len(times)
                    key = f"{dpi_name}_batch{batch_size}"
                    device_results[key] = {
                        "avg_s": round(avg, 4),
                        "per_page_s": round(avg / batch_size, 4),
                        "img_size": f"{w}x{h}",
                    }
                    _log.info(
                        f"  {key}: avg={avg:.4f}s, per_page={avg/batch_size:.4f}s"
                    )

            del predictor
        except Exception as e:
            _log.error(f"  Device {device} failed: {e}")
            device_results["error"] = str(e)

        results[device] = device_results

    RESULTS["layout"] = results
    return results


# =========================================================================
# Benchmark 3: TableFormer Standalone
# =========================================================================
def bench_tableformer():
    """Test TableFormer table structure model."""
    _log.info("=" * 60)
    _log.info("BENCHMARK: TableFormer Standalone")
    _log.info("=" * 60)

    results = {}

    try:
        from docling_ibm_models.tableformer.data_management.tf_predictor import (
            TFPredictor,
        )
    except ImportError:
        _log.error("docling_ibm_models.tableformer not installed")
        return results

    model_path = ARTIFACTS_PATH / "docling-project--docling-models"
    if not model_path.exists():
        _log.warning(f"TableFormer model not found: {model_path}")
        return results

    for mode in ["fast", "accurate"]:
        _log.info(f"\n--- Mode: {mode} ---")
        try:
            with _timer(f"TableFormer init ({mode})") as t:
                config = {
                    "mode": mode,
                    "artifacts_path": str(model_path),
                }
                predictor = TFPredictor(config)
            results[f"{mode}_init_time"] = t.elapsed

            # Create a simple test table image
            table_img = _make_test_image(400, 300)

            page_input = {
                "width": 400.0,
                "height": 300.0,
                "image": table_img,
                "tokens": [],
            }
            table_bbox = [0.0, 0.0, 400.0, 300.0]

            # Warmup
            try:
                _ = predictor.multi_table_predict(page_input, [table_bbox])
            except Exception:
                pass  # May fail on synthetic image

            # Timed runs
            times = []
            n_runs = 3
            for _ in range(n_runs):
                t0 = time.perf_counter()
                try:
                    _ = predictor.multi_table_predict(page_input, [table_bbox])
                except Exception:
                    pass
                times.append(time.perf_counter() - t0)

            if times:
                avg = sum(times) / len(times)
                results[f"{mode}_avg_s"] = round(avg, 4)
                _log.info(f"  {mode}: avg={avg:.4f}s per table")

            del predictor
        except Exception as e:
            _log.error(f"  Mode {mode} failed: {e}")
            results[f"{mode}_error"] = str(e)

    RESULTS["tableformer"] = results
    return results


# =========================================================================
# Benchmark 4: Granite-Docling-258M via Transformers
# =========================================================================
def bench_vlm_granite_docling():
    """Test granite-docling-258M VLM with different precision modes."""
    _log.info("=" * 60)
    _log.info("BENCHMARK: granite-docling-258M via Transformers")
    _log.info("=" * 60)

    results = {}

    model_id = "ibm-granite/granite-docling-258M"

    # Check local model paths
    local_paths = [
        ARTIFACTS_PATH / "ibm-granite--granite-docling-258M",
        HF_CACHE / "hub" / "models--ibm-granite--granite-docling-258M",
        Path(os.path.expanduser("~/neo/docling-models/ibm-granite--granite-docling-258M")),
    ]
    local_path = None
    for p in local_paths:
        if p.exists():
            local_path = str(p)
            break

    if local_path:
        _log.info(f"Using local model: {local_path}")
        model_id = local_path

    try:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from PIL import Image
    except ImportError:
        _log.error("transformers or torch not installed")
        return results

    if not torch.cuda.is_available():
        _log.warning("CUDA not available, testing CPU only")

    test_img = Image.fromarray(_make_test_image(1200, 1600))

    dtype_configs = [
        ("fp32", torch.float32, "cpu"),
    ]
    if torch.cuda.is_available():
        dtype_configs.extend(
            [
                ("fp32_cuda", torch.float32, "cuda"),
                ("fp16_cuda", torch.float16, "cuda"),
                ("bf16_cuda", torch.bfloat16, "cuda"),
            ]
        )

    for dtype_name, dtype, device in dtype_configs:
        _log.info(f"\n--- Config: {dtype_name} ---")
        config_results = {}

        try:
            with _timer(f"Model load ({dtype_name})") as t:
                processor = AutoProcessor.from_pretrained(model_id)
                model = AutoModelForVision2Seq.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                ).to(device)
            config_results["load_time"] = t.elapsed

            prompt = "Convert this page to docling."
            inputs = processor(images=test_img, text=prompt, return_tensors="pt").to(
                device
            )

            # Warmup
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=50)

            # Cold inference (first real run)
            with _timer(f"Cold inference ({dtype_name})") as t:
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=512)
            config_results["cold_inference_s"] = t.elapsed
            n_tokens = output.shape[1] - inputs["input_ids"].shape[1]
            config_results["cold_tokens"] = int(n_tokens)
            config_results["cold_tok_per_s"] = round(n_tokens / t.elapsed, 1) if t.elapsed > 0 else 0

            # Warm inference (average of 3 runs)
            times = []
            token_counts = []
            for _ in range(3):
                t0 = time.perf_counter()
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=512)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                n_tok = output.shape[1] - inputs["input_ids"].shape[1]
                token_counts.append(int(n_tok))

            avg_time = sum(times) / len(times)
            avg_tokens = sum(token_counts) / len(token_counts)
            config_results["warm_avg_s"] = round(avg_time, 4)
            config_results["warm_avg_tokens"] = round(avg_tokens, 1)
            config_results["warm_tok_per_s"] = round(avg_tokens / avg_time, 1) if avg_time > 0 else 0

            _log.info(
                f"  {dtype_name}: load={config_results['load_time']:.2f}s, "
                f"cold={config_results['cold_inference_s']:.4f}s ({config_results['cold_tok_per_s']} tok/s), "
                f"warm={avg_time:.4f}s ({config_results['warm_tok_per_s']} tok/s)"
            )

            del model, processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            _log.error(f"  Config {dtype_name} failed: {e}")
            config_results["error"] = str(e)

        results[dtype_name] = config_results

    RESULTS["vlm_granite_docling"] = results
    return results


# =========================================================================
# Benchmark 5: Granite-Docling-258M via vLLM
# =========================================================================
def bench_vlm_vllm():
    """Test granite-docling-258M via vLLM engine."""
    _log.info("=" * 60)
    _log.info("BENCHMARK: granite-docling-258M via vLLM")
    _log.info("=" * 60)

    results = {}

    model_id = "ibm-granite/granite-docling-258M"

    # Check local model paths
    local_paths = [
        ARTIFACTS_PATH / "ibm-granite--granite-docling-258M",
        Path(os.path.expanduser("~/neo/docling-models/ibm-granite--granite-docling-258M")),
    ]
    for p in local_paths:
        if p.exists():
            model_id = str(p)
            break

    try:
        from vllm import LLM, SamplingParams
        from PIL import Image
    except ImportError:
        _log.error("vllm not installed")
        return results

    test_img = Image.fromarray(_make_test_image(1200, 1600)).convert("RGB")

    gpu_mem_configs = [0.3, 0.5, 0.7]

    for gpu_mem in gpu_mem_configs:
        _log.info(f"\n--- gpu_memory_utilization={gpu_mem} ---")
        config_results = {}

        try:
            with _timer(f"vLLM init (gpu_mem={gpu_mem})") as t:
                llm = LLM(
                    model=model_id,
                    gpu_memory_utilization=gpu_mem,
                    max_model_len=4096,
                    trust_remote_code=True,
                )
            config_results["init_time"] = t.elapsed

            sampling_params = SamplingParams(
                max_tokens=512,
                temperature=0.0,
            )

            prompt = "Convert this page to docling."
            llm_input = {
                "prompt": prompt,
                "multi_modal_data": {"image": test_img},
            }

            # Warmup
            _ = llm.generate([llm_input], sampling_params=sampling_params)

            # Timed runs
            times = []
            token_counts = []
            for _ in range(3):
                t0 = time.perf_counter()
                outputs = llm.generate([llm_input], sampling_params=sampling_params)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                n_tok = len(outputs[0].outputs[0].token_ids)
                token_counts.append(n_tok)

            avg_time = sum(times) / len(times)
            avg_tokens = sum(token_counts) / len(token_counts)
            config_results["avg_s"] = round(avg_time, 4)
            config_results["avg_tokens"] = round(avg_tokens, 1)
            config_results["tok_per_s"] = round(avg_tokens / avg_time, 1) if avg_time > 0 else 0

            _log.info(
                f"  gpu_mem={gpu_mem}: init={config_results['init_time']:.2f}s, "
                f"avg={avg_time:.4f}s, {config_results['tok_per_s']} tok/s"
            )

            del llm
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            _log.error(f"  gpu_mem={gpu_mem} failed: {e}")
            config_results["error"] = str(e)

        results[f"gpu_mem_{gpu_mem}"] = config_results

    RESULTS["vlm_vllm"] = results
    return results


# =========================================================================
# Benchmark 6: Granite-Vision-3.3-2b Standalone
# =========================================================================
def bench_vlm_granite_vision():
    """Test granite-vision-3.3-2b for picture description."""
    _log.info("=" * 60)
    _log.info("BENCHMARK: granite-vision-3.3-2b Standalone")
    _log.info("=" * 60)

    results = {}

    model_id = "ibm-granite/granite-vision-3.3-2b"

    local_paths = [
        ARTIFACTS_PATH / "ibm-granite--granite-vision-3.3-2b",
        HF_CACHE / "hub" / "models--ibm-granite--granite-vision-3.3-2b",
        Path(os.path.expanduser("~/neo/docling-models/ibm-granite--granite-vision-3.3-2b")),
    ]
    for p in local_paths:
        if p.exists():
            model_id = str(p)
            break

    try:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from PIL import Image
    except ImportError:
        _log.error("transformers not installed")
        return results

    if not torch.cuda.is_available():
        _log.warning("CUDA not available, skipping large VLM benchmark")
        return results

    test_img = Image.fromarray(_make_test_image(800, 600))

    try:
        with _timer("Model load (bf16 cuda)") as t:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            ).to("cuda")
        results["load_time"] = t.elapsed

        prompt = "Describe the contents of this image in detail."
        inputs = processor(images=test_img, text=prompt, return_tensors="pt").to("cuda")

        # Warmup
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=50)

        # Timed runs
        times = []
        token_counts = []
        for _ in range(3):
            t0 = time.perf_counter()
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=256)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            n_tok = output.shape[1] - inputs["input_ids"].shape[1]
            token_counts.append(int(n_tok))

        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        results["avg_s"] = round(avg_time, 4)
        results["avg_tokens"] = round(avg_tokens, 1)
        results["tok_per_s"] = round(avg_tokens / avg_time, 1) if avg_time > 0 else 0

        _log.info(
            f"  load={results['load_time']:.2f}s, "
            f"avg={avg_time:.4f}s, {results['tok_per_s']} tok/s"
        )

        del model, processor
        torch.cuda.empty_cache()

    except Exception as e:
        _log.error(f"  Failed: {e}")
        results["error"] = str(e)

    RESULTS["vlm_granite_vision"] = results
    return results


# =========================================================================
# Benchmark 7: SmolDocling-256M-preview
# =========================================================================
def bench_vlm_smoldocling():
    """Test SmolDocling-256M-preview as comparison to granite-docling."""
    _log.info("=" * 60)
    _log.info("BENCHMARK: SmolDocling-256M-preview")
    _log.info("=" * 60)

    results = {}

    model_id = "docling-project/SmolDocling-256M-preview"

    local_paths = [
        ARTIFACTS_PATH / "docling-project--SmolDocling-256M-preview",
        Path(os.path.expanduser("~/neo/docling-models/docling-project--SmolDocling-256M-preview")),
    ]
    for p in local_paths:
        if p.exists():
            model_id = str(p)
            break

    try:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from PIL import Image
    except ImportError:
        _log.error("transformers not installed")
        return results

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    test_img = Image.fromarray(_make_test_image(1200, 1600))

    try:
        with _timer(f"Model load ({device})") as t:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=dtype,
            ).to(device)
        results["load_time"] = t.elapsed

        prompt = "Convert this page to docling."
        inputs = processor(images=test_img, text=prompt, return_tensors="pt").to(device)

        # Warmup
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=50)

        # Timed runs
        times = []
        token_counts = []
        for _ in range(3):
            t0 = time.perf_counter()
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=512)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            n_tok = output.shape[1] - inputs["input_ids"].shape[1]
            token_counts.append(int(n_tok))

        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        results["avg_s"] = round(avg_time, 4)
        results["avg_tokens"] = round(avg_tokens, 1)
        results["tok_per_s"] = round(avg_tokens / avg_time, 1) if avg_time > 0 else 0

        _log.info(
            f"  load={results['load_time']:.2f}s, "
            f"avg={avg_time:.4f}s, {results['tok_per_s']} tok/s"
        )

        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        _log.error(f"  Failed: {e}")
        results["error"] = str(e)

    RESULTS["vlm_smoldocling"] = results
    return results


# =========================================================================
# Main
# =========================================================================
BENCHMARKS = {
    "rapidocr": bench_rapidocr,
    "layout": bench_layout,
    "tableformer": bench_tableformer,
    "vlm-granite": bench_vlm_granite_docling,
    "vlm-vllm": bench_vlm_vllm,
    "vlm-vision": bench_vlm_granite_vision,
    "vlm-smol": bench_vlm_smoldocling,
}


def main():
    parser = argparse.ArgumentParser(
        description="Standalone model benchmarks for docling pipeline components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        nargs="*",
        choices=list(BENCHMARKS.keys()),
        help="Specific models to benchmark (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name, func in BENCHMARKS.items():
            print(f"  {name:20s} — {func.__doc__.strip().split(chr(10))[0]}")
        return

    models = args.model if args.model else list(BENCHMARKS.keys())

    _log.info(f"Running benchmarks: {models}")
    _log.info(f"Artifacts path: {ARTIFACTS_PATH}")

    for model_name in models:
        if model_name in BENCHMARKS:
            try:
                BENCHMARKS[model_name]()
            except Exception as e:
                _log.error(f"Benchmark {model_name} failed with error: {e}")
                RESULTS[model_name] = {"error": str(e)}

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(json.dumps(RESULTS, indent=2, default=str))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(RESULTS, f, indent=2, default=str)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
