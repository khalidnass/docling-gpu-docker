#!/usr/bin/env python3
"""
Benchmark RapidOCR v3.7.0 with PP-OCRv5 server models: CPU vs GPU configurations.

Configs tested:
  1. v5-cpu-default   — ORT CPU, default settings
  2. v5-gpu-default   — ORT CUDA EP, default settings
  3. v5-gpu-tuned     — ORT CUDA EP + EXHAUSTIVE cuDNN + kNextPowerOfTwo + threads=1
  4. v5-gpu-tuned-batch — Same as #3 + rec_batch_num=12, cls_batch_num=12

Usage:
  python3 benchmark_rapidocr.py [--files FILE1 FILE2 ...] [--runs N] [--dpi DPI]
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── PDF → image ──────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str, dpi: int = 144) -> List[np.ndarray]:
    """Convert PDF pages to numpy arrays at given DPI using pypdfium2."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path)
    images = []
    scale = dpi / 72.0
    for page in doc:
        bitmap = page.render(scale=scale)
        pil_img = bitmap.to_pil()
        # Convert to BGR (OpenCV format) as RapidOCR expects
        arr = np.array(pil_img)
        if arr.ndim == 3 and arr.shape[2] == 4:  # RGBA
            arr = arr[:, :, :3]
        arr = arr[:, :, ::-1]  # RGB → BGR
        images.append(arr)
    doc.close()
    return images


# ── Config definitions ───────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent / "models" / "ppocr_v5"
DET_MODEL = str(MODEL_DIR / "PP-OCRv5_server_det_infer.onnx")
REC_MODEL = str(MODEL_DIR / "PP-OCRv5_server_rec_infer.onnx")
REC_KEYS = str(MODEL_DIR / "ppocrv5_dict.txt")


def make_v5_params(use_cuda: bool, big_batch: bool = False) -> Dict[str, Any]:
    """Build RapidOCR params dict for PP-OCRv5 server models."""
    params: Dict[str, Any] = {
        "Det.model_path": DET_MODEL,
        "Rec.model_path": REC_MODEL,
        "Rec.rec_keys_path": REC_KEYS,
    }

    if use_cuda:
        params["EngineConfig.onnxruntime.use_cuda"] = True
        params["EngineConfig.onnxruntime.cuda_ep_cfg"] = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }

    if big_batch:
        params["Cls.cls_batch_num"] = 12
        params["Rec.rec_batch_num"] = 12

    return params


def make_v4_params(use_cuda: bool, big_batch: bool = False) -> Dict[str, Any]:
    """Build RapidOCR params dict for default PP-OCRv4 mobile models."""
    params: Dict[str, Any] = {}

    if use_cuda:
        params["EngineConfig.onnxruntime.use_cuda"] = True
        params["EngineConfig.onnxruntime.cuda_ep_cfg"] = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }

    if big_batch:
        params["Cls.cls_batch_num"] = 12
        params["Rec.rec_batch_num"] = 12

    return params


CONFIGS = [
    ("v4-gpu-default",       lambda: make_v4_params(use_cuda=True)),
    ("v4-gpu-batch12",       lambda: make_v4_params(use_cuda=True, big_batch=True)),
    ("v5-gpu-default",       lambda: make_v5_params(use_cuda=True)),
    ("v5-gpu-batch12",       lambda: make_v5_params(use_cuda=True, big_batch=True)),
]


# ── Per-image benchmark with stage timing ────────────────────────────────────

def benchmark_image(engine, img: np.ndarray) -> Dict[str, Any]:
    """Run OCR on a single image, returning per-stage timings."""
    op_record: Dict[str, Any] = {}

    # Preprocess
    t0 = time.perf_counter()
    processed_img, op_record = engine.preprocess_img(img.copy())
    t_preprocess = time.perf_counter() - t0

    # Detection
    t0 = time.perf_counter()
    try:
        cropped_img_list, det_res = engine.detect_and_crop(processed_img, op_record)
    except Exception:
        cropped_img_list = []
        det_res = None
    t_det = time.perf_counter() - t0
    num_boxes = len(cropped_img_list)

    # Classification
    t_cls = 0.0
    if num_boxes > 0 and engine.use_cls:
        t0 = time.perf_counter()
        try:
            cls_img_list, cls_res = engine.cls_and_rotate(cropped_img_list)
        except Exception:
            cls_img_list = cropped_img_list
        t_cls = time.perf_counter() - t0
    else:
        cls_img_list = cropped_img_list

    # Recognition
    t_rec = 0.0
    num_texts = 0
    if num_boxes > 0 and engine.use_rec:
        t0 = time.perf_counter()
        try:
            rec_res = engine.recognize_txt(cls_img_list)
            if rec_res.txts:
                num_texts = len([t for t in rec_res.txts if t.strip()])
        except Exception:
            pass
        t_rec = time.perf_counter() - t0

    total = t_preprocess + t_det + t_cls + t_rec
    return {
        "preprocess": t_preprocess,
        "det": t_det,
        "cls": t_cls,
        "rec": t_rec,
        "total": total,
        "num_boxes": num_boxes,
        "num_texts": num_texts,
    }


# ── Main benchmark logic ────────────────────────────────────────────────────

def run_benchmark(
    config_name: str,
    params_fn,
    images: List[np.ndarray],
    file_label: str,
    num_warmup: int = 2,
    num_runs: int = 3,
) -> Dict[str, Any]:
    """Benchmark a single config on a set of images, returning aggregated results."""
    from rapidocr import RapidOCR

    print(f"\n{'='*70}")
    print(f"  Config: {config_name}  |  File: {file_label}  |  {len(images)} pages")
    print(f"{'='*70}")

    params = params_fn()
    engine = RapidOCR(params=params)

    # Warmup
    print(f"  Warming up ({num_warmup} images)...", end=" ", flush=True)
    warmup_imgs = images[:num_warmup] if len(images) >= num_warmup else images
    for wi in warmup_imgs:
        engine(wi)
    print("done")

    # Benchmark runs
    all_runs = []
    for run_idx in range(num_runs):
        run_results = []
        t_run_start = time.perf_counter()
        for img_idx, img in enumerate(images):
            result = benchmark_image(engine, img)
            run_results.append(result)
            print(f"    [Run {run_idx+1}] Page {img_idx+1}/{len(images)}: "
                  f"{result['total']:.2f}s (det={result['det']:.2f} cls={result['cls']:.2f} "
                  f"rec={result['rec']:.2f} boxes={result['num_boxes']})", flush=True)
        t_run_total = time.perf_counter() - t_run_start

        # Aggregate this run
        totals = {
            "det": sum(r["det"] for r in run_results),
            "cls": sum(r["cls"] for r in run_results),
            "rec": sum(r["rec"] for r in run_results),
            "total": t_run_total,
            "num_boxes": sum(r["num_boxes"] for r in run_results),
            "num_texts": sum(r["num_texts"] for r in run_results),
        }
        all_runs.append(totals)
        print(f"  Run {run_idx+1}/{num_runs}: {t_run_total:.2f}s "
              f"(det={totals['det']:.2f}s cls={totals['cls']:.2f}s rec={totals['rec']:.2f}s "
              f"boxes={totals['num_boxes']} texts={totals['num_texts']})")

    # Compute mean ± std
    stats = {}
    for key in ["det", "cls", "rec", "total"]:
        vals = [r[key] for r in all_runs]
        stats[key] = {"mean": np.mean(vals), "std": np.std(vals)}

    stats["num_boxes"] = int(np.mean([r["num_boxes"] for r in all_runs]))
    stats["num_texts"] = int(np.mean([r["num_texts"] for r in all_runs]))

    print(f"  ── Mean: {stats['total']['mean']:.2f}s ± {stats['total']['std']:.2f}s "
          f"(det={stats['det']['mean']:.2f}s cls={stats['cls']['mean']:.2f}s "
          f"rec={stats['rec']['mean']:.2f}s)")

    return {
        "config": config_name,
        "file": file_label,
        "pages": len(images),
        "stats": stats,
        "runs": all_runs,
    }


def print_summary(all_results: List[Dict[str, Any]]):
    """Print a comparison table of all benchmark results."""
    print(f"\n\n{'='*100}")
    print("  SUMMARY: PP-OCRv5 Server — CPU vs GPU Benchmark")
    print(f"{'='*100}")

    # Group by file
    files = list(dict.fromkeys(r["file"] for r in all_results))
    configs = list(dict.fromkeys(r["config"] for r in all_results))

    # Header
    hdr = f"{'Config':<25} {'File':<18} {'Pages':>5} {'Total (s)':>12} {'Det (s)':>10} {'Cls (s)':>10} {'Rec (s)':>10} {'Boxes':>7} {'Texts':>7}"
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for f in files:
        for c in configs:
            matches = [r for r in all_results if r["file"] == f and r["config"] == c]
            if not matches:
                continue
            r = matches[0]
            s = r["stats"]
            print(f"{c:<25} {f:<18} {r['pages']:>5} "
                  f"{s['total']['mean']:>8.2f}±{s['total']['std']:.2f} "
                  f"{s['det']['mean']:>7.2f}±{s['det']['std']:.1f} "
                  f"{s['cls']['mean']:>7.2f}±{s['cls']['std']:.1f} "
                  f"{s['rec']['mean']:>7.2f}±{s['rec']['std']:.1f} "
                  f"{s['num_boxes']:>7} {s['num_texts']:>7}")
        print()

    # Speedup table
    print(f"\n{'─'*60}")
    print("  Speedup vs CPU (total time)")
    print(f"{'─'*60}")
    hdr2 = f"{'File':<18} {'Config':<25} {'Speedup':>10}"
    print(hdr2)
    print("-" * len(hdr2))

    for f in files:
        cpu_result = [r for r in all_results if r["file"] == f and r["config"] == configs[0]]
        if not cpu_result:
            continue
        cpu_time = cpu_result[0]["stats"]["total"]["mean"]
        for c in configs:
            matches = [r for r in all_results if r["file"] == f and r["config"] == c]
            if not matches:
                continue
            gpu_time = matches[0]["stats"]["total"]["mean"]
            speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
            print(f"{f:<18} {c:<25} {speedup:>9.1f}x")
        print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark RapidOCR v3.7.0 PP-OCRv5 CPU vs GPU")
    parser.add_argument("--files", nargs="+",
                        default=[
                            "/home/jalalirs/neo/docling_benchmark/files/testcase1.pdf",
                            "/home/jalalirs/neo/docling_benchmark/files/testcase3.pdf",
                            "/home/jalalirs/neo/docling_benchmark/files/testcase5.pdf",
                        ],
                        help="PDF files to benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs per config")
    parser.add_argument("--dpi", type=int, default=144, help="DPI for PDF rendering")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup images")
    parser.add_argument("--configs", nargs="+",
                        choices=[c[0] for c in CONFIGS],
                        default=None,
                        help="Subset of configs to run (default: all)")
    args = parser.parse_args()

    # Validate models exist
    for p in [DET_MODEL, REC_MODEL, REC_KEYS]:
        if not os.path.exists(p):
            print(f"ERROR: Model not found: {p}")
            sys.exit(1)

    # Validate PDF files
    for f in args.files:
        if not os.path.exists(f):
            print(f"ERROR: PDF not found: {f}")
            sys.exit(1)

    # Select configs
    configs_to_run = CONFIGS
    if args.configs:
        configs_to_run = [(n, fn) for n, fn in CONFIGS if n in args.configs]

    # Pre-render all PDFs
    print("Rendering PDFs to images...")
    file_images: List[Tuple[str, List[np.ndarray]]] = []
    for pdf_path in args.files:
        label = Path(pdf_path).stem
        images = pdf_to_images(pdf_path, dpi=args.dpi)
        print(f"  {label}: {len(images)} pages, "
              f"shape={images[0].shape if images else 'N/A'}")
        file_images.append((label, images))

    # Run benchmarks
    all_results = []
    for config_name, params_fn in configs_to_run:
        for file_label, images in file_images:
            result = run_benchmark(
                config_name, params_fn, images, file_label,
                num_warmup=args.warmup, num_runs=args.runs,
            )
            all_results.append(result)

    # Summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
