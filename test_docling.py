#!/usr/bin/env python3
"""
Docling-Serve Diagnostic & Benchmark Script

Usage:
  # Run diagnostics only (inside pod or locally)
  python3 test_docling.py --diag

  # Run benchmark against local or remote docling-serve
  python3 test_docling.py --benchmark --base-url http://localhost:5001 --input-dir ./files

  # Run both diagnostics and benchmark
  python3 test_docling.py --diag --benchmark --base-url http://localhost:5001 --input-dir ./files

  # Run benchmark with specific tasks (1=default, 2=rapidocr, 3=advanced, 4=all features)
  python3 test_docling.py --benchmark --base-url http://localhost:5001 --input-dir ./files --tasks 1 2 3

  # Specify number of runs per test
  python3 test_docling.py --benchmark --base-url http://localhost:5001 --input-dir ./files --runs 3
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# DIAGNOSTICS (run inside pod or locally)
# ============================================================

def run_diagnostics():
    print("=" * 70)
    print("DOCLING-SERVE DIAGNOSTIC REPORT")
    print("=" * 70)

    # 1. CPU
    print("\n[1/10] CPU")
    print(f"  Cores available: {os.cpu_count()}")
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    print(f"  Model: {line.split(':')[1].strip()}")
                    break
    except Exception:
        print("  Model: (could not read /proc/cpuinfo)")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'not set')}")

    # 2. GPU
    print("\n[2/10] GPU")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  PyTorch CUDA: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Compute cap: {torch.cuda.get_device_capability(0)}")
            print(f"  VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)} GB")
            print(f"  VRAM used: {round(torch.cuda.memory_allocated(0) / 1e9, 2)} GB")
        else:
            print("  CUDA: NOT AVAILABLE")
    except ImportError:
        print("  torch not installed")

    # 3. ONNX Runtime
    print("\n[3/10] ONNX Runtime")
    try:
        import onnxruntime as ort
        print(f"  Version: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"  Providers: {providers}")
        if "CUDAExecutionProvider" in providers:
            print("  CUDA EP: YES")
        else:
            print("  CUDA EP: MISSING!")
    except ImportError:
        print("  onnxruntime not installed")

    # 4. RapidOCR CUDA fix
    print("\n[4/10] RapidOCR CUDA Fix")
    rapid_path = Path("/opt/app-root/lib/python3.12/site-packages/docling/models/stages/ocr/rapid_ocr_model.py")
    if rapid_path.exists():
        content = rapid_path.read_text()
        if "EngineConfig.onnxruntime.use_cuda" in content:
            print("  Status: APPLIED")
        else:
            print("  Status: MISSING - RapidOCR running on CPU!")
    else:
        print(f"  File not found: {rapid_path}")

    # 5. Models
    print("\n[5/10] Models")
    models_path = Path(os.environ.get("DOCLING_SERVE_ARTIFACTS_PATH",
                                       "/opt/app-root/src/.cache/docling/models"))
    if models_path.exists():
        for item in sorted(models_path.iterdir()):
            if item.is_dir():
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                print(f"  {item.name}/ ({round(size/1e6, 1)} MB)")
            else:
                print(f"  {item.name} ({round(item.stat().st_size/1e6, 1)} MB)")
    else:
        print(f"  NOT FOUND: {models_path}")

    # 6. RapidOCR ONNX models
    print("\n[6/10] RapidOCR ONNX Models")
    rapid_model_path = models_path / "RapidOcr" / "onnx" / "PP-OCRv4"
    if rapid_model_path.exists():
        for f in sorted(rapid_model_path.rglob("*.onnx")):
            print(f"  {f.relative_to(rapid_model_path)} ({round(f.stat().st_size/1e6, 1)} MB)")
    else:
        print(f"  NOT FOUND: {rapid_model_path}")

    # 7. DocumentFigureClassifier path
    print("\n[7/10] DocumentFigureClassifier")
    clf_v1 = models_path / "docling-project--DocumentFigureClassifier"
    clf_v2 = models_path / "docling-project--DocumentFigureClassifier-v2.0"
    print(f"  DocumentFigureClassifier: {'EXISTS' if clf_v1.exists() else 'MISSING'}")
    print(f"  DocumentFigureClassifier-v2.0: {'EXISTS' if clf_v2.exists() else 'MISSING'}")
    if clf_v2.is_symlink():
        print(f"  v2.0 is symlink -> {os.readlink(clf_v2)}")
    if not clf_v2.exists():
        print("  WARNING: Classification will fail! Need symlink or rename.")

    # 8. Environment
    print("\n[8/10] Key Environment Variables")
    env_vars = [
        "DOCLING_SERVE_ENG_LOC_NUM_WORKERS",
        "DOCLING_SERVE_ARTIFACTS_PATH",
        "DOCLING_ARTIFACTS_PATH",
        "LD_LIBRARY_PATH",
        "NVIDIA_VISIBLE_DEVICES",
        "DOCLING_PICTURE_DESCRIPTION_VLM_MODEL_ID",
        "DOCLING_PICTURE_DESCRIPTION_VLM_DEVICE",
        "TESSDATA_PREFIX",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    for var in env_vars:
        print(f"  {var}: {os.environ.get(var, 'not set')}")

    # 9. cuBLAS test
    print("\n[9/10] cuBLAS GPU Test")
    try:
        import torch
        if torch.cuda.is_available():
            a = torch.randn(2048, 2048, device="cuda")
            b = torch.randn(2048, 2048, device="cuda")
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                c = a @ b
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"  matmul 2048x2048 x10: {round((t1-t0)*1000, 1)} ms")
            print("  cuBLAS: OK")
            del a, b, c
            torch.cuda.empty_cache()
        else:
            print("  Skipped (no CUDA)")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 10. ONNX CUDA session test
    print("\n[10/10] ONNX CUDA Session Test")
    try:
        import onnxruntime as ort
        providers_cfg = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        det_candidates = list((models_path / "RapidOcr").rglob("*det*.onnx")) if (models_path / "RapidOcr").exists() else []
        if det_candidates:
            det_model = str(det_candidates[0])
            sess = ort.InferenceSession(det_model, providers=providers_cfg)
            active = sess.get_providers()
            print(f"  Det model providers: {active}")
            if "CUDAExecutionProvider" in active:
                print("  RapidOCR Det on GPU: YES")
            else:
                print("  RapidOCR Det on GPU: NO - running on CPU!")
            del sess
        else:
            print("  No det model found for testing")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Security packages
    print("\n[BONUS] Security Package Versions")
    try:
        import PIL
        print(f"  pillow: {PIL.__version__}")
    except Exception:
        print("  pillow: not installed")
    try:
        import cryptography
        print(f"  cryptography: {cryptography.__version__}")
    except Exception:
        print("  cryptography: not installed")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


# ============================================================
# BENCHMARK (run locally against docling-serve)
# ============================================================

TASKS = {
    "1": {
        "name": "default (no args)",
        "fields": [("document_timeout", "600")],
    },
    "2": {
        "name": "rapidocr basic",
        "fields": [("ocr_engine", "rapidocr"), ("document_timeout", "600")],
    },
    "3": {
        "name": "rapidocr advanced",
        "fields": [
            ("do_ocr", "true"),
            ("do_table_structure", "true"),
            ("table_mode", "accurate"),
            ("ocr_engine", "rapidocr"),
            ("do_code_enrichment", "true"),
            ("do_formula_enrichment", "true"),
            ("do_picture_classification", "true"),
            ("document_timeout", "600"),
        ],
    },
    "4": {
        "name": "all features",
        "fields": [
            ("do_ocr", "true"),
            ("do_table_structure", "true"),
            ("table_mode", "accurate"),
            ("ocr_engine", "rapidocr"),
            ("do_code_enrichment", "true"),
            ("do_formula_enrichment", "true"),
            ("do_picture_classification", "true"),
            ("do_picture_description", "true"),
            ("document_timeout", "600"),
        ],
    },
}


def find_pdfs(input_dir):
    d = Path(input_dir)
    if not d.is_dir():
        sys.exit(f"Input directory not found: {d}")
    pdfs = sorted(d.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in: {d}")
    return pdfs


def convert_file(base_url, pdf_path, fields, timeout_s=600):
    url = f"{base_url.rstrip('/')}/v1/convert/file"
    cmd = ["curl", "-sS", "--fail-with-body", "-X", "POST", url,
           "-F", f"files=@{pdf_path}"]
    for k, v in fields:
        cmd.extend(["-F", f"{k}={v}"])
    cmd.extend(["-w", "\nHTTPSTATUS:%{http_code}"])

    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}

    wall = time.perf_counter() - start
    stdout = proc.stdout.decode("utf-8", errors="replace")

    # Parse HTTP status
    lines = stdout.rsplit("\nHTTPSTATUS:", 1)
    body = lines[0] if len(lines) == 2 else stdout
    http_code = int(lines[1].strip()) if len(lines) == 2 else None

    if proc.returncode != 0 or (http_code and http_code >= 400):
        return {"ok": False, "error": f"http_{http_code}", "wall": wall}

    try:
        data = json.loads(body)
        proc_time = data.get("processing_time")
        return {
            "ok": True,
            "wall": wall,
            "processing_time": float(proc_time) if proc_time else None,
        }
    except Exception:
        return {"ok": False, "error": "invalid_json", "wall": wall}


def run_benchmark(base_url, input_dir, tasks, runs, timeout):
    pdfs = find_pdfs(input_dir)

    # Check health
    try:
        proc = subprocess.run(
            ["curl", "-s", f"{base_url.rstrip('/')}/health"],
            capture_output=True, timeout=10, check=False)
        health = proc.stdout.decode()
        if '"ok"' not in health:
            print(f"WARNING: Health check failed: {health}")
    except Exception as e:
        print(f"WARNING: Cannot reach server: {e}")

    results = {}
    task_list = [t for t in tasks if t in TASKS]

    print(f"\nBenchmark: {len(pdfs)} files x {len(task_list)} tasks x {runs} runs")
    print(f"Server: {base_url}")
    print("-" * 70)

    for pdf in pdfs:
        fname = pdf.name
        results[fname] = {}

        for tid in task_list:
            task = TASKS[tid]
            walls = []
            procs = []
            errors = []

            for r in range(runs):
                res = convert_file(base_url, str(pdf), task["fields"], timeout)
                if res["ok"]:
                    walls.append(res["wall"])
                    if res.get("processing_time"):
                        procs.append(res["processing_time"])
                else:
                    errors.append(res.get("error", "unknown"))

            results[fname][task["name"]] = {
                "wall_avg": round(sum(walls) / len(walls), 2) if walls else None,
                "proc_avg": round(sum(procs) / len(procs), 2) if procs else None,
                "wall_min": round(min(walls), 2) if walls else None,
                "wall_max": round(max(walls), 2) if walls else None,
                "runs_ok": len(walls),
                "runs_fail": len(errors),
                "errors": errors[:3],
            }

            status = f"{round(sum(procs)/len(procs), 1)}s" if procs else "FAIL"
            print(f"  {fname:20s} | {task['name']:20s} | {status}")

    return results


def print_results_table(results):
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'File':<22} {'Task':<22} {'Proc Avg (s)':>12} {'Wall Avg (s)':>12} {'OK/Fail':>8}")
    print("-" * 90)

    for fname, tasks in results.items():
        for task_name, data in tasks.items():
            proc = f"{data['proc_avg']:.2f}" if data['proc_avg'] else "N/A"
            wall = f"{data['wall_avg']:.2f}" if data['wall_avg'] else "N/A"
            ok_fail = f"{data['runs_ok']}/{data['runs_fail']}"
            print(f"{fname:<22} {task_name:<22} {proc:>12} {wall:>12} {ok_fail:>8}")
        print("-" * 90)


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Docling-Serve Diagnostic & Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--diag", action="store_true",
                    help="Run diagnostics (GPU, ONNX, models, env)")
    p.add_argument("--benchmark", action="store_true",
                    help="Run benchmark against docling-serve")
    p.add_argument("--base-url", default="http://localhost:5001",
                    help="Docling-serve URL (default: http://localhost:5001)")
    p.add_argument("--input-dir", default="./files",
                    help="Directory with test PDF files (default: ./files)")
    p.add_argument("--tasks", nargs="*", default=["1", "2", "3"],
                    help="Task IDs: 1=default 2=rapidocr 3=advanced 4=all (default: 1 2 3)")
    p.add_argument("--runs", type=int, default=2,
                    help="Runs per test (default: 2)")
    p.add_argument("--timeout", type=int, default=600,
                    help="Timeout per request in seconds (default: 600)")
    p.add_argument("--out-json", default=None,
                    help="Save results to JSON file")

    args = p.parse_args()

    if not args.diag and not args.benchmark:
        p.print_help()
        print("\nSpecify --diag and/or --benchmark")
        sys.exit(1)

    if args.diag:
        run_diagnostics()

    if args.benchmark:
        results = run_benchmark(args.base_url, args.input_dir,
                                args.tasks, args.runs, args.timeout)
        print_results_table(results)

        if args.out_json:
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            data = {
                "timestamp": ts,
                "base_url": args.base_url,
                "runs": args.runs,
                "tasks": args.tasks,
                "results": results,
            }
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
