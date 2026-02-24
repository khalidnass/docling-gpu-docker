#!/usr/bin/env python3
"""
ONNX Runtime Session Options Benchmark — Deep-dive into GPU optimization.

Tests different ONNX RT session configurations for RapidOCR's PP-OCRv4 models
to find optimal settings for GPU execution.

Usage:
    python3 benchmark_onnx_sessions.py
    python3 benchmark_onnx_sessions.py --out-json onnx_results.json
    python3 benchmark_onnx_sessions.py --model det     # Test detection model only
    python3 benchmark_onnx_sessions.py --model rec     # Test recognition model only
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
_log = logging.getLogger("onnx_bench")

ARTIFACTS_PATH = Path(
    os.environ.get(
        "DOCLING_SERVE_ARTIFACTS_PATH",
        os.environ.get(
            "DOCLING_ARTIFACTS_PATH",
            os.path.expanduser("~/.cache/docling/models"),
        ),
    )
)


def _make_test_input(model_type: str) -> np.ndarray:
    """Create appropriate test input for each model type."""
    if model_type == "det":
        # Detection: expects [1, 3, H, W] float32
        return np.random.rand(1, 3, 960, 960).astype(np.float32)
    elif model_type == "cls":
        # Classification: expects [1, 3, 48, 192] float32
        return np.random.rand(1, 3, 48, 192).astype(np.float32)
    elif model_type == "rec":
        # Recognition: expects [1, 3, 48, W] float32 (variable width)
        return np.random.rand(1, 3, 48, 320).astype(np.float32)
    else:
        return np.random.rand(1, 3, 224, 224).astype(np.float32)


def bench_session_options(model_path: str, model_type: str) -> dict:
    """Run comprehensive ONNX session option benchmarks on a single model."""
    import onnxruntime as ort

    results = {}
    _log.info(f"\n{'='*60}")
    _log.info(f"Benchmarking: {model_type} — {Path(model_path).name}")
    _log.info(f"{'='*60}")

    providers = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in providers
    has_tensorrt = "TensorrtExecutionProvider" in providers

    _log.info(f"Available providers: {providers}")

    test_input = _make_test_input(model_type)

    # Base configurations to test
    configs = [
        {
            "name": "cpu_default",
            "providers": ["CPUExecutionProvider"],
            "session_opts": {},
        },
    ]

    if has_cuda:
        # CUDA default
        configs.append(
            {
                "name": "cuda_default",
                "providers": [
                    ("CUDAExecutionProvider", {"device_id": 0}),
                    "CPUExecutionProvider",
                ],
                "session_opts": {},
            }
        )

        # CUDA with cudnn conv algo search: EXHAUSTIVE
        configs.append(
            {
                "name": "cuda_cudnn_exhaustive",
                "providers": [
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": 0,
                            "cudnn_conv_algo_search": "EXHAUSTIVE",
                        },
                    ),
                    "CPUExecutionProvider",
                ],
                "session_opts": {},
            }
        )

        # CUDA with cudnn conv algo search: HEURISTIC
        configs.append(
            {
                "name": "cuda_cudnn_heuristic",
                "providers": [
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": 0,
                            "cudnn_conv_algo_search": "HEURISTIC",
                        },
                    ),
                    "CPUExecutionProvider",
                ],
                "session_opts": {},
            }
        )

        # CUDA with max workspace
        configs.append(
            {
                "name": "cuda_max_workspace",
                "providers": [
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": 0,
                            "cudnn_conv_use_max_workspace": "1",
                        },
                    ),
                    "CPUExecutionProvider",
                ],
                "session_opts": {},
            }
        )

        # CUDA with graph optimization
        configs.append(
            {
                "name": "cuda_graph_opt_all",
                "providers": [
                    ("CUDAExecutionProvider", {"device_id": 0}),
                    "CPUExecutionProvider",
                ],
                "session_opts": {"graph_optimization_level": ort.GraphOptimizationLevel.ORT_ENABLE_ALL},
            }
        )

        # Thread count variations
        for threads in [1, 2, 4, 8]:
            configs.append(
                {
                    "name": f"cuda_threads_{threads}",
                    "providers": [
                        ("CUDAExecutionProvider", {"device_id": 0}),
                        "CPUExecutionProvider",
                    ],
                    "session_opts": {"intra_op_num_threads": threads},
                }
            )

    if has_tensorrt:
        configs.append(
            {
                "name": "tensorrt_default",
                "providers": [
                    (
                        "TensorrtExecutionProvider",
                        {
                            "device_id": 0,
                            "trt_fp16_enable": True,
                        },
                    ),
                    "CPUExecutionProvider",
                ],
                "session_opts": {},
            }
        )

    for config in configs:
        _log.info(f"\n--- Config: {config['name']} ---")

        try:
            # Create session options
            sess_opts = ort.SessionOptions()
            for key, value in config["session_opts"].items():
                setattr(sess_opts, key, value)

            # Create session
            t0 = time.perf_counter()
            sess = ort.InferenceSession(
                model_path,
                sess_options=sess_opts,
                providers=config["providers"],
            )
            init_time = time.perf_counter() - t0

            active_providers = sess.get_providers()
            input_name = sess.get_inputs()[0].name
            input_shape = sess.get_inputs()[0].shape

            _log.info(f"  Active providers: {active_providers}")
            _log.info(f"  Input: {input_name}, shape={input_shape}")
            _log.info(f"  Init time: {init_time:.4f}s")

            # Reshape test input if needed
            inp = test_input
            expected_shape = input_shape
            if len(expected_shape) == 4 and None not in expected_shape:
                inp = np.random.rand(*expected_shape).astype(np.float32)

            # Warmup (5 runs)
            for _ in range(5):
                _ = sess.run(None, {input_name: inp})

            # Timed runs
            n_runs = 20
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = sess.run(None, {input_name: inp})
                times.append(time.perf_counter() - t0)

            avg = sum(times) / len(times)
            std = np.std(times)
            p50 = np.percentile(times, 50)
            p95 = np.percentile(times, 95)
            p99 = np.percentile(times, 99)

            config_result = {
                "init_time_s": round(init_time, 4),
                "active_providers": active_providers,
                "avg_s": round(avg, 6),
                "std_s": round(float(std), 6),
                "min_s": round(min(times), 6),
                "max_s": round(max(times), 6),
                "p50_s": round(float(p50), 6),
                "p95_s": round(float(p95), 6),
                "p99_s": round(float(p99), 6),
                "n_runs": n_runs,
            }

            _log.info(
                f"  avg={avg*1000:.2f}ms, std={std*1000:.2f}ms, "
                f"p50={p50*1000:.2f}ms, p95={p95*1000:.2f}ms"
            )

            del sess

        except Exception as e:
            _log.error(f"  Config {config['name']} failed: {e}")
            config_result = {"error": str(e)}

        results[config["name"]] = config_result

    return results


def bench_io_binding(model_path: str, model_type: str) -> dict:
    """Test IO Binding (zero-copy GPU inference) vs default."""
    import onnxruntime as ort

    results = {}

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        _log.info("Skipping IO Binding test (no CUDA)")
        return results

    _log.info(f"\n{'='*60}")
    _log.info(f"IO Binding Test: {model_type}")
    _log.info(f"{'='*60}")

    try:
        import torch

        sess = ort.InferenceSession(
            model_path,
            providers=[
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ],
        )

        input_name = sess.get_inputs()[0].name
        output_names = [o.name for o in sess.get_outputs()]

        test_input = _make_test_input(model_type)

        # Standard run (numpy)
        for _ in range(5):
            _ = sess.run(None, {input_name: test_input})

        times_standard = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = sess.run(None, {input_name: test_input})
            times_standard.append(time.perf_counter() - t0)

        results["standard_avg_ms"] = round(
            sum(times_standard) / len(times_standard) * 1000, 3
        )

        # IO Binding run
        input_tensor = torch.from_numpy(test_input).cuda()

        io_binding = sess.io_binding()
        io_binding.bind_input(
            name=input_name,
            device_type="cuda",
            device_id=0,
            element_type=np.float32,
            shape=tuple(test_input.shape),
            buffer_ptr=input_tensor.data_ptr(),
        )
        for out_name in output_names:
            io_binding.bind_output(out_name, device_type="cuda", device_id=0)

        # Warmup
        for _ in range(5):
            sess.run_with_iobinding(io_binding)

        times_io = []
        for _ in range(20):
            t0 = time.perf_counter()
            sess.run_with_iobinding(io_binding)
            times_io.append(time.perf_counter() - t0)

        results["io_binding_avg_ms"] = round(
            sum(times_io) / len(times_io) * 1000, 3
        )

        speedup = results["standard_avg_ms"] / results["io_binding_avg_ms"] if results["io_binding_avg_ms"] > 0 else 0
        results["speedup_x"] = round(speedup, 2)

        _log.info(
            f"  Standard: {results['standard_avg_ms']:.3f}ms, "
            f"IO Binding: {results['io_binding_avg_ms']:.3f}ms, "
            f"Speedup: {speedup:.2f}x"
        )

        del sess, input_tensor
        torch.cuda.empty_cache()

    except Exception as e:
        _log.error(f"  IO Binding test failed: {e}")
        results["error"] = str(e)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Runtime session options benchmark for RapidOCR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        nargs="*",
        choices=["det", "cls", "rec"],
        default=["det", "cls", "rec"],
        help="Which models to test (default: all)",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    # Locate models
    model_paths = {
        "det": ARTIFACTS_PATH / "RapidOcr" / "onnx" / "PP-OCRv4" / "det" / "ch_PP-OCRv4_det_infer.onnx",
        "cls": ARTIFACTS_PATH / "RapidOcr" / "onnx" / "PP-OCRv4" / "cls" / "ch_ppocr_mobile_v2.0_cls_infer.onnx",
        "rec": ARTIFACTS_PATH / "RapidOcr" / "onnx" / "PP-OCRv4" / "rec" / "ch_PP-OCRv4_rec_infer.onnx",
    }

    # Check ONNX Runtime
    try:
        import onnxruntime as ort

        _log.info(f"ONNX Runtime: {ort.__version__}")
        _log.info(f"Providers: {ort.get_available_providers()}")
    except ImportError:
        _log.error("onnxruntime not installed!")
        sys.exit(1)

    all_results = {}

    for model_type in args.model:
        model_path = model_paths[model_type]
        if not model_path.exists():
            _log.warning(f"Model not found: {model_path}")
            continue

        _log.info(f"\nModel: {model_path.name} ({model_path.stat().st_size / 1e6:.1f} MB)")

        # Session options benchmark
        session_results = bench_session_options(str(model_path), model_type)
        all_results[f"{model_type}_sessions"] = session_results

        # IO Binding benchmark
        io_results = bench_io_binding(str(model_path), model_type)
        if io_results:
            all_results[f"{model_type}_io_binding"] = io_results

    # Print summary
    print("\n" + "=" * 70)
    print("ONNX SESSION OPTIONS BENCHMARK SUMMARY")
    print("=" * 70)

    for section, data in all_results.items():
        print(f"\n--- {section} ---")
        if isinstance(data, dict):
            for config_name, config_data in data.items():
                if isinstance(config_data, dict):
                    avg = config_data.get("avg_s")
                    if avg is not None:
                        print(f"  {config_name:30s}: {avg*1000:8.2f}ms")
                    elif "error" in config_data:
                        print(f"  {config_name:30s}: ERROR - {config_data['error'][:50]}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
