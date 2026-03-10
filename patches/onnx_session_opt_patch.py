#!/usr/bin/env python3
"""Build-time patch: Optimize ONNX Runtime session options for GPU throughput.

Configures ONNX RT sessions with:
  - enable_mem_pattern=True: Reuse memory allocations across inference calls
  - execution_mode=ORT_SEQUENTIAL: Better for GPU (avoids CPU thread pool overhead)
  - intra_op_num_threads=1: Minimal CPU threading (GPU does the work)

Note: Full IO Binding (keeping tensors on GPU between det→cls→rec) is not
implemented because CPU postprocessing between stages (box extraction, image
cropping, CTC decode) requires data on CPU. The GPU↔CPU copy overhead is
inherent to the PP-OCR architecture.

Applied at Docker build time.
"""
from pathlib import Path

ORT_SESSION = Path(
    "/opt/app-root/lib/python3.12/site-packages/"
    "rapidocr/inference_engine/onnxruntime/main.py"
)

src = ORT_SESSION.read_text()

# Patch _init_sess_opts to add memory pattern and execution mode
OLD_OPTS = """\
    @staticmethod
    def _init_sess_opts(cfg: Dict[str, Any]) -> SessionOptions:
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = cfg.enable_cpu_mem_arena
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL"""

NEW_OPTS = """\
    @staticmethod
    def _init_sess_opts(cfg: Dict[str, Any]) -> SessionOptions:
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = cfg.enable_cpu_mem_arena
        sess_opt.enable_mem_pattern = True
        sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL"""

assert OLD_OPTS in src, "FATAL: ONNX session opts patch target not found"
src = src.replace(OLD_OPTS, NEW_OPTS)

# Ensure onnxruntime is imported as ort (for ExecutionMode)
if "import onnxruntime as ort" not in src:
    src = src.replace(
        "from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions",
        "import onnxruntime as ort\n"
        "from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions",
    )

ORT_SESSION.write_text(src)
print("[BUILD] ONNX session optimization patch applied successfully", flush=True)
