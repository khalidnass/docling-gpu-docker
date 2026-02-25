#!/bin/bash
# ============================================================
# H20 Docling-Serve Diagnostic & Benchmark
# Copy-paste this entire script into OpenShift pod terminal
# ============================================================

# Auto-detect docling-serve URL
if curl -s http://localhost:5001/health 2>/dev/null | grep -q ok; then
  BASE_URL="http://localhost:5001"
elif curl -s http://localhost:8080/health 2>/dev/null | grep -q ok; then
  BASE_URL="http://localhost:8080"
else
  BASE_URL="http://localhost:5001"
fi
echo "Server: $BASE_URL"

# Find python
PY=$(which python3 2>/dev/null || echo "/opt/app-root/bin/python3")

echo ""
echo "========================================================================"
echo "PART 1: DIAGNOSTICS"
echo "========================================================================"

$PY -u << 'PYEOF'
import os, sys, time, json
from pathlib import Path

print("=" * 70)
print("DOCLING-SERVE DIAGNOSTIC REPORT")
print("=" * 70)

# 1. CPU
print("\n[1/10] CPU")
print(f"  Cores: {os.cpu_count()}")
try:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if "model name" in line:
                print(f"  Model: {line.split(':')[1].strip()}")
                break
except: print("  Model: unknown")
print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'NOT SET')}")

# 2. GPU
print("\n[2/10] GPU")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Compute cap: {torch.cuda.get_device_capability(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  VRAM total: {round(mem/1e9, 1)} GB")
        print(f"  VRAM used: {round(torch.cuda.memory_allocated(0)/1e9, 2)} GB")
    else:
        print("  *** CUDA NOT AVAILABLE ***")
except ImportError:
    print("  *** torch not installed ***")

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
        print("  *** CUDA EP: MISSING - OCR RUNNING ON CPU! ***")
except ImportError:
    print("  *** onnxruntime not installed ***")

# 4. RapidOCR CUDA fix
print("\n[4/10] RapidOCR CUDA Fix")
rp = Path("/opt/app-root/lib/python3.12/site-packages/docling/models/stages/ocr/rapid_ocr_model.py")
if rp.exists():
    content = rp.read_text()
    if "EngineConfig.onnxruntime.use_cuda" in content:
        print("  Status: APPLIED")
    else:
        print("  *** MISSING - RapidOCR RUNNING ON CPU! ***")
else:
    print(f"  File not found: {rp}")

# 5. Models
print("\n[5/10] Models")
mp = Path(os.environ.get("DOCLING_SERVE_ARTIFACTS_PATH", "/opt/app-root/src/.cache/docling/models"))
if mp.exists():
    for item in sorted(mp.iterdir()):
        if item.is_dir():
            sz = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            print(f"  {item.name}/ ({round(sz/1e6,1)} MB)")
        else:
            print(f"  {item.name}")
else:
    print(f"  *** NOT FOUND: {mp} ***")

# 6. RapidOCR ONNX models
print("\n[6/10] RapidOCR ONNX Models")
rmp = mp / "RapidOcr" / "onnx" / "PP-OCRv4"
if rmp.exists():
    for f in sorted(rmp.rglob("*.onnx")):
        print(f"  {f.relative_to(rmp)} ({round(f.stat().st_size/1e6,1)} MB)")
else:
    print(f"  NOT FOUND: {rmp}")

# 7. DocumentFigureClassifier
print("\n[7/10] DocumentFigureClassifier")
cv1 = mp / "docling-project--DocumentFigureClassifier"
cv2 = mp / "docling-project--DocumentFigureClassifier-v2.0"
print(f"  v1 path: {'EXISTS' if cv1.exists() else 'MISSING'}")
print(f"  v2.0 path: {'EXISTS' if cv2.exists() else 'MISSING'}")
if cv2.is_symlink():
    print(f"  v2.0 symlink -> {os.readlink(cv2)}")
if cv2.exists():
    cfg = cv2 / "config.json"
    if cfg.exists():
        data = json.loads(cfg.read_text())
        n_labels = len(data.get("id2label", {}))
        print(f"  Classes: {n_labels} (expect 26 for v2.0, 16 for v1)")

# 8. Environment
print("\n[8/10] Key Environment Variables")
for var in ["DOCLING_SERVE_ENG_LOC_NUM_WORKERS", "DOCLING_SERVE_ARTIFACTS_PATH",
            "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
            "TORCH_COMPILE_DISABLE", "TORCHINDUCTOR_DISABLE",
            "DOCLING_PICTURE_DESCRIPTION_VLM_MODEL_ID",
            "DOCLING_PICTURE_DESCRIPTION_VLM_DEVICE"]:
    print(f"  {var}: {os.environ.get(var, 'NOT SET')}")

# 9. cuBLAS test
print("\n[9/10] cuBLAS GPU Test (matmul 2048x2048 x10)")
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
        dt = (time.time() - t0) * 1000
        print(f"  Time: {round(dt, 1)} ms")
        if dt < 50:
            print("  Result: EXCELLENT (GPU fully active)")
        elif dt < 200:
            print("  Result: GOOD")
        elif dt < 1000:
            print("  Result: SLOW (possible GPU issue)")
        else:
            print("  *** VERY SLOW - GPU may not be working! ***")
        del a, b, c
        torch.cuda.empty_cache()
    else:
        print("  Skipped (no CUDA)")
except Exception as e:
    print(f"  *** FAILED: {e} ***")

# 10. ONNX CUDA session test
print("\n[10/10] ONNX CUDA Session Test")
try:
    import onnxruntime as ort
    pcfg = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    det_list = list((mp / "RapidOcr").rglob("*det*.onnx")) if (mp / "RapidOcr").exists() else []
    if det_list:
        sess = ort.InferenceSession(str(det_list[0]), providers=pcfg)
        active = sess.get_providers()
        print(f"  Active providers: {active}")
        if "CUDAExecutionProvider" in active:
            print("  RapidOCR Det on GPU: YES")
        else:
            print("  *** RapidOCR Det on GPU: NO - RUNNING ON CPU! ***")
        del sess
    else:
        print("  No det model found")
except Exception as e:
    print(f"  *** FAILED: {e} ***")

# Bonus: torch.compile status
print("\n[BONUS] torch.compile")
tc = os.environ.get("TORCH_COMPILE_DISABLE", "NOT SET")
ti = os.environ.get("TORCHINDUCTOR_DISABLE", "NOT SET")
print(f"  TORCH_COMPILE_DISABLE={tc}  TORCHINDUCTOR_DISABLE={ti}")
if tc == "1":
    print("  Status: Disabled (safe for all GPUs)")
else:
    print("  *** WARNING: torch.compile enabled - may crash on some GPUs ***")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
PYEOF

echo ""
echo "========================================================================"
echo "PART 2: BENCHMARK"
echo "========================================================================"

# Create a small test PDF using Python (no external files needed)
TESTDIR="/tmp/docling-bench"
mkdir -p "$TESTDIR"

$PY -u << 'MKPDF'
# Generate a minimal test PDF with text + table (no extra deps needed)
import struct, zlib
pdf = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
4 0 obj
<</Length 380>>
stream
BT
/F1 24 Tf 72 720 Td (Docling Benchmark Test Document) Tj
/F1 12 Tf 0 -40 Td (This is a test document for benchmarking docling-serve.) Tj
0 -20 Td (It contains text, a simple table structure, and multiple paragraphs.) Tj
0 -40 Td (Section 1: Introduction) Tj
0 -20 Td (Lorem ipsum dolor sit amet, consectetur adipiscing elit.) Tj
0 -20 Td (Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000206 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
698
%%EOF"""
with open("/tmp/docling-bench/test.pdf", "wb") as f:
    f.write(pdf)
print("Test PDF created: /tmp/docling-bench/test.pdf")
MKPDF

# Wait for server
echo ""
echo "Checking server at $BASE_URL ..."
for i in $(seq 1 12); do
  if curl -s "$BASE_URL/health" 2>/dev/null | grep -q ok; then
    echo "Server is ready."
    break
  fi
  echo "  Waiting... ($((i*5))s)"
  sleep 5
done

# Warmup run (first request is always slow due to model loading)
echo ""
echo "Warmup run (loading models into GPU)..."
WARMUP_START=$(date +%s)
curl -sS -X POST "$BASE_URL/v1/convert/file" \
  -F "files=@/tmp/docling-bench/test.pdf" \
  -F "document_timeout=600" \
  -o /dev/null -w "  HTTP %{http_code} in %{time_total}s\n" 2>&1
WARMUP_END=$(date +%s)
echo "  Warmup took $((WARMUP_END - WARMUP_START))s"

echo ""
echo "----------------------------------------------------------------------"
echo "Running benchmarks (2 runs each)..."
echo "----------------------------------------------------------------------"

# Task 1: Default
echo ""
echo "Task 1: Default (no extra args)"
for run in 1 2; do
  echo -n "  Run $run: "
  RESULT=$(curl -sS -X POST "$BASE_URL/v1/convert/file" \
    -F "files=@/tmp/docling-bench/test.pdf" \
    -F "document_timeout=600" \
    -w "\nHTTPTIME:%{time_total}" 2>&1)
  WALL=$(echo "$RESULT" | grep "HTTPTIME:" | cut -d: -f2)
  PROC=$(echo "$RESULT" | grep -o '"processing_time":[0-9.]*' | cut -d: -f2)
  echo "wall=${WALL}s  proc=${PROC:-N/A}s"
done

# Task 2: RapidOCR
echo ""
echo "Task 2: RapidOCR basic"
for run in 1 2; do
  echo -n "  Run $run: "
  RESULT=$(curl -sS -X POST "$BASE_URL/v1/convert/file" \
    -F "files=@/tmp/docling-bench/test.pdf" \
    -F "ocr_engine=rapidocr" \
    -F "document_timeout=600" \
    -w "\nHTTPTIME:%{time_total}" 2>&1)
  WALL=$(echo "$RESULT" | grep "HTTPTIME:" | cut -d: -f2)
  PROC=$(echo "$RESULT" | grep -o '"processing_time":[0-9.]*' | cut -d: -f2)
  echo "wall=${WALL}s  proc=${PROC:-N/A}s"
done

# Task 3: Advanced (OCR + tables + classification)
echo ""
echo "Task 3: Advanced (OCR + table + classification)"
for run in 1 2; do
  echo -n "  Run $run: "
  RESULT=$(curl -sS -X POST "$BASE_URL/v1/convert/file" \
    -F "files=@/tmp/docling-bench/test.pdf" \
    -F "do_ocr=true" \
    -F "do_table_structure=true" \
    -F "table_mode=accurate" \
    -F "ocr_engine=rapidocr" \
    -F "do_picture_classification=true" \
    -F "document_timeout=600" \
    -w "\nHTTPTIME:%{time_total}" 2>&1)
  WALL=$(echo "$RESULT" | grep "HTTPTIME:" | cut -d: -f2)
  PROC=$(echo "$RESULT" | grep -o '"processing_time":[0-9.]*' | cut -d: -f2)
  echo "wall=${WALL}s  proc=${PROC:-N/A}s"
done

echo ""
echo "========================================================================"
echo "DONE - Copy the full output above and share it"
echo "========================================================================"

# Cleanup
rm -rf "$TESTDIR"
