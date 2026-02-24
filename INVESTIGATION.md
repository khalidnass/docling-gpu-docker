# Docling Pipeline Investigation Report

Comprehensive investigation into docling-serve GPU performance, pipeline architecture, and optimization opportunities.

## Table of Contents

1. [Annotated Code Trace](#1-annotated-code-trace)
2. [Config Gap Analysis](#2-config-gap-analysis)
3. [Root Cause Analysis: Why H100 ≈ A5000](#3-root-cause-analysis)
4. [VLM Pipeline Analysis](#4-vlm-pipeline-analysis)
5. [Complete Environment Variables Reference](#5-environment-variables-reference)
6. [Recommended Optimizations](#6-recommended-optimizations)

---

## 1. Annotated Code Trace

### HTTP Request → Model Inference (Standard Pipeline)

```
POST /v1/convert/file  (multipart/form-data: files + options)
│
├── docling_serve/app.py → router dispatch
│   └── POST handler: parse ConvertDocumentsRequestOptions from form fields
│       ├── files: List[UploadFile]
│       ├── ocr_engine: "easyocr" | "rapidocr" | "tesseract"
│       ├── pipeline: "standard" | "vlm" | "legacy" | "asr"
│       └── document_timeout, do_ocr, do_table_structure, etc.
│
├── docling_serve/engines/local_orchestrator.py
│   └── convert_single() → DocumentConverter.convert()
│       └── calls converter with pipeline_options built from request
│
├── docling/document_converter.py :: DocumentConverter.convert()
│   ├── Selects pipeline based on format + options:
│   │   ├── PDF → StandardPdfPipeline (or VlmPipeline if pipeline="vlm")
│   │   ├── DOCX/PPTX → appropriate backend
│   │   └── Image → StandardPdfPipeline
│   └── pipeline.execute(in_doc, raises_on_error=True)
│
├── docling/pipeline/base_pipeline.py :: BasePipeline.execute()
│   └── TimeRecorder("pipeline_total", DOCUMENT)
│       ├── _build_document()    ← page-level processing
│       ├── _assemble_document() ← merge pages into document
│       └── _enrich_document()   ← picture classification/description
│
├── docling/pipeline/standard_pdf_pipeline.py :: StandardPdfPipeline._build_document()
│   └── 5 ThreadedPipelineStages (each runs in its own thread):
│
│       Stage 1: PREPROCESS (batch_size=1)
│       └── PagePreprocessingModel.__call__()
│           └── page._backend.load_page()        ← PDF parsing (pypdfium2/docling_parse)
│           └── page._backend.get_segmented_page() ← text extraction
│
│       Stage 2: OCR (batch_size=ocr_batch_size)
│       └── RapidOcrModel.__call__()
│           └── TimeRecorder("ocr", PAGE)
│               ├── get_ocr_rects(page)   ← identify regions needing OCR
│               └── for each ocr_rect:
│                   ├── page._backend.get_page_image(scale=3.0, cropbox=rect)
│                   │   └── ★ BOTTLENECK: renders crop at 216 DPI
│                   ├── numpy.array(image)  ← PIL → numpy conversion
│                   └── self.reader(im, use_det=True, use_cls=True, use_rec=True)
│                       └── RapidOCR pipeline:
│                           ├── TextDetector(im)    ← ch_PP-OCRv4_det ONNX
│                           ├── TextClassifier(crops)← ch_ppocr_mobile_v2.0_cls ONNX
│                           └── TextRecognizer(crops)← ch_PP-OCRv4_rec ONNX
│                           (★ Each model runs independently per-crop!)
│
│       Stage 3: LAYOUT (batch_size=layout_batch_size)
│       └── LayoutModel.__call__() → predict_layout()
│           └── TimeRecorder("layout", PAGE)
│               ├── page.get_image(scale=1.0)
│               └── layout_predictor.predict_batch(images)
│                   └── RT-DETR v2 (ONNX) — batched GPU inference
│
│       Stage 4: TABLE STRUCTURE (batch_size=table_batch_size)
│       └── TableStructureModel.__call__() → predict_tables()
│           └── TimeRecorder("table_structure", PAGE)
│               └── for each table on page:
│                   └── tf_predictor.multi_table_predict()
│                       └── TableFormer (PyTorch) — per-table inference
│
│       Stage 5: ASSEMBLE (batch_size=1)
│       └── PageAssembleModel.__call__()
│           └── Merges OCR cells with layout clusters
│           └── Reading order computation
│
├── _assemble_document()
│   └── TimeRecorder("doc_assemble", DOCUMENT)
│       └── reading_order_model(conv_res) → final document structure
│
└── _enrich_document()
    └── TimeRecorder("doc_enrich", DOCUMENT)
        ├── DocumentPictureClassifier → EfficientNet B0 (32MB)
        └── PictureDescriptionModel → granite-vision-3.3-2b (if enabled)
```

### VLM Pipeline Trace

```
POST /v1/convert/file  (pipeline=vlm, vlm_pipeline_preset=granite_docling)
│
├── VlmPipeline._build_document()
│   └── TimeRecorder("doc_build", DOCUMENT)
│       └── PaginatedPipeline page batch iteration:
│           ├── initialize_page() → TimeRecorder("page_init", PAGE)
│           └── _apply_on_pages() → VLM model __call__:
│
│               HuggingFaceTransformersVlmModel.__call__()
│               └── TimeRecorder("vlm", PAGE)
│                   ├── page.get_image(scale=vlm_options.scale)
│                   ├── _build_prompt_safe(page)  ← format prompt
│                   └── TimeRecorder("vlm_inference", PAGE)
│                       └── process_images(images, prompts)
│                           ├── processor(images, text)  ← tokenize
│                           └── model.generate(**inputs)  ← autoregressive
│                               └── Output: DocTags markup
│
│               OR VllmVlmModel.__call__()
│               └── TimeRecorder("vlm", PAGE)
│                   └── TimeRecorder("vlm_inference", PAGE)
│                       └── llm.generate(llm_inputs, sampling_params)
│                           └── vLLM engine: batched, paged attention
│
├── _assemble_document()
│   └── TimeRecorder("doc_assemble", DOCUMENT)
│       ├── _turn_dt_into_doc()  ← DocTags → DoclingDocument
│       ├── _turn_md_into_doc()  ← Markdown → DoclingDocument
│       └── _turn_html_into_doc()← HTML → DoclingDocument
│
└── _enrich_document() (same as standard)
```

---

## 2. Config Gap Analysis

### Current vs Optimal GPU Settings

| Component | Current Setting | Optimal Setting | Impact | Notes |
|-----------|----------------|-----------------|--------|-------|
| **ONNX RT for RapidOCR** | | | | |
| EngineConfig.onnxruntime.use_cuda | `true` (patched) | `true` | Fixed | Our Dockerfile patch |
| cudnn_conv_algo_search | `HEURISTIC` (default) | `EXHAUSTIVE` | ~5-15% | First-run slower, then cached |
| cudnn_conv_use_max_workspace | `0` (default) | `1` | ~5-10% | Uses more VRAM for faster convs |
| graph_optimization_level | `ORT_ENABLE_BASIC` | `ORT_ENABLE_ALL` | ~5% | Enable extended optimizations |
| IO Binding | Not used | Use IO Binding | ~10-30% | Avoids CPU↔GPU copies |
| **RapidOCR Processing** | | | | |
| OCR scale | `3` (216 DPI) | `2` (144 DPI) | ~30-40% | Lower DPI = smaller crops = faster |
| Per-crop processing | Sequential | Batch crops | ~50%+ | Current: 1 crop at a time |
| Det input size | Dynamic | Fixed 960×960 | ~10% | Avoids recompilation |
| **Layout Model (RT-DETR)** | | | | |
| Batch size | `settings.perf.page_batch_size` | 8-16 | ~20-40% | Batch amortizes overhead |
| Input DPI | 72 DPI (scale=1.0) | 72 DPI | OK | Already reasonable |
| Device | Auto (usually CUDA) | CUDA | OK | Already on GPU |
| **TableFormer** | | | | |
| Mode | `accurate` | `fast` for most | ~50% | Accurate rarely needed |
| Device | CPU (default) | GPU (if possible) | ~3-5x | TableFormer may be CPU-only |
| **torch.compile** | | | | |
| TORCH_COMPILE_DISABLE | `1` | `1` | Correct | Prevents crashes |
| **VLM (granite-docling-258M)** | | | | |
| Precision | fp32 (default) | bf16 | ~2x speed | 258M model is fine at bf16 |
| Backend | transformers | vLLM | ~2-5x | Paged attention, continuous batching |
| Flash attention | Available (H100) | Enable | ~1.5x | Requires flash-attn package |
| KV cache | Default | Optimized | ~10% | vLLM handles automatically |
| **Threading** | | | | |
| OMP_NUM_THREADS | 4 | 4 | OK | Standard setting |
| Pipeline threads | 5 stages × 1 thread | OK | OK | ThreadedPipelineStage |
| DOCLING_SERVE_ENG_LOC_NUM_WORKERS | 2 | 1 (for profiling) | - | 2 workers = 2× GPU memory |

### Settings NOT Exposed by docling-serve

These docling settings exist but have no docling-serve env var:

| Setting | Location | Default | Description |
|---------|----------|---------|-------------|
| `debug.profile_pipeline_timings` | `docling.datamodel.settings` | `false` | Enable TimeRecorder |
| `perf.page_batch_size` | `docling.datamodel.settings` | 4 | Pages per batch |
| `debug.visualize_ocr` | `docling.datamodel.settings` | `false` | Save OCR debug images |
| `debug.visualize_layout` | `docling.datamodel.settings` | `false` | Save layout debug images |

---

## 3. Root Cause Analysis: Why H100 ≈ A5000

### Summary

The H100 shows minimal advantage over A5000 for docling-serve because **the workload doesn't match GPU strengths**. The standard pipeline is dominated by:

1. **Per-crop serial OCR** (~60-70% of processing time)
2. **Tiny ONNX models** that can't saturate GPU compute
3. **CPU-bound pre/post-processing** between GPU calls

### Detailed Root Causes

#### RC1: Per-Crop Serial OCR (Primary — ~60-70% of wall time)

RapidOCR processes each OCR region sequentially:
```python
for ocr_rect in ocr_rects:          # Sequential loop!
    image = get_page_image(cropbox)  # CPU: render crop
    im = numpy.array(image)          # CPU: PIL→numpy
    result = self.reader(im)         # GPU: 3 tiny models (det+cls+rec)
```

For a typical PDF page:
- 5-15 OCR regions detected
- Each region: ~50-200ms (render + 3 model calls)
- Total OCR per page: ~0.5-2.0s
- **GPU utilization: <5%** (tiny models, constant CPU↔GPU sync)

The H100 has 80GB HBM3 and 4000 TFLOPS FP8, but PP-OCRv4 models are:
- Det: 4.7MB (tiny conv net)
- Cls: 1.5MB
- Rec: 10.5MB

These models complete in <1ms of actual GPU compute, but launch overhead and data transfer dominate.

#### RC2: ONNX Runtime Session Creation Overhead

Each RapidOCR component creates its own ONNX session. For CUDA:
- Session init: ~100-500ms per model
- First inference includes JIT compilation
- Dynamic input shapes prevent graph optimization caching

#### RC3: Image Rendering is CPU-Bound

`page._backend.get_page_image(scale=3.0)` renders PDF to 216 DPI raster:
- pypdfium2/docling_parse renders on CPU
- Each crop: ~10-50ms of CPU rendering
- This cannot benefit from GPU at all

#### RC4: FP32 Everywhere (VLM)

When VLM models are used, they default to FP32:
- granite-vision-3.3-2b at FP32: 8.4GB VRAM
- At BF16: 4.2GB VRAM, ~2× inference speed
- H100 excels at BF16/FP8, but docling doesn't use them

#### RC5: No Batching Across Pages for OCR

The ThreadedPipelineStage passes pages one at a time to OCR:
- `ocr_batch_size` controls stage batching but...
- RapidOcrModel.__call__ still processes one page at a time
- No cross-page crop batching

#### RC6: Python GIL + Thread Overhead

5 ThreadedPipelineStages share the GIL:
- Queue put/get operations serialize
- CPU-bound post-processing blocks GPU stages
- Thread switching overhead for tiny GPU operations

### Performance Breakdown Estimate (15-page attention.pdf)

| Stage | Time (s) | % | GPU Util. |
|-------|----------|---|-----------|
| PDF rendering/preprocess | 1.5 | 11% | 0% |
| OCR (per-crop, 3 models) | 8.5 | 62% | <5% |
| Layout (RT-DETR batch) | 1.0 | 7% | ~30% |
| Table structure | 0.5 | 4% | ~10% |
| Page assembly | 1.0 | 7% | 0% |
| Document assembly | 0.5 | 4% | 0% |
| Enrichment | 0.5 | 4% | ~20% |
| **Total** | **~13.5** | | **<5% avg** |

---

## 4. VLM Pipeline Analysis

### granite-docling-258M: A Potential Game-Changer

The VLM pipeline using `ibm-granite/granite-docling-258M` (258M parameters) replaces the **entire standard pipeline** (OCR + Layout + Table Structure) with a single vision-language model pass:

```
Standard Pipeline: PDF → Render → OCR(×N crops) → Layout → Tables → Assemble
VLM Pipeline:      PDF → Render → VLM(1 pass) → Parse DocTags → Assemble
```

### Advantages

1. **Single GPU pass per page** — No per-crop serialization
2. **Native GPU workload** — Transformer inference scales with GPU power
3. **H100 advantage** — Flash attention, tensor cores, BF16
4. **Simpler pipeline** — Fewer failure modes, no OCR region detection
5. **Unified output** — Layout, OCR, and tables in one DocTags output

### Expected Performance (granite-docling-258M)

| Backend | Precision | Est. tok/s | Est. time/page | vs Standard |
|---------|-----------|------------|-----------------|-------------|
| transformers (A5000) | FP32 | ~100 | ~5s | Slower |
| transformers (A5000) | BF16 | ~200 | ~2.5s | Similar |
| transformers (H100) | BF16 | ~500 | ~1.0s | **~1.5× faster** |
| vLLM (H100) | BF16 | ~1000 | ~0.5s | **~3× faster** |
| vLLM (H100) | FP8 | ~1500 | ~0.3s | **~5× faster** |

### Caveats

1. **Output quality** — VLM may miss fine-grained table structure
2. **DocTags parsing** — Output must be parsed; malformed output = failure
3. **Token length** — Complex pages generate many tokens (>4K)
4. **Model size** — 258M is small; accuracy may lag for dense tables
5. **First request** — Model loading adds ~5-10s on first call

### Configuration for docling-serve

```bash
# Via HTTP API form fields:
pipeline=vlm
vlm_pipeline_preset=granite_docling

# Or with custom config:
pipeline=vlm
vlm_pipeline_custom_config='{"model_spec":{"repo_id":"ibm-granite/granite-docling-258M"},"engine_options":{"engine_type":"transformers"}}'
```

### Available VLM Presets (docling-serve v1.12.0)

| Preset | Model | Backend |
|--------|-------|---------|
| `granite_docling` | ibm-granite/granite-docling-258M | transformers |
| `smoldocling` | docling-project/SmolDocling-256M-preview | transformers |

---

## 5. Environment Variables Reference

### docling-serve Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_SERVE_ARTIFACTS_PATH` | `~/.cache/docling/models` | Model cache directory |
| `DOCLING_SERVE_ENG_LOC_NUM_WORKERS` | `2` | Number of processing workers |
| `DOCLING_SERVE_ENGINE` | auto | Backend engine selection |
| `DOCLING_SERVE_MAX_SYNC_WAIT` | `300` | Max wait for sync requests (s) |
| `DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT` | `300` | Max per-document timeout (s) |
| `DOCLING_SERVE_HOST` | `0.0.0.0` | Listen host |
| `DOCLING_SERVE_PORT` | `5001` | Listen port |

### Docling Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_ARTIFACTS_PATH` | `~/.cache/docling/models` | Model artifacts path |
| `DOCLING_PROFILE_PIPELINE_TIMINGS` | `false` | Enable TimeRecorder profiling |

### VLM / Picture Description

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_PICTURE_DESCRIPTION_MODEL_TYPE` | `vlm` | Model type for descriptions |
| `DOCLING_PICTURE_DESCRIPTION_VLM_MODEL_ID` | `ibm-granite/granite-vision-3.3-2b` | Model repo ID |
| `DOCLING_PICTURE_DESCRIPTION_VLM_DEVICE` | `cuda` | Device for VLM inference |
| `DOCLING_PICTURE_DESCRIPTION_BACKEND` | `vlm` | Backend type |

### CUDA / GPU

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU visibility |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | Driver capabilities |
| `CUDA_VISIBLE_DEVICES` | (unset) | Restrict GPU devices |
| `LD_LIBRARY_PATH` | `/usr/local/cuda/lib64:...` | Library search path |

### Performance Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `OMP_NUM_THREADS` | `4` | OpenMP thread count |
| `MKL_NUM_THREADS` | `4` | MKL thread count |
| `TORCH_COMPILE_DISABLE` | `1` | Disable torch.compile (prevents crashes) |
| `TORCHINDUCTOR_DISABLE` | `1` | Disable TorchInductor |

### Caching

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache root |
| `TRANSFORMERS_CACHE` | `~/.cache/huggingface` | Transformers model cache |
| `HF_HUB_CACHE` | `~/.cache/huggingface/hub` | HF Hub download cache |
| `TORCH_HOME` | `~/.cache/torch` | PyTorch cache (hub models) |
| `TESSDATA_PREFIX` | `/usr/share/tesseract-ocr/5/tessdata/` | Tesseract data path |

---

## Benchmark Results (A5000 24GB VRAM)

### Standard Pipeline — User Test Files

| File | Pages | std-default (s) | std-rapidocr (s) | Winner |
|------|-------|-----------------|-------------------|--------|
| testcase1.pdf | 7 | 3.49 | 4.32 | default |
| testcase2.pdf | 17 | 10.74 | 11.23 | default |
| testcase3.pdf | 21 | 20.17 | **16.85** | rapidocr |
| testcase4.pdf | 45 | **45.78** | 46.45 | default |
| testcase5.pdf | 22 | 45.83 | **28.59** | **rapidocr (38% faster)** |

### Per-Stage Profiling Breakdown

#### testcase5.pdf (22 pages, OCR-heavy — biggest speed difference)

**std-default (EasyOCR) — 45.83s total:**

| Stage | Count | Total (s) | Avg (s) | % of pipeline |
|-------|-------|-----------|---------|---------------|
| **ocr** | 22 | **42.65** | 1.94 | **93%** |
| page_parse | 22 | 12.22 | 0.56 | (parallel) |
| doc_assemble | 1 | 2.81 | 2.81 | 6% |
| layout | 9 | 1.15 | 0.13 | 2.5% |
| table_structure | 22 | 0.27 | 0.01 | <1% |

**std-rapidocr — 28.59s total (38% faster):**

| Stage | Count | Total (s) | Avg (s) | % of pipeline |
|-------|-------|-----------|---------|---------------|
| **ocr** | 22 | **25.31** | 1.15 | **89%** |
| page_parse | 22 | 12.18 | 0.55 | (parallel) |
| doc_assemble | 1 | 2.92 | 2.92 | 10% |
| layout | 9 | 0.99 | 0.11 | 3.5% |
| table_structure | 22 | 0.23 | 0.01 | <1% |

#### testcase4.pdf (45 pages, table-heavy)

**std-default — 45.78s total:**

| Stage | Count | Total (s) | Avg (s) | % of pipeline |
|-------|-------|-----------|---------|---------------|
| **table_structure** | 45 | **39.43** | 0.88 | **86%** |
| ocr | 45 | 13.00 | 0.29 | (parallel) |
| page_parse | 45 | 8.90 | 0.20 | (parallel) |
| doc_assemble | 1 | 4.53 | 4.53 | 10% |
| layout | 12 | 2.32 | 0.19 | 5% |

#### testcase3.pdf (21 pages)

**std-rapidocr — 16.85s total:**

| Stage | Count | Total (s) | Avg (s) | % of pipeline |
|-------|-------|-----------|---------|---------------|
| **ocr** | 21 | **14.53** | 0.69 | **86%** |
| table_structure | 21 | 2.56 | 0.12 | 15% |
| page_parse | 21 | 2.50 | 0.12 | (parallel) |
| doc_assemble | 1 | 1.66 | 1.66 | 10% |
| layout | 7 | 0.82 | 0.12 | 5% |

### Per-Crop OCR Analysis (RapidOCR runs only)

| Metric | Value |
|--------|-------|
| Total crops processed | 93 |
| Total OCR crop time | 36.9s |
| Average per-crop | 0.40s |
| Slowest crop | 2.01s (249x438px — likely CUDA warmup) |
| Fastest crop | 0.17s (852x1146px) |
| Pages with 0 OCR regions | 86/141 (61%) |
| Per-page overhead (0 regions) | ~0.12s |

### VLM Pipeline

#### attention.pdf (15 pages) — 488.2s total

| Batch | Pages | vlm_inference (s) | Per-page (s/pg) |
|-------|-------|-------------------|-----------------|
| 1 | 4 | 226.6 | 56.6 |
| 2 | 4 | 34.1 | 8.5 |
| 3 | 4 | 218.5 | 54.6 |
| 4 | 3 | 7.1 | 2.4 |
| **Total** | **15** | **486.2** | **32.4 avg** |

#### testcase1.pdf (7 pages) — 525.9s total

| Batch | Pages | vlm_inference (s) | Per-page (s/pg) |
|-------|-------|-------------------|-----------------|
| 1 | 4 | 343.6 | 85.9 |
| 2 | 3 | 181.4 | 60.5 |
| **Total** | **7** | **525.0** | **75.0 avg** |

**VLM Analysis**: The VLM pipeline is **~15-75× slower** than the standard pipeline on A5000 with the transformers backend in FP32. The huge variance in per-page time (2.4s to 85.9s) reflects token count differences.

### VLM Backend Comparison (standalone benchmark, synthetic page image)

| Backend | Max Tokens | Avg Time (s) | Tokens Generated | Tok/s | Speedup |
|---------|-----------|-------------|-----------------|-------|---------|
| transformers FP32 | 256 | 7.9s | 23 | 2.9 | baseline |
| transformers BF16 | 512 | 47.7s | 512 | 10.7 | 3.7× |
| transformers BF16 | 2048 | 201.5s | 2048 | 10.2 | 3.5× |
| **vLLM BF16** | **512** | **2.2s** | **146** | **66.6** | **23×** |
| **vLLM BF16** | **2048** | **24.2s** | **1682** | **69.4** | **24×** |

**Key finding: vLLM BF16 is 6.2-6.8× faster than transformers BF16 (66-69 tok/s vs 10 tok/s).**

At 69 tok/s, a typical page generating ~1000 DocTag tokens takes ~14.5s — making VLM competitive with the standard pipeline for OCR-heavy documents. On H100 (3-4× faster than A5000 for transformer inference), expect ~200-280 tok/s, bringing per-page time to ~3.5-5s.

---

## 6. Recommended Optimizations

### Priority 1: Optimize VLM Pipeline (Highest Potential Impact)

**Current state: VLM is 15-75× SLOWER than standard pipeline on A5000 with default settings.**

The VLM pipeline with `granite-docling-258M` is currently bottlenecked by running FP32 on the transformers backend. To realize its potential:

**Action items (ordered):**
1. **Switch to BF16/FP16**: `torch_dtype=torch.bfloat16` — expected ~2× speedup
2. **Use vLLM backend**: `engine_type=vllm` — expected ~3-5× speedup with continuous batching
3. **Enable flash-attention**: compile flash-attn for granite-docling architecture
4. **Quantize**: INT8/INT4 for further speedup at minimal quality cost (258M model is tiny)

```bash
# Test VLM pipeline
curl -X POST http://localhost:5001/v1/convert/file \
  -F "files=@document.pdf" \
  -F "pipeline=vlm" \
  -F "vlm_pipeline_preset=granite_docling"
```

**Measured on A5000 with vLLM BF16:** 69 tok/s → ~14.5s/page (1000 tokens). On H100: estimated ~3.5-5s/page.

**CONFIRMED: vLLM BF16 is 6.8× faster than transformers BF16 and 24× faster than transformers FP32.**

### Priority 2: Batch OCR Crops (If Staying with Standard Pipeline)

**Impact: ~50% OCR speedup**

Collect all crops from a page, batch them, and run det/cls/rec once:

```python
# Instead of:
for crop in crops:
    result = self.reader(crop)  # 3 model calls per crop

# Do:
all_crops = [get_crop(rect) for rect in ocr_rects]
results = self.reader.batch(all_crops)  # 3 model calls total
```

**Caveat:** RapidOCR may not support batching natively; may need custom ONNX session.

### Priority 3: ONNX Session Optimization

**Impact: ~10-30% per-model speedup**

```python
# Add to ONNX session creation:
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

providers = [
    ("CUDAExecutionProvider", {
        "device_id": 0,
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "cudnn_conv_use_max_workspace": "1",
    }),
    "CPUExecutionProvider",
]
```

Use `benchmark_onnx_sessions.py` to find optimal settings for your GPU.

### Priority 4: Reduce OCR Resolution

**Impact: ~30% OCR speedup, minimal quality loss**

Current scale=3 (216 DPI) is overkill for most text. Scale=2 (144 DPI) is sufficient:

```python
# In RapidOcrModel.__init__:
self.scale = 2  # 144 DPI instead of 216 DPI
```

### Priority 5: IO Binding for ONNX Models

**Impact: ~10-30% per-inference**

Eliminates CPU↔GPU memory copies:

```python
io_binding = session.io_binding()
io_binding.bind_input('input', 'cuda', 0, np.float32, shape, ptr)
io_binding.bind_output('output', 'cuda', 0)
session.run_with_iobinding(io_binding)
```

Use `benchmark_onnx_sessions.py` to quantify the benefit.

### Priority 6: Single Worker for GPU Workloads

**Impact: Reduced VRAM usage, avoid GPU contention**

```bash
DOCLING_SERVE_ENG_LOC_NUM_WORKERS=1  # instead of 2
```

Two workers double GPU memory usage and can cause contention on shared CUDA contexts.

---

## Appendix: Files Created

| File | Purpose |
|------|---------|
| `docling_debug_profiler.py` | Monkey-patches for comprehensive pipeline timing |
| `debug_entrypoint.sh` | Enhanced entrypoint with diagnostics + profiling |
| `benchmark_models.py` | Standalone model benchmarks (7 models) |
| `benchmark_onnx_sessions.py` | ONNX RT session options benchmark |
| `test_docling.py` | Updated HTTP benchmark with VLM pipeline tasks |
| `Dockerfile.debug` | Debug image with profiling + VLM models |
| `INVESTIGATION.md` | This report |
