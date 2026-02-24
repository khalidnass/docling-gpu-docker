#!/usr/bin/env python3
"""Focused benchmark: granite-docling-258M — transformers BF16 vs vLLM BF16."""

import gc
import sys
import time

import numpy as np
import torch
from PIL import Image

MODEL_PATH = "/opt/app-root/src/.cache/docling/models/ibm-granite--granite-docling-258M"

def make_test_image():
    """Realistic doc page."""
    img = np.ones((1400, 1000, 3), dtype=np.uint8) * 250
    rng = np.random.RandomState(42)
    img[60:66, 100:600] = 30
    img[70:74, 100:500] = 30
    for y in range(120, 1200, 22):
        img[y:y+2, 100:100+rng.randint(500,850)] = 50
    for y in range(600, 800, 25):
        img[y:y+1, 200:800] = 80
    for x in range(200, 801, 120):
        img[600:800, x:x+1] = 80
    return Image.fromarray(img)


def bench_transformers_bf16(image, max_tokens=512, runs=3):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"\n{'='*60}")
    print(f"Transformers BF16 — max_tokens={max_tokens}")
    print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    print(f"  Load: {time.perf_counter()-t0:.2f}s")

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Convert this page to docling."}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=32)
    torch.cuda.synchronize()

    results = []
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ntok = out.shape[1] - inputs["input_ids"].shape[1]
        tps = ntok / elapsed
        results.append((elapsed, ntok, tps))
        print(f"  Run {i+1}: {elapsed:.2f}s, {ntok} tok, {tps:.1f} tok/s")
        del out

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VRAM: {vram:.2f} GB")

    del model, processor, inputs
    gc.collect(); torch.cuda.empty_cache()

    avg_time = sum(r[0] for r in results) / len(results)
    avg_tok = sum(r[1] for r in results) / len(results)
    avg_tps = sum(r[2] for r in results) / len(results)
    return {"backend": "transformers-bf16", "avg_s": avg_time, "avg_tok": avg_tok, "tps": avg_tps, "vram": vram}


def bench_vllm_bf16(image, max_tokens=512, runs=3):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("\nvLLM not available, skipping")
        return None

    print(f"\n{'='*60}")
    print(f"vLLM BF16 — max_tokens={max_tokens}")
    print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=max_tokens + 512,  # input + output
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    )
    print(f"  Load: {time.perf_counter()-t0:.2f}s")

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    # Build prompt matching the chat template
    prompt = "<|start_of_role|>user<|end_of_role|><image>Convert this page to docling.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
    req = {"prompt": prompt, "multi_modal_data": {"image": image}}

    # Warmup
    llm.generate([req], SamplingParams(max_tokens=32, temperature=0.0))

    results = []
    for i in range(runs):
        t0 = time.perf_counter()
        out = llm.generate([req], sampling_params)
        elapsed = time.perf_counter() - t0
        ntok = len(out[0].outputs[0].token_ids)
        tps = ntok / elapsed
        results.append((elapsed, ntok, tps))
        print(f"  Run {i+1}: {elapsed:.2f}s, {ntok} tok, {tps:.1f} tok/s")
        del out

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VRAM: {vram:.2f} GB")

    del llm
    gc.collect(); torch.cuda.empty_cache()

    avg_time = sum(r[0] for r in results) / len(results)
    avg_tok = sum(r[1] for r in results) / len(results)
    avg_tps = sum(r[2] for r in results) / len(results)
    return {"backend": "vllm-bf16", "avg_s": avg_time, "avg_tok": avg_tok, "tps": avg_tps, "vram": vram}


def main():
    print("=" * 60)
    print("granite-docling-258M — BF16 Backend Comparison")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    image = make_test_image()
    all_results = []

    # Test with 512 tokens (representative of a simple page)
    for max_tok in [512, 2048]:
        print(f"\n\n{'#'*60}")
        print(f"# max_tokens = {max_tok}")
        print(f"{'#'*60}")

        r1 = bench_transformers_bf16(image, max_tokens=max_tok, runs=3)
        all_results.append((max_tok, r1))

        r2 = bench_vllm_bf16(image, max_tokens=max_tok, runs=3)
        if r2:
            all_results.append((max_tok, r2))

    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Backend':<22} {'MaxTok':<10} {'Avg(s)':<10} {'Tokens':<10} {'Tok/s':<10} {'VRAM(GB)':<10}")
    print("-" * 80)
    for max_tok, r in all_results:
        print(f"{r['backend']:<22} {max_tok:<10} {r['avg_s']:<10.2f} {r['avg_tok']:<10.0f} {r['tps']:<10.1f} {r['vram']:<10.2f}")
    print("-" * 80)

    # Speedup
    tf_results = [r for _, r in all_results if r["backend"] == "transformers-bf16"]
    vl_results = [r for _, r in all_results if r["backend"] == "vllm-bf16"]
    if tf_results and vl_results:
        print("\nvLLM speedup over transformers:")
        for tf, vl in zip(tf_results, vl_results):
            speedup = tf["tps"] / vl["tps"] if vl["tps"] > 0 else 0
            print(f"  tok/s ratio: {vl['tps']:.1f} / {tf['tps']:.1f} = {vl['tps']/tf['tps']:.2f}x")


if __name__ == "__main__":
    main()
