#!/usr/bin/env python3
"""Build-time patch: Replace per-crop OCR with batched cls+rec across all crops on a page.

Instead of calling RapidOCR once per OCR region (3 ONNX calls each: det→cls→rec),
this batches all detected sub-crops from all regions and runs cls+rec just once.

Reduces GPU calls from 3N to N+2 (N det + 1 cls + 1 rec), cutting ~60% of
ONNX GPU context switches for multi-crop pages.

Applied at Docker build time after Steps 4-6 patches.
"""
from pathlib import Path

RAPID_OCR_MODEL = Path(
    "/opt/app-root/lib/python3.12/site-packages/"
    "docling/models/stages/ocr/rapid_ocr_model.py"
)

src = RAPID_OCR_MODEL.read_text()

# ── 1. Add imports ──────────────────────────────────────────────────────────
import_marker = "from docling.utils.utils import download_url_with_progress"
new_imports = """\
from docling.utils.utils import download_url_with_progress
from rapidocr.ch_ppocr_rec import TextRecInput as _BatchRecInput
from rapidocr.utils.process_img import map_boxes_to_original as _map_boxes_orig"""

assert import_marker in src, "FATAL: import marker not found"
src = src.replace(import_marker, new_imports)

# ── 2. Add _batch_ocr_crops method before __call__ ─────────────────────────
BATCH_METHOD = '''\
    def _batch_ocr_crops(self, page, ocr_rects):
        """Batch cls+rec across all OCR crops on a page for better GPU utilization."""
        reader = self.reader
        reader.update_params(
            use_det=self.options.use_det,
            use_cls=self.options.use_cls,
            use_rec=self.options.use_rec,
        )

        # Phase 1: Render each crop and run det (sequential — different sizes)
        crop_infos = []
        all_sub_crops = []

        for ocr_rect in ocr_rects:
            if ocr_rect.area() == 0:
                continue

            high_res_image = page._backend.get_page_image(
                scale=self.scale, cropbox=ocr_rect
            )
            im = numpy.array(high_res_image)
            del high_res_image

            ori_img = reader.load_img(im)
            img, op_record = reader.preprocess_img(ori_img)

            det_boxes = None
            sub_crops = []

            if reader.use_det:
                try:
                    sub_crops, det_res = reader.detect_and_crop(img, op_record)
                    if det_res.boxes is not None:
                        ori_h, ori_w = ori_img.shape[:2]
                        det_boxes = _map_boxes_orig(
                            det_res.boxes.copy(), op_record, ori_h, ori_w
                        )
                except Exception:
                    del im
                    continue
            else:
                sub_crops = [img]

            if not sub_crops:
                del im
                continue

            crop_infos.append({
                "ocr_rect": ocr_rect,
                "det_boxes": det_boxes,
                "sub_start": len(all_sub_crops),
                "sub_count": len(sub_crops),
            })
            all_sub_crops.extend(sub_crops)
            del im

        if not all_sub_crops:
            return []

        # Phase 2: Batch cls on ALL sub-crops at once
        if reader.use_cls:
            try:
                cls_res = reader.text_cls(all_sub_crops)
                all_sub_crops = cls_res.img_list
            except Exception:
                pass  # keep unrotated images as fallback

        # Phase 3: Batch rec on ALL sub-crops at once
        rec_txts = [""] * len(all_sub_crops)
        rec_scores = [0.0] * len(all_sub_crops)

        if reader.use_rec:
            try:
                rec_input = _BatchRecInput(img=all_sub_crops, return_word_box=False)
                rec_res = reader.text_rec(rec_input)
                if rec_res.txts is not None:
                    rec_txts = list(rec_res.txts)
                    rec_scores = list(rec_res.scores)
            except Exception as e:
                _log.warning("[OCR_BATCH] rec failed: %s", e)

        # Phase 4: Reconstruct TextCells per-crop
        all_ocr_cells = []
        cell_idx = 0

        for info in crop_infos:
            ocr_rect = info["ocr_rect"]
            det_boxes = info["det_boxes"]
            start = info["sub_start"]
            count = info["sub_count"]

            crop_txts = rec_txts[start : start + count]
            crop_scores = rec_scores[start : start + count]

            if det_boxes is not None:
                boxes_list = det_boxes.tolist()
            else:
                # No det: use zero-area placeholder (coordinates don't matter)
                boxes_list = [[[0, 0], [0, 0], [0, 0], [0, 0]]] * count

            for box, txt, score in zip(boxes_list, crop_txts, crop_scores):
                if not txt or not txt.strip():
                    continue
                all_ocr_cells.append(
                    TextCell(
                        index=cell_idx,
                        text=txt,
                        orig=txt,
                        confidence=score,
                        from_ocr=True,
                        rect=BoundingRectangle.from_bounding_box(
                            BoundingBox.from_tuple(
                                coord=(
                                    (box[0][0] / self.scale) + ocr_rect.l,
                                    (box[0][1] / self.scale) + ocr_rect.t,
                                    (box[2][0] / self.scale) + ocr_rect.l,
                                    (box[2][1] / self.scale) + ocr_rect.t,
                                ),
                                origin=CoordOrigin.TOPLEFT,
                            )
                        ),
                    )
                )
                cell_idx += 1

        return all_ocr_cells

'''

# Insert before __call__
call_def = "    def __call__("
assert call_def in src, "FATAL: __call__ definition not found"
src = src.replace(call_def, BATCH_METHOD + call_def, 1)

# ── 3. Replace per-crop loop with batched call ─────────────────────────────
# Target: the for-loop and post_process_cells inside the with-TimeRecorder block
OLD_LOOP = (
    "                    all_ocr_cells = []\n"
    "                    for ocr_rect in ocr_rects:\n"
    "                        # Skip zero area boxes\n"
    "                        if ocr_rect.area() == 0:\n"
    "                            continue\n"
    "                        high_res_image = page._backend.get_page_image(\n"
    "                            scale=self.scale, cropbox=ocr_rect\n"
    "                        )\n"
    "                        im = numpy.array(high_res_image)\n"
    "                        result = self.reader(\n"
    "                            im,\n"
    "                            use_det=self.options.use_det,\n"
    "                            use_cls=self.options.use_cls,\n"
    "                            use_rec=self.options.use_rec,\n"
    "                        )\n"
    "                        if result is None or result.boxes is None:\n"
    '                            _log.warning("RapidOCR returned empty result!")\n'
    "                            continue\n"
    "                        result = list(\n"
    "                            zip(result.boxes.tolist(), result.txts, result.scores)\n"
    "                        )\n"
    "\n"
    "                        del high_res_image\n"
    "                        del im\n"
    "\n"
    "                        if result is not None:\n"
    "                            cells = [\n"
    "                                TextCell(\n"
    "                                    index=ix,\n"
    "                                    text=line[1],\n"
    "                                    orig=line[1],\n"
    "                                    confidence=line[2],\n"
    "                                    from_ocr=True,\n"
    "                                    rect=BoundingRectangle.from_bounding_box(\n"
    "                                        BoundingBox.from_tuple(\n"
    "                                            coord=(\n"
    "                                                (line[0][0][0] / self.scale)\n"
    "                                                + ocr_rect.l,\n"
    "                                                (line[0][0][1] / self.scale)\n"
    "                                                + ocr_rect.t,\n"
    "                                                (line[0][2][0] / self.scale)\n"
    "                                                + ocr_rect.l,\n"
    "                                                (line[0][2][1] / self.scale)\n"
    "                                                + ocr_rect.t,\n"
    "                                            ),\n"
    "                                            origin=CoordOrigin.TOPLEFT,\n"
    "                                        )\n"
    "                                    ),\n"
    "                                )\n"
    "                                for ix, line in enumerate(result)\n"
    "                            ]\n"
    "                            all_ocr_cells.extend(cells)\n"
    "\n"
    "                    # Post-process the cells\n"
    "                    self.post_process_cells(all_ocr_cells, page)"
)

NEW_LOOP = (
    "                    all_ocr_cells = self._batch_ocr_crops(page, ocr_rects)\n"
    "\n"
    "                    # Post-process the cells\n"
    "                    self.post_process_cells(all_ocr_cells, page)"
)

assert OLD_LOOP in src, "FATAL: per-crop OCR loop not found (was Step 5 applied first?)"
src = src.replace(OLD_LOOP, NEW_LOOP, 1)

# ── Write back ──────────────────────────────────────────────────────────────
RAPID_OCR_MODEL.write_text(src)
print("[BUILD] OCR page-level batch patch applied successfully", flush=True)
