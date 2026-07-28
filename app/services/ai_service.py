import logging
import os
import asyncio
import cv2
import httpx
import numpy as np
import replicate
from ..config import get_settings

settings = get_settings()
logger = logging.getLogger("e2m.ai")

# Normalization mapping for DINO labels
LABEL_MAP = {
    "glass": "window",
    "glass window": "window",
    "garage": "door",
    "garage door": "door",
    "column": "pillar",
    "railing": "balcony",
    "terrace": "parapet",
    "chimney": "roof"
}


def _classify_region_by_rules(label_raw, rel_y, bw, bh, w, h):
    """Fallback classifier based on geometry heuristics."""
    label_lower = (label_raw or "").lower()

    if "roof" in label_lower or "eave" in label_lower or "chimney" in label_lower:
        return "roof"
    if "window" in label_lower or "glass" in label_lower:
        return "window"
    if "balcony" in label_lower or "railing" in label_lower:
        return "balcony"
    if "pillar" in label_lower or "column" in label_lower:
        return "pillar"
    if "parapet" in label_lower or "terrace" in label_lower or "fence" in label_lower:
        return "parapet"
    if "gate" in label_lower or "door" in label_lower:
        return "door"

    if rel_y < 0.15:
        return "roof"
    if bw < w * 0.12 and bh > h * 0.15:
        return "pillar"
        
    return "wall"


def _compute_bbox_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def _apply_nms(regions, iou_threshold=0.45):
    """Applies Non-Maximum Suppression to remove overlapping duplicate masks."""
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda r: r["confidence"], reverse=True)
    keep = []

    while sorted_regions:
        current = sorted_regions.pop(0)
        keep.append(current)
        sorted_regions = [
            r for r in sorted_regions
            if _compute_bbox_iou(current["bbox"], r["bbox"]) < iou_threshold
        ]

    return keep


async def _fetch_mask_and_extract_polygon(http_client, mask_url, min_area, img_w, img_h):
    """Downloads an individual SAM 2 PNG mask and extracts polygon contour coordinates."""
    try:
        resp = await http_client.get(mask_url, timeout=10.0)
        if resp.status_code != 200:
            return None

        mask_bytes = np.frombuffer(resp.content, np.uint8)
        mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            return None

        _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Take largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < min_area:
            return None

        # Contour approximation for smooth polygon rendering
        epsilon = 0.004 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx_contour) < 3:
            return None

        polygon = [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in approx_contour]
        x, y, bw, bh = cv2.boundingRect(approx_contour)

        # Skip full-frame image masks
        if bw > 0.96 * img_w and bh > 0.96 * img_h:
            return None

        return {
            "polygon": polygon,
            "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
            "area": round(float(area), 1)
        }
    except Exception as e:
        logger.warning(f"Failed to process SAM 2 mask from {mask_url}: {e}")
        return None


async def detect_regions(image_url):
    """
    Ultra-Fast Pipeline:
    1. Grounding DINO -> Bounding Boxes + Labels (1 call)
    2. SAM 2 -> Whole-Image Automatic Instance Masks (1 call)
    3. Concurrent PNG fetching & Polygon Contour Extraction
    4. IoU Label Transfer + NMS Deduplication
    """
    if not settings.replicate_api_token or settings.replicate_api_token == "your-replicate-api-token":
        return {"success": False, "error": "Replicate API token not configured.", "regions": []}

    uploads_dir = os.path.abspath(settings.upload_dir)
    if image_url.startswith("/uploads/"):
        local_path = os.path.join(uploads_dir, image_url[len("/uploads/"):])
    elif image_url.startswith("uploads/"):
        local_path = os.path.abspath(image_url)
    else:
        local_path = image_url

    if not os.path.exists(local_path):
        return {"success": False, "error": f"Image file not found at: {local_path}", "regions": []}

    img = cv2.imread(local_path)
    if img is None:
        return {"success": False, "error": "Failed to read image with OpenCV", "regions": []}

    img_h, img_w = img.shape[:2]
    min_area = (img_w * img_h) * 0.0002

    client = replicate.Client(api_token=settings.replicate_api_token)

    dino_query = ",".join([
        "roof", "wall", "window", "glass window", "door", 
        "garage door", "balcony", "railing", "column", 
        "pillar", "chimney", "gate", "fence", "parapet"
    ])

    # -------------------------------------------------------------
    # Step 1: Run Grounding DINO and SAM 2 in Parallel
    # -------------------------------------------------------------
    logger.info("Triggering Grounding DINO and SAM 2 concurrently...")

    async def _run_dino():
        with open(local_path, "rb") as f:
            return await asyncio.to_thread(
                client.run,
                "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",
                input={
                    "image": f,
                    "query": dino_query,
                    "box_threshold": 0.40,
                    "text_threshold": 0.25,
                    "show_visualisation": False
                }
            )

    async def _run_sam2():
        with open(local_path, "rb") as f:
            return await asyncio.to_thread(
                client.run,
                "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
                input={
                    "image": f,
                    "points_per_side": 32,
                    "pred_iou_thresh": 0.80,
                    "stability_score_thresh": 0.85,
                    "use_m2m": True
                }
            )

    dino_res, sam2_res = await asyncio.gather(_run_dino(), _run_sam2(), return_exceptions=True)

    # Validate Grounding DINO outputs
    dino_boxes = []
    if isinstance(dino_res, dict) and "detections" in dino_res:
        for det in dino_res.get("detections", []):
            box = det.get("bbox") or det.get("box")
            if not box:
                continue
            if isinstance(box, dict):
                b = [box.get("xmin", 0), box.get("ymin", 0), box.get("xmax", 0), box.get("ymax", 0)]
            else:
                b = list(map(float, box))
                if b[2] < b[0] or b[3] < b[1]:
                    b[2] += b[0]
                    b[3] += b[1]
            dino_boxes.append({
                "label": det.get("label", "wall").lower(),
                "confidence": float(det.get("confidence", det.get("score", 0.80))),
                "bbox": b
            })

    # Validate SAM 2 outputs
    individual_mask_urls = []
    if isinstance(sam2_res, dict) and "individual_masks" in sam2_res:
        individual_mask_urls = sam2_res.get("individual_masks", [])

    if not individual_mask_urls:
        logger.warning("SAM 2 produced no individual masks.")
        return {"success": True, "regions": [], "method": "dino-sam2-full-image"}

    logger.info(f"SAM 2 generated {len(individual_mask_urls)} masks. Extracting polygons...")

    # -------------------------------------------------------------
    # Step 2: Download Mask PNGs and Extract Polygons Concurrently
    # -------------------------------------------------------------
    async with httpx.AsyncClient() as http_client:
        parse_tasks = [
            _fetch_mask_and_extract_polygon(http_client, mask_url, min_area, img_w, img_h)
            for mask_url in individual_mask_urls
        ]
        extracted_masks = await asyncio.gather(*parse_tasks)

    # -------------------------------------------------------------
    # Step 3: Match SAM 2 Masks to DINO Bounding Boxes (IoU Transfer)
    # -------------------------------------------------------------
    raw_regions = []
    for mask_data in extracted_masks:
        if not mask_data:
            continue

        sam_box = mask_data["bbox"]
        best_dino = None
        best_iou = 0.0

        for dino in dino_boxes:
            iou = _compute_bbox_iou(sam_box, dino["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_dino = dino

        if best_dino and best_iou >= 0.20:
            raw_label = best_dino["label"]
            norm_label = LABEL_MAP.get(raw_label, raw_label)
            conf = round((0.4 * best_dino["confidence"]) + 0.60, 3)
        else:
            rel_y = float(sam_box[1]) / float(img_h)
            bw = sam_box[2] - sam_box[0]
            bh = sam_box[3] - sam_box[1]
            norm_label = _classify_region_by_rules("", rel_y, bw, bh, img_w, img_h)
            conf = 0.70

        raw_regions.append({
            "type": norm_label,
            "polygon": mask_data["polygon"],
            "bbox": sam_box,
            "area": mask_data["area"],
            "confidence": conf,
            "raw_label": norm_label
        })

    # -------------------------------------------------------------
    # Step 4: Non-Maximum Suppression and Final Formatting
    # -------------------------------------------------------------
    deduped_regions = _apply_nms(raw_regions, iou_threshold=0.45)
    deduped_regions.sort(key=lambda r: r["area"], reverse=True)

    counts = {}
    for r in deduped_regions:
        t = r["type"]
        counts[t] = counts.get(t, 0) + 1
        r["id"] = f"{t}_{counts[t]}"
        r["label"] = f"{t.replace('_', ' ').title()} {counts[t]}"

    logger.info(f"Pipeline complete! Extracted {len(deduped_regions)} high-precision architectural regions.")

    return {
        "success": True,
        "regions": deduped_regions,
        "method": "dino-sam2-full-image-iou"
    }
