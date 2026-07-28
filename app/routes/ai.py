import logging
import os
import asyncio
import io
import cv2
import json
import re
import numpy as np
import replicate
from replicate.exceptions import ReplicateError
from ..config import get_settings

settings = get_settings()
logger = logging.getLogger("e2m.ai")

# Normalization mapping for DINO labels to unify classes
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

# Maximum parallel SAM 3 API calls sent to Replicate simultaneously
CONCURRENCY_LIMIT = getattr(settings, "replicate_concurrency_limit", 3)


def _classify_region_by_rules(label_raw, rel_y, bw, bh, w, h, img_crop=None):
    """Fallback classifier based on geometry and visual heuristics."""
    label_lower = (label_raw or "").lower()

    if "roof" in label_lower or "eave" in label_lower or "chimney" in label_lower:
        return "roof"
    if "window" in label_lower or "glass" in label_lower or "trim" in label_lower:
        return "window"
    if "balcony" in label_lower or "railing" in label_lower:
        return "balcony"
    if "pillar" in label_lower or "column" in label_lower:
        return "pillar"
    if "parapet" in label_lower or "terrace" in label_lower or "fence" in label_lower:
        return "parapet"
    if "gate" in label_lower or "door" in label_lower or "entrance" in label_lower:
        return "gate"
        
    if img_crop is not None and img_crop.size > 0:
        gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray_crop)
        if mean_val < 110 and (bw < 0.4 * w and bh < 0.4 * h):
            if bw * bh > 1000:
                return "window"

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


def _parse_box_coords(box_data):
    """Safely extracts [x1, y1, x2, y2] from various BBox schema formats."""
    if not box_data:
        return None
    try:
        if isinstance(box_data, dict):
            x1 = float(box_data.get("xmin", box_data.get("x1", 0)))
            y1 = float(box_data.get("ymin", box_data.get("y1", 0)))
            x2 = float(box_data.get("xmax", box_data.get("x2", 0)))
            y2 = float(box_data.get("ymax", box_data.get("y2", 0)))
        elif isinstance(box_data, list) and len(box_data) == 4:
            x1, y1, x2, y2 = map(float, box_data)
            if x2 < x1 or y2 < y1:  # Fallback for [x, y, w, h] format
                x2 = x1 + x2
                y2 = y1 + y2
        else:
            return None
        return [x1, y1, x2, y2] if (x2 > x1 and y2 > y1) else None
    except Exception:
        return None


def _apply_dino_nms(detections, iou_threshold=0.55):
    """Prunes duplicate/overlapping DINO bounding boxes BEFORE calling SAM 3."""
    if not detections:
        return []

    sorted_dets = sorted(
        detections, 
        key=lambda d: float(d.get("confidence", d.get("score", 0.85))), 
        reverse=True
    )
    keep = []

    while sorted_dets:
        current = sorted_dets.pop(0)
        curr_box = _parse_box_coords(current.get("bbox") or current.get("box"))
        if not curr_box:
            continue
            
        keep.append(current)

        filtered = []
        for det in sorted_dets:
            box = _parse_box_coords(det.get("bbox") or det.get("box"))
            if box and _compute_bbox_iou(curr_box, box) >= iou_threshold:
                continue  # Drop duplicate box
            filtered.append(det)

        sorted_dets = filtered

    return keep


async def _run_replicate_with_retry(client, model, input_data, max_retries=6):
    """
    Executes a Replicate model call with non-blocking async retries, parsing 429 
    rate limit reset times dynamically or applying exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(client.run, model, input=input_data)
        except Exception as e:
            err_msg = str(e)
            status_code = getattr(e, "status", None)

            # Check if rate limit error was thrown (HTTP 429 or error text)
            if status_code == 429 or "429" in err_msg or "rate limit" in err_msg.lower() or "resets in" in err_msg.lower():
                # Extract wait time from error message like "resets in ~9s" if present
                reset_match = re.search(r"resets in ~?(\d+)s", err_msg, re.IGNORECASE)
                if reset_match:
                    wait_time = int(reset_match.group(1)) + 1
                else:
                    wait_time = (2 ** attempt) + 1  # Exponential backoff fallback

                logger.warning(
                    f"Replicate 429 Rate Limit hit on attempt {attempt + 1}/{max_retries}. "
                    f"Waiting {wait_time}s before retrying..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Non-retryable Replicate error encountered: {e}")
                raise e

    raise RuntimeError(f"Exceeded maximum retries ({max_retries}) for model {model}")


def _generate_fg_bg_points(target_box_crop, crop_w, crop_h):
    """Generates 16 foreground grid points and 8 background padding points."""
    tx1, ty1, tx2, ty2 = target_box_crop
    tbw = tx2 - tx1
    tbh = ty2 - ty1

    points = []
    labels = []

    # 1. Foreground Points (4x4 Grid)
    fg_x = np.linspace(tx1 + tbw * 0.15, tx2 - tbw * 0.15, 4)
    fg_y = np.linspace(ty1 + tbh * 0.15, ty2 - tbh * 0.15, 4)
    for x in fg_x:
        for y in fg_y:
            points.append([int(x), int(y)])
            labels.append(1)

    # 2. Background Points
    bg_candidates = [
        [tx1 / 2.0, ty1 / 2.0],
        [tx1 + tbw / 2.0, ty1 / 2.0],
        [tx2 + (crop_w - tx2) / 2.0, ty1 / 2.0],
        [tx1 / 2.0, ty1 + tbh / 2.0],
        [tx2 + (crop_w - tx2) / 2.0, ty1 + tbh / 2.0],
        [tx1 / 2.0, ty2 + (crop_h - ty2) / 2.0],
        [tx1 + tbw / 2.0, ty2 + (crop_h - ty2) / 2.0],
        [tx2 + (crop_w - tx2) / 2.0, ty2 + (crop_h - ty2) / 2.0]
    ]

    for px, py in bg_candidates:
        if 0 <= px < crop_w and 0 <= py < crop_h:
            points.append([int(px), int(py)])
            labels.append(0)

    return points, labels


async def detect_regions(image_url):
    """
    Production-grade detection pipeline:
    1. Grounding DINO -> Raw bounding boxes
    2. Pre-SAM NMS -> Filter duplicate target boxes up front
    3. Concurrency-throttled SAM 3 execution with 429 retry backoff
    4. Polygon extraction, box validation, and output construction
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
        return {"success": False, "error": "Image file not found on server", "regions": []}

    img = cv2.imread(local_path)
    if img is None:
        return {"success": False, "error": "Failed to read image with OpenCV", "regions": []}
    
    h, w = img.shape[:2]
    total_image_area = w * h
    min_area = total_image_area * 0.0002

    client = replicate.Client(api_token=settings.replicate_api_token)

    dino_query = ",".join([
        "roof", "wall", "window", "glass window", "door", 
        "garage door", "balcony", "railing", "column", 
        "pillar", "chimney", "gate", "fence", "parapet"
    ])

    # 1. Grounding DINO Call with 429 Retry Protection
    try:
        logger.info("Executing Grounding DINO detection...")
        with open(local_path, "rb") as image_file:
            dino_output = await _run_replicate_with_retry(
                client,
                "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",
                input_data={
                    "image": image_file,
                    "query": dino_query,
                    "box_threshold": 0.25,
                    "text_threshold": 0.25,
                    "show_visualisation": False
                }
            )
    except Exception as e:
        logger.error(f"Grounding DINO Inference failed: {e}")
        return {"success": False, "error": f"Grounding DINO Error: {str(e)}", "regions": []}

    detections = dino_output.get("detections", []) if isinstance(dino_output, dict) else (dino_output if isinstance(dino_output, list) else [])
    
    if not detections:
        return {"success": True, "regions": [], "method": "grounding-dino-sam3-rate-limited"}

    # 2. Apply Pre-SAM NMS to cut down unnecessary API calls
    pruned_detections = _apply_dino_nms(detections, iou_threshold=0.55)
    logger.info(f"DINO found {len(detections)} raw boxes -> Pruned down to {len(pruned_detections)} distinct targets.")

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def _process_single_target(det, idx):
        raw_label = det.get("label", "wall").lower()
        norm_label = LABEL_MAP.get(raw_label, raw_label)
        dino_conf = float(det.get("confidence", det.get("score", 0.85)))

        box = _parse_box_coords(det.get("bbox") or det.get("box"))
        if not box:
            return None

        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1

        # Padded Crop (20% context)
        pad_x, pad_y = int(bw * 0.20), int(bh * 0.20)
        crop_x1, crop_y1 = max(0, int(x1) - pad_x), max(0, int(y1) - pad_y)
        crop_x2, crop_y2 = min(w, int(x2) + pad_x), min(h, int(y2) + pad_y)
        
        crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1
        if crop_w < 10 or crop_h < 10:
            return None

        crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        is_success, buffer = cv2.imencode(".jpg", crop_img)
        if not is_success:
            return None
            
        io_buf = io.BytesIO(buffer)
        io_buf.name = "crop.jpg"

        target_box_crop = [int(x1 - crop_x1), int(y1 - crop_y1), int(x2 - crop_x1), int(y2 - crop_y1)]
        pts, pt_labels = _generate_fg_bg_points(target_box_crop, crop_w, crop_h)

        # Semaphore regulates how many SAM 3 calls fire concurrently
        async with semaphore:
            try:
                sam_output = await _run_replicate_with_retry(
                    client,
                    "yodagg/sam3-image-seg:753fe4dbdd890a55e176f19b0603ae1b43c9e7fbd916070df53ffdb2451c7a57",
                    input_data={
                        "image": io_buf,
                        "prompt": norm_label,
                        "points": json.dumps(pts),
                        "point_labels": json.dumps(pt_labels),
                        "return_polygons": True,
                        "visualize_output": False,
                        "multimask_output": True,
                        "confidence_threshold": 0.4,
                        "max_masks": 5
                    }
                )
            except Exception as sam_err:
                logger.warning(f"SAM 3 failed for target '{norm_label}': {sam_err}")
                return None

        if not sam_output or not isinstance(sam_output, dict):
            return None

        pred_polygons = sam_output.get("pred_polygons", [])
        pred_scores = sam_output.get("pred_scores", [])
        pred_boxes = sam_output.get("pred_boxes", [])

        if not pred_polygons or not pred_scores:
            return None

        best_idx = int(np.argmax(pred_scores))
        best_sam_score = float(pred_scores[best_idx])

        if best_idx < len(pred_boxes):
            sam_box = pred_boxes[best_idx]
            if len(sam_box) == 4 and _compute_bbox_iou(target_box_crop, sam_box) < 0.15:
                return None  # Mask drifted away from target

        combined_conf = (0.6 * dino_conf) + (0.4 * best_sam_score)
        if combined_conf < 0.45:
            return None

        # Recursive Polygon Unwrapping
        valid_pts = pred_polygons[best_idx]
        while (
            isinstance(valid_pts, list)
            and len(valid_pts) > 0
            and isinstance(valid_pts[0], list)
            and len(valid_pts[0]) > 0
            and isinstance(valid_pts[0][0], list)
        ):
            valid_pts = valid_pts[0]

        if len(valid_pts) < 3:
            return None

        polygon = [{"x": float(pt[0]) + crop_x1, "y": float(pt[1]) + crop_y1} for pt in valid_pts]
        pts_np = np.array([[pt["x"], pt["y"]] for pt in polygon], dtype=np.float32)
        area = cv2.contourArea(pts_np)
        
        if area < min_area:
            return None

        tx, ty, tbw, tbh = cv2.boundingRect(pts_np)
        if tbw > 0.95 * w and tbh > 0.95 * h:
            return None

        rel_y = float(ty) / float(h)
        mapped_type = _classify_region_by_rules(norm_label, rel_y, tbw, tbh, w, h, crop_img)

        return {
            "id": f"{mapped_type}_{idx}",
            "type": mapped_type,
            "polygon": polygon,
            "bbox": [float(tx), float(ty), float(tx + tbw), float(ty + tbh)],
            "area": round(float(area), 1),
            "confidence": round(combined_conf, 3),
            "raw_label": norm_label
        }

    # Execute throttled async pipeline
    tasks = [_process_single_target(det, i) for i, det in enumerate(pruned_detections)]
    results = await asyncio.gather(*tasks)

    valid_regions = [r for r in results if r is not None]
    valid_regions.sort(key=lambda r: r["area"], reverse=True)

    counts = {}
    for r in valid_regions:
        t = r["type"]
        counts[t] = counts.get(t, 0) + 1
        r["label"] = f"{t.replace('_', ' ').title()} {counts[t]}"

    logger.info(f"Successfully processed {len(valid_regions)} regions without rate-limiting failures.")
    
    return {
        "success": True,
        "regions": valid_regions,
        "method": "dino-sam3-rate-limited"
    }


async def generate_visualization(image_url, mask_url, material_prompt, region_type):
    """Generates modified material renders using Stable Diffusion Inpainting via Replicate."""
    if not settings.replicate_api_token or settings.replicate_api_token == "your-replicate-api-token":
        return {"success": False, "error": "Replicate API token not configured."}
        
    client = replicate.Client(api_token=settings.replicate_api_token)
    prompt = (
        f"Photorealistic exterior house {region_type} with {material_prompt}, "
        f"natural lighting, architectural photography, high detail, consistent perspective, seamless blend"
    )
    
    try:
        output = await _run_replicate_with_retry(
            client,
            "stability-ai/stable-diffusion-inpainting", 
            input_data={
                "image": image_url, 
                "mask": mask_url, 
                "prompt": prompt, 
                "negative_prompt": "blurry, distorted, cartoon, illustration, low quality, inconsistent lighting", 
                "num_outputs": 1, 
                "guidance_scale": 7.5, 
                "num_inference_steps": 50
            }
        )
        result_url = output[0] if isinstance(output, list) and len(output) > 0 else output
        return {"success": True, "generated_image_url": result_url}
    except Exception as e:
        logger.error(f"Inpainting generation failed: {e}")
        return {"success": False, "error": str(e)}


async def generate_full_visualization(image_url, regions_config):
    """Applies material edits across multiple targeted regions sequentially."""
    results = []
    for region in regions_config:
        result = await generate_visualization(
            image_url=image_url, 
            mask_url=region.get("mask_url", ""), 
            material_prompt=region.get("material_prompt", ""), 
            region_type=region.get("type", "wall")
        )
        results.append(result)
        
    return {"success": any(r.get("success") for r in results), "results": results}