import logging
import os
import asyncio
import io
import cv2
import json
import numpy as np
import replicate
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

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def _apply_nms(regions, iou_threshold=0.45):
    """Applies Non-Maximum Suppression to remove heavily overlapping duplicate masks."""
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


def _generate_fg_bg_points(target_box_crop, crop_w, crop_h):
    """
    Generates structured positive points inside the target object region
    and negative points in the padded context region outside it.
    
    target_box_crop: [tx1, ty1, tx2, ty2] inside the crop coordinates
    """
    tx1, ty1, tx2, ty2 = target_box_crop
    tbw = tx2 - tx1
    tbh = ty2 - ty1

    points = []
    labels = []

    # 1. Foreground Points (4x4 Grid in central 70% of target box)
    fg_x = np.linspace(tx1 + tbw * 0.15, tx2 - tbw * 0.15, 4)
    fg_y = np.linspace(ty1 + tbh * 0.15, ty2 - tbh * 0.15, 4)
    for x in fg_x:
        for y in fg_y:
            points.append([int(x), int(y)])
            labels.append(1)

    # 2. Background Points (8 points sampling the padded perimeter)
    bg_candidates = [
        [tx1 / 2.0, ty1 / 2.0],                             # Top-Left pad
        [tx1 + tbw / 2.0, ty1 / 2.0],                       # Top pad
        [tx2 + (crop_w - tx2) / 2.0, ty1 / 2.0],            # Top-Right pad
        [tx1 / 2.0, ty1 + tbh / 2.0],                       # Left pad
        [tx2 + (crop_w - tx2) / 2.0, ty1 + tbh / 2.0],      # Right pad
        [tx1 / 2.0, ty2 + (crop_h - ty2) / 2.0],            # Bottom-Left pad
        [tx1 + tbw / 2.0, ty2 + (crop_h - ty2) / 2.0],      # Bottom pad
        [tx2 + (crop_w - tx2) / 2.0, ty2 + (crop_h - ty2) / 2.0] # Bottom-Right pad
    ]

    for px, py in bg_candidates:
        # Only add background points if they lie within valid crop boundaries
        if 0 <= px < crop_w and 0 <= py < crop_h:
            points.append([int(px), int(py)])
            labels.append(0)

    return points, labels


async def detect_regions(image_url):
    """
    Optimized Pipeline:
    1. Grounding DINO -> Bounding Boxes & Confidence Scores
    2. Padded Cropping + Positive/Negative Grid Point Sampling
    3. SAM 3 -> Polygons & Predicted Boxes
    4. Box Overlap Validation + Weighted Score Calculation
    5. NMS Deduplication Pass
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

    try:
        logger.info("Executing Grounding DINO detection...")
        with open(local_path, "rb") as image_file:
            dino_output = await asyncio.to_thread(
                client.run,
                "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",
                input={
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
        return {"success": True, "regions": [], "method": "grounding-dino-sam3-guided"}

    logger.info(f"DINO found {len(detections)} potential regions. Parsing...")

    raw_regions = []

    for det in detections:
        raw_label = det.get("label", "wall").lower()
        norm_label = LABEL_MAP.get(raw_label, raw_label)
        dino_conf = float(det.get("confidence", det.get("score", 0.85)))
        
        # 1. Dynamic Bounding Box Extraction
        box_data = det.get("bbox") or det.get("box")
        if not box_data:
            continue
            
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
                continue
        except Exception as e:
            logger.warning(f"Failed to parse bbox {box_data}: {e}")
            continue

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        # 2. Padded Cropping (20% context pad)
        pad_x = int(bw * 0.20)
        pad_y = int(bh * 0.20)
        
        crop_x1 = max(0, int(x1) - pad_x)
        crop_y1 = max(0, int(y1) - pad_y)
        crop_x2 = min(w, int(x2) + pad_x)
        crop_y2 = min(h, int(y2) + pad_y)
        
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        if crop_w < 10 or crop_h < 10:
            continue

        crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        is_success, buffer = cv2.imencode(".jpg", crop_img)
        if not is_success:
            continue
            
        io_buf = io.BytesIO(buffer)
        io_buf.name = "crop.jpg"

        # 3. Target region in crop-local coordinates
        target_box_crop = [
            int(x1 - crop_x1),
            int(y1 - crop_y1),
            int(x2 - crop_x1),
            int(y2 - crop_y1)
        ]

        # 4. Generate Foreground (1) and Background (0) Points
        pts, pt_labels = _generate_fg_bg_points(target_box_crop, crop_w, crop_h)

        # 5. Execute SAM 3 on Crop
        try:
            sam_output = await asyncio.to_thread(
                client.run,
                "yodagg/sam3-image-seg:753fe4dbdd890a55e176f19b0603ae1b43c9e7fbd916070df53ffdb2451c7a57",
                input={
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
            logger.warning(f"SAM 3 failed for '{norm_label}' crop: {sam_err}")
            continue

        if not sam_output or not isinstance(sam_output, dict):
            continue

        pred_polygons = sam_output.get("pred_polygons", [])
        pred_scores = sam_output.get("pred_scores", [])
        pred_boxes = sam_output.get("pred_boxes", [])

        if not pred_polygons or not pred_scores:
            continue

        # 6. Select and Validate Best Mask
        best_idx = int(np.argmax(pred_scores))
        best_sam_score = float(pred_scores[best_idx])

        # Validate predicted mask box overlap against intended target box
        if best_idx < len(pred_boxes):
            sam_box = pred_boxes[best_idx]
            if len(sam_box) == 4:
                overlap_iou = _compute_bbox_iou(target_box_crop, sam_box)
                if overlap_iou < 0.15:  # Reject mask if SAM drifted completely away from DINO target
                    logger.debug(f"Rejected mask for {norm_label} due to low target overlap IoU ({overlap_iou:.2f})")
                    continue

        # Weighted confidence calculation
        combined_conf = (0.6 * dino_conf) + (0.4 * best_sam_score)
        if combined_conf < 0.45:
            continue

        # 7. Recursive Polygon Unwrapping
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
            continue

        # 8. Translate coordinates to global image space
        polygon = [
            {
                "x": float(pt[0]) + crop_x1, 
                "y": float(pt[1]) + crop_y1
            } 
            for pt in valid_pts
        ]
        
        pts_np = np.array([[pt["x"], pt["y"]] for pt in polygon], dtype=np.float32)
        area = cv2.contourArea(pts_np)
        
        if area < min_area:
            continue

        tx, ty, tbw, tbh = cv2.boundingRect(pts_np)
        if tbw > 0.95 * w and tbh > 0.95 * h:
            continue

        rel_y = float(ty) / float(h)
        mapped_type = _classify_region_by_rules(norm_label, rel_y, tbw, tbh, w, h, crop_img)

        raw_regions.append({
            "id": f"{mapped_type}_{len(raw_regions)}",
            "type": mapped_type,
            "polygon": polygon,
            "bbox": [float(tx), float(ty), float(tx + tbw), float(ty + tbh)],
            "area": round(float(area), 1),
            "confidence": round(combined_conf, 3),
            "raw_label": norm_label
        })

    # 9. Non-Maximum Suppression (Deduplication)
    deduped_regions = _apply_nms(raw_regions, iou_threshold=0.45)
    deduped_regions.sort(key=lambda r: r["area"], reverse=True)

    # Re-index labels cleanly
    counts = {}
    for r in deduped_regions:
        t = r["type"]
        counts[t] = counts.get(t, 0) + 1
        r["label"] = f"{t.replace('_', ' ').title()} {counts[t]}"

    logger.info(f"Extracted {len(deduped_regions)} deduplicated, high-precision architectural regions.")
    
    return {
        "success": True,
        "regions": deduped_regions,
        "method": "dino-sam3-point-guided"
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
        output = await asyncio.to_thread(
            client.run,
            "stability-ai/stable-diffusion-inpainting", 
            input={
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