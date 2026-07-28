import logging
import os
import replicate
import asyncio
from ..config import get_settings

settings = get_settings()
logger = logging.getLogger("e2m.ai")


def _classify_region_by_label(label):
    """Map a segmentation label to a region type."""
    label_lower = (label or "").lower()
    if "wall" in label_lower:
        return "wall"
    if "building" in label_lower:
        return "wall"  # building maps to wall for renovation purposes
    if "fence" in label_lower:
        return "gate"
    if "vegetation" in label_lower or "tree" in label_lower:
        return "gate"
    return None


async def _detect_regions_hf(image_path):
    """Detect regions using SegFormer semantic segmentation via HF InferenceClient.

    Uses nvidia/segformer-b5-finetuned-cityscapes-1024-1024 through the official
    HuggingFace InferenceClient SDK (provider="hf-inference").

    Pipeline:
    1. Run semantic segmentation
    2. Extract wall/building masks
    3. Merge masks
    4. OpenCV cleanup (morphology close/open, remove tiny blobs, fill holes)
    5. Convert to polygons via cv2.findContours() + cv2.approxPolyDP()
    """
    hf_token = settings.hf_api_token or os.environ.get("HF_API_TOKEN", "")
    if not hf_token:
        return None

    try:
        import cv2
        import numpy as np
        img = cv2.imread(image_path)
        h, w = img.shape[:2] if img is not None else (600, 800)
    except Exception:
        h, w = 600, 800

    # --- SegFormer semantic segmentation via InferenceClient ---
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(
            provider="hf-inference",
            api_key=hf_token,
            timeout=120,
        )

        segments = await asyncio.to_thread(
            client.image_segmentation,
            image=image_path,
            model="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
            threshold=0.3,
        )
        logger.info(f"_detect_regions_hf: SegFormer found {len(segments)} segments")
    except Exception as e:
        logger.error(f"_detect_regions_hf: SegFormer exception: {e}")
        return None

    # --- Step 2: Extract wall/building masks ---
    target_labels = {"wall", "building"}
    wall_masks = []
    for seg in segments:
        # Handle both dict and object return types
        if isinstance(seg, dict):
            label = seg.get("label", "").lower()
            mask = seg.get("mask")
        else:
            label = getattr(seg, "label", "").lower()
            mask = getattr(seg, "mask", None)

        if label in target_labels and mask is not None:
            wall_masks.append(mask)

    if not wall_masks:
        logger.warning("_detect_regions_hf: no wall/building segments found")
        return None

    logger.info(f"_detect_regions_hf: found {len(wall_masks)} wall/building masks")

    # --- Step 3: Merge masks ---
    merged_mask = np.zeros((h, w), dtype=np.uint8)
    for mask in wall_masks:
        if hasattr(mask, "convert"):
            mask_np = np.array(mask.convert("L"))
        elif isinstance(mask, np.ndarray):
            mask_np = mask
        else:
            mask_np = np.array(mask)

        # Resize mask to match image dimensions
        if mask_np.shape[:2] != (h, w):
            mask_np = cv2.resize(mask_np, (w, h))

        # Threshold to binary
        _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        merged_mask = np.maximum(merged_mask, mask_binary)

    # --- Step 4: OpenCV cleanup ---
    # Fill holes
    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(merged_mask)
    cv2.fillPoly(filled, contours, 255)

    # Morphology close (fill small gaps)
    kernel_close = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_close)

    # Morphology open (remove small noise)
    kernel_open = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # Remove tiny blobs
    min_area = (w * h) * 0.01  # 1% of image area
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(opened)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(cleaned, [contour], -1, 255, -1)

    # --- Step 5: Extract polygons ---
    regions = []
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Approximate polygon to reduce excessive points
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in approx]

        if len(polygon) >= 3:
            x, y, bw, bh = cv2.boundingRect(contour)
            regions.append({
                "type": "wall",
                "polygon": polygon,
                "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
                "area": round(area, 1),
                "label": "Wall",
            })

    if not regions:
        logger.warning("_detect_regions_hf: no valid regions found after cleanup")
        return None

    # Prefer one large wall polygon over many fragmented regions
    regions.sort(key=lambda r: r["area"], reverse=True)
    logger.info(f"_detect_regions_hf: returning {len(regions)} regions")
    return regions[:10]


def _detect_regions_local(image_path):
    """Structured facade-based detection with fixed band boundaries.

    This is a fallback detector that uses OpenCV edge detection to identify
    facade regions. It does not require any external API calls.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []
    img = cv2.imread(image_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    if bw < w * 0.3 or bh < h * 0.3:
        sorted_c = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(sorted_c) > 1:
            x, y, bw, bh = cv2.boundingRect(sorted_c[1])
        if bw < w * 0.3 or bh < h * 0.3:
            return []
    logger.info(f"_detect_regions_local: facade bbox=({x},{y},{bw},{bh}), image=({w},{h})")
    regions = []
    # NARROWER wall band: 12%-70% (was 15%-85%)
    roof_y_end = y + int(bh * 0.12)
    wall_y_start = y + int(bh * 0.12)
    wall_y_end = y + int(bh * 0.70)
    bottom_y_start = y + int(bh * 0.70)
    # Roof
    if roof_y_end > y:
        regions.append({"type": "roof", "polygon": [{"x": float(x), "y": float(y)}, {"x": float(x+bw), "y": float(y)}, {"x": float(x+bw), "y": float(roof_y_end)}, {"x": float(x), "y": float(roof_y_end)}], "bbox": [float(x), float(y), float(x+bw), float(roof_y_end)], "area": round(bw*(roof_y_end-y), 1), "label": "Roof"})
    # Walls split left/right
    wall_height = wall_y_end - wall_y_start
    if wall_height > 10:
        mid_x = x + bw // 2
        regions.append({"type": "wall", "polygon": [{"x": float(x), "y": float(wall_y_start)}, {"x": float(mid_x), "y": float(wall_y_start)}, {"x": float(mid_x), "y": float(wall_y_end)}, {"x": float(x), "y": float(wall_y_end)}], "bbox": [float(x), float(wall_y_start), float(mid_x), float(wall_y_end)], "area": round((mid_x-x)*wall_height, 1), "label": "Wall (Left)"})
        regions.append({"type": "wall", "polygon": [{"x": float(mid_x), "y": float(wall_y_start)}, {"x": float(x+bw), "y": float(wall_y_start)}, {"x": float(x+bw), "y": float(wall_y_end)}, {"x": float(mid_x), "y": float(wall_y_end)}], "bbox": [float(mid_x), float(wall_y_start), float(x+bw), float(wall_y_end)], "area": round((x+bw-mid_x)*wall_height, 1), "label": "Wall (Right)"})
    # Gate split left/right (larger band)
    if bottom_y_start < y + bh:
        bottom_height = (y + bh) - bottom_y_start
        if bottom_height > 10:
            bq1 = x + bw // 4
            bq3 = x + 3 * bw // 4
            regions.append({"type": "gate", "polygon": [{"x": float(x), "y": float(bottom_y_start)}, {"x": float(bq1), "y": float(bottom_y_start)}, {"x": float(bq1), "y": float(y+bh)}, {"x": float(x), "y": float(y+bh)}], "bbox": [float(x), float(bottom_y_start), float(bq1), float(y+bh)], "area": round((bq1-x)*bottom_height, 1), "label": "Gate (Left)"})
            regions.append({"type": "gate", "polygon": [{"x": float(bq3), "y": float(bottom_y_start)}, {"x": float(x+bw), "y": float(bottom_y_start)}, {"x": float(x+bw), "y": float(y+bh)}, {"x": float(bq3), "y": float(y+bh)}], "bbox": [float(bq3), float(bottom_y_start), float(x+bw), float(y+bh)], "area": round((x+bw-bq3)*bottom_height, 1), "label": "Gate (Right)"})
    # Window detection with STRICTER filtering
    lower_white = np.array([190, 190, 190])  # Raised from 180
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(img, lower_white, upper_white)
    white_dilated = cv2.dilate(white_mask, np.ones((3, 3), np.uint8), iterations=1)
    wall_mask = np.zeros_like(white_dilated)
    wall_mask[wall_y_start:wall_y_end, x:x+bw] = 255
    window_mask = cv2.bitwise_and(white_dilated, wall_mask)
    window_contours, _ = cv2.findContours(window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in window_contours:
        area = cv2.contourArea(contour)
        if area < (w*h)*0.001 or area > (w*h)*0.03:  # Max 3% (was 5%)
            continue
        cx, cy, cbw, cbh = cv2.boundingRect(contour)
        aspect_ratio = cbw / max(cbh, 1)
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:  # Added upper limit
            continue
        if cbh < 8:
            continue
        if cy > wall_y_end - (wall_height * 0.15):  # Exclude bottom 15% of wall band
            continue
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in approx]
        wall_label = "Wall (Left)" if cx < mid_x else "Wall (Right)"
        regions.append({"type": "window", "polygon": polygon if len(polygon) >= 3 else [{"x": float(cx), "y": float(cy)}, {"x": float(cx+cbw), "y": float(cy)}, {"x": float(cx+cbw), "y": float(cy+cbh)}, {"x": float(cx), "y": float(cy+cbh)}], "bbox": [float(cx), float(cy), float(cx+cbw), float(cy+cbh)], "area": round(area, 1), "label": f"Window ({wall_label})"})
    # Pillars
    facade_roi = gray[y:y+bh, x:x+bw]
    if facade_roi.shape[1] > 10:
        edge_w = max(10, bw // 20)
        left_edge = facade_roi[:, :edge_w]
        if left_edge.mean() < gray.mean() * 0.8:
            regions.append({"type": "pillar", "polygon": [{"x": float(x), "y": float(wall_y_start)}, {"x": float(x+edge_w), "y": float(wall_y_start)}, {"x": float(x+edge_w), "y": float(wall_y_end)}, {"x": float(x), "y": float(wall_y_end)}], "bbox": [float(x), float(wall_y_start), float(x+edge_w), float(wall_y_end)], "area": round(edge_w*wall_height, 1), "label": "Pillar (Left)"})
        right_edge = facade_roi[:, -edge_w:]
        if right_edge.mean() < gray.mean() * 0.8:
            regions.append({"type": "pillar", "polygon": [{"x": float(x+bw-edge_w), "y": float(wall_y_start)}, {"x": float(x+bw), "y": float(wall_y_start)}, {"x": float(x+bw), "y": float(wall_y_end)}, {"x": float(x+bw-edge_w), "y": float(wall_y_end)}], "bbox": [float(x+bw-edge_w), float(wall_y_start), float(x+bw), float(wall_y_end)], "area": round(edge_w*wall_height, 1), "label": "Pillar (Right)"})
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    logger.info(f"_detect_regions_local: returning {len(regions)} regions")
    return regions[:10]


async def detect_regions(image_url):
    uploads_dir = os.path.abspath(settings.upload_dir)
    if image_url.startswith("/uploads/"):
        local_path = os.path.join(uploads_dir, image_url[len("/uploads/"):])
    elif image_url.startswith("uploads/"):
        local_path = os.path.abspath(image_url)
    else:
        local_path = image_url
    logger.info(f"detect_regions: image_url={image_url}, local_path={local_path}, exists={os.path.exists(local_path)}")
    if not os.path.exists(local_path):
        logger.error(f"detect_regions: image file not found at {local_path}")
        return {"success": False, "error": "Image file not found", "regions": []}
    logger.info("detect_regions: trying SegFormer semantic segmentation pipeline")
    hf_result = await _detect_regions_hf(local_path)
    if hf_result:
        logger.info(f"detect_regions: SegFormer pipeline succeeded, {len(hf_result)} regions")
        return {"success": True, "regions": hf_result, "method": "segformer-cityscapes"}
    logger.error("detect_regions: SegFormer pipeline failed")
    return {"success": False, "error": "SegFormer pipeline failed", "regions": []}


async def generate_visualization(image_url, mask_url, material_prompt, region_type):
    if not settings.replicate_api_token or settings.replicate_api_token == "your-replicate-api-token":
        return {"success": False, "error": "Replicate API token not configured. Set REPLICATE_API_TOKEN in .env"}
    client = replicate.Client(api_token=settings.replicate_api_token)
    prompt = (f"Photorealistic exterior house {region_type} with {material_prompt}, natural lighting, architectural photography, high detail, consistent perspective, seamless blend with surroundings")
    try:
        output = client.run("stability-ai/stable-diffusion-inpainting", input={"image": image_url, "mask": mask_url, "prompt": prompt, "negative_prompt": "blurry, distorted, cartoon, illustration, low quality, inconsistent lighting", "num_outputs": 1, "guidance_scale": 7.5, "num_inference_steps": 50})
        result_url = output[0] if isinstance(output, list) and len(output) > 0 else output
        return {"success": True, "generated_image_url": result_url}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def generate_full_visualization(image_url, regions_config):
    results = []
    for region in regions_config:
        result = await generate_visualization(image_url=image_url, mask_url=region.get("mask_url", ""), material_prompt=region.get("material_prompt", ""), region_type=region.get("type", "wall"))
        results.append(result)
    return {"success": any(r.get("success") for r in results), "results": results}
