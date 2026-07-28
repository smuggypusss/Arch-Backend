import logging
import os
import asyncio
import cv2
import numpy as np
import replicate
from ..config import get_settings

settings = get_settings()
logger = logging.getLogger("e2m.ai")


def _classify_region_by_rules(label_raw, rel_y, bw, bh, w, h, img_crop=None):
    """
    Classifies regions based on labels, geometric rules, and visual heuristics.
    """
    label_lower = (label_raw or "").lower()

    if "roof" in label_lower or "eave" in label_lower:
        return "roof"
    if "window" in label_lower or "trim" in label_lower or "glass" in label_lower:
        return "window"
    if "balcony" in label_lower or "railing" in label_lower:
        return "balcony"
    if "pillar" in label_lower or "column" in label_lower:
        return "pillar"
    if "parapet" in label_lower or "terrace" in label_lower:
        return "parapet"
    if "gate" in label_lower or "entrance" in label_lower or "door" in label_lower:
        return "gate"
        
    # Visual Heuristic for Modern Dark Glass Windows:
    # Modern glass is darker than white stucco walls (mean brightness < 110)
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


def _split_mask_with_distance_transform(binary_mask, min_area):
    """
    Uses Distance Transform + Watershed to split adjacent touching objects.
    """
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    if dist_transform.max() == 0:
        return [binary_mask]

    _, foreground_seeds = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    foreground_seeds = np.uint8(foreground_seeds)

    num_seeds, markers = cv2.connectedComponents(foreground_seeds)
    if num_seeds <= 2:
        return [binary_mask]

    mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    markers = markers + 1
    markers[binary_mask == 0] = 0

    cv2.watershed(mask_3ch, markers)

    split_masks = []
    for label_id in range(2, num_seeds + 1):
        sub_mask = np.zeros_like(binary_mask)
        sub_mask[markers == label_id] = 255
        if cv2.countNonZero(sub_mask) >= min_area:
            split_masks.append(sub_mask)

    return split_masks if split_masks else [binary_mask]


async def detect_regions(image_url):
    """
    Mask2Former (ADE20K) pipeline integrated with custom classification rules and heuristics.
    """
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
    hf_token = settings.hf_api_token or os.environ.get("HF_API_TOKEN", "")

    if not hf_token:
        return {"success": False, "error": "HF API token missing", "regions": []}

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(provider="hf-inference", api_key=hf_token, timeout=60)
        
        logger.info("Executing Mask2Former ADE20K inference...")
        raw_results = await asyncio.to_thread(
            client.image_segmentation,
            image=local_path,
            model="facebook/mask2former-swin-large-ade-semantic"
        )
    except Exception as e:
        logger.error(f"Mask2Former Inference failed: {e}")
        return {"success": False, "error": f"HF Inference Error: {str(e)}", "regions": []}

    regions = []
    min_area = total_image_area * 0.0002

    for item in raw_results:
        score = item.get("score", 1.0)
        if score < 0.20:
            continue

        label_raw = item.get("label", "").lower()
        mask_data = item.get("mask")
        if not mask_data or not hasattr(mask_data, "convert"):
            continue
            
        mask_np = np.array(mask_data.convert("L"))

        if mask_np.shape[:2] != (h, w):
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        sub_masks = _split_mask_with_distance_transform(binary_mask, min_area)

        for sub_mask in sub_masks:
            num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(sub_mask, connectivity=8)

            for comp_idx in range(1, num_labels):
                area = stats[comp_idx, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue

                comp_mask = np.zeros((h, w), dtype=np.uint8)
                comp_mask[labels_im == comp_idx] = 255

                contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                largest_contour = max(contours, key=cv2.contourArea)
                tx, ty, tbw, tbh = cv2.boundingRect(largest_contour)

                if tbw > 0.95 * w and tbh > 0.95 * h:
                    continue

                rel_y = float(ty) / float(h)

                # Extract image crop for visual window heuristic
                x_c, y_c, w_c, h_c = int(tx), int(ty), int(tbw), int(tbh)
                img_crop = img[max(0, y_c):min(h, y_c+h_c), max(0, x_c):min(w, x_c+w_c)]

                # Classify using rules + visual crop
                mapped_type = _classify_region_by_rules(label_raw, rel_y, tbw, tbh, w, h, img_crop)

                arc_len = cv2.arcLength(largest_contour, True)
                if area < 4000:
                    epsilon = 0.003 * arc_len
                elif area < 20000:
                    epsilon = 0.006 * arc_len
                else:
                    epsilon = 0.012 * arc_len

                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                polygon = [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in approx]

                if len(polygon) < 3:
                    continue

                regions.append({
                    "id": f"{mapped_type}_{len(regions)}",
                    "type": mapped_type,
                    "polygon": polygon,
                    "bbox": [float(tx), float(ty), float(tx + tbw), float(ty + tbh)],
                    "area": round(float(area), 1),
                    "confidence": round(float(score), 3),
                    "raw_label": label_raw
                })

    regions.sort(key=lambda r: r["area"], reverse=True)

    counts = {}
    for r in regions:
        t = r["type"]
        counts[t] = counts.get(t, 0) + 1
        r["label"] = f"{t.replace('_', ' ').title()} {counts[t]}"

    logger.info(f"Extracted {len(regions)} refined instance regions with heuristic rules")
    return {
        "success": True,
        "regions": regions,
        "method": "mask2former-heuristic-rules"
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