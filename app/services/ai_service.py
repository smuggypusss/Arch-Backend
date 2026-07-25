import logging
import ssl
import replicate
import httpx
import os
from urllib.parse import urlparse
from ..config import get_settings
from ..utils.dns_resolver import (
    resolve_hostname,
    resolve_hostname_doh,
    get_hardcoded_ip,
    cache_resolved_ip,
)

settings = get_settings()
logger = logging.getLogger("e2m.ai")


async def _http_post_with_dns_fallback(url, headers, json_data, timeout=60):
    """Make an HTTP POST request with DNS fallback.

    If the system DNS fails to resolve the hostname (getaddrinfo failed),
    use the raw DNS resolver to obtain the IP address and retry the request
    with the IP in the URL while preserving the original Host header.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=json_data)
            return resp
    except Exception as e:
        error_msg = str(e).lower()
        is_dns_error = (
            "getaddrinfo" in error_msg
            or "name or service not known" in error_msg
            or "errno 11001" in error_msg
            or "nodename nor servname" in error_msg
        )
        if not is_dns_error:
            raise

        # DNS resolution failed — try raw DNS fallback
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            raise

        # Try raw UDP DNS first, then fall back to DNS-over-HTTPS
        ip = resolve_hostname(hostname)
        if not ip:
            logger.info("_http_post_with_dns_fallback: raw DNS failed, trying DoH for %s", hostname)
            ip = await resolve_hostname_doh(hostname)

        # Last resort: hardcoded IP fallback for known endpoints
        if not ip:
            logger.info("_http_post_with_dns_fallback: DoH failed, trying hardcoded IP for %s", hostname)
            ip = get_hardcoded_ip(hostname)

        if not ip:
            logger.warning("_http_post_with_dns_fallback: all DNS methods could not resolve %s", hostname)
            raise

        # Cache successful resolution for future use
        cache_resolved_ip(hostname, ip)

        logger.info("_http_post_with_dns_fallback: DNS fallback resolved %s -> %s", hostname, ip)

        # Build an SSL context that skips hostname verification (we connect
        # to the IP, not the hostname) while still verifying the certificate
        # chain when possible.
        ssl_context = ssl.create_default_context()
        try:
            ssl_context.check_hostname = False
        except (AttributeError, ValueError):
            pass

        transport = httpx.AsyncHTTPTransport(verify=ssl_context)

        # Replace hostname with IP in the URL
        new_url = url.replace(f"{parsed.scheme}://{hostname}", f"{parsed.scheme}://{ip}", 1)

        # Preserve the original Host header so the server routes correctly
        new_headers = {**headers, "Host": hostname}

        async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
            resp = await client.post(new_url, headers=new_headers, json=json_data)
            return resp


def _classify_region(rel_y, label, bw, bh, w, h):
    label_lower = (label or "").lower()
    if "roof" in label_lower or "eave" in label_lower:
        return "roof"
    if "window" in label_lower or "trim" in label_lower:
        return "window"
    if "balcony" in label_lower or "railing" in label_lower:
        return "balcony"
    if "pillar" in label_lower or "column" in label_lower:
        return "pillar"
    if "parapet" in label_lower or "terrace" in label_lower:
        return "parapet"
    if "gate" in label_lower or "entrance" in label_lower or "door" in label_lower:
        return "gate"
    if rel_y < 0.15:
        return "roof"
    if bw < w * 0.12 and bh > h * 0.15:
        return "pillar"
    return "wall"


async def _detect_regions_hf(image_path):
    """Detect regions using huggingface_hub.InferenceClient (bypasses DNS issues)."""
    hf_token = settings.hf_api_token or os.environ.get("HF_API_TOKEN", "")
    if not hf_token:
        return None

    try:
        from huggingface_hub import InferenceClient
        import asyncio
    except ImportError:
        logger.warning("_detect_regions_hf: huggingface_hub not installed")
        return None

    try:
        import cv2
        img = cv2.imread(image_path)
        h, w = img.shape[:2] if img is not None else (600, 800)
    except Exception:
        h, w = 600, 800

    try:
        from PIL import Image
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"_detect_regions_hf: failed to load image: {e}")
        return None

    client = InferenceClient(
        provider="fal-ai",
        api_key=hf_token,
        timeout=120,
    )

    gdino_prompt = "exterior wall. window. roof. balcony. pillar. parapet. door. gate."

    # --- Grounding DINO object detection ---
    boxes = []
    labels = []
    try:
        detections = await asyncio.to_thread(
            client.object_detection,
            image=pil_image,
            prompt=gdino_prompt,
            model="facebook/grounding-dino",
            threshold=0.4,
        )
        logger.info(f"_detect_regions_hf: Grounding DINO found {len(detections)} detections")
        for det in detections:
            score = det.get("score", 0)
            if score < 0.3:
                continue
            label = det.get("label", "")
            box = det.get("box", {})
            x1, y1 = box.get("x", 0), box.get("y", 0)
            x2, y2 = box.get("x2", x1), box.get("y2", y1)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            region_type = _classify_region(y1 / h, label, x2 - x1, y2 - y1, w, h)
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(region_type)
        if not boxes:
            logger.warning("_detect_regions_hf: no boxes after filtering")
            return None
        logger.info(f"_detect_regions_hf: {len(boxes)} boxes after filtering")
    except Exception as e:
        logger.error(f"_detect_regions_hf: Grounding DINO exception: {e}")
        return None

    # --- SAM2 segmentation ---
    regions = []
    try:
        masks_data = await asyncio.to_thread(
            client.semantic_segmentation,
            image=pil_image,
            model="facebook/sam-2",
        )
        # SAM2 may return masks differently; fall back to bounding boxes
        for i, mask_result in enumerate(masks_data):
            if i >= len(labels):
                break
            region_type = labels[i]
            x1, y1, x2, y2 = boxes[i]
            polygon = _mask_to_polygon(mask_result, x1, y1, x2, y2, w, h)
            if polygon:
                regions.append({"type": region_type, "polygon": polygon, "bbox": [x1, y1, x2, y2], "area": round((x2-x1)*(y2-y1), 1), "label": region_type.capitalize()})
            else:
                polygon = [{"x": x1, "y": y1}, {"x": x2, "y": y1}, {"x": x2, "y": y2}, {"x": x1, "y": y2}]
                regions.append({"type": region_type, "polygon": polygon, "bbox": [x1, y1, x2, y2], "area": round((x2-x1)*(y2-y1), 1), "label": region_type.capitalize()})
    except Exception as e:
        logger.error(f"_detect_regions_hf: SAM2 exception: {e}")
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box
            regions.append({"type": label, "polygon": [{"x": x1, "y": y1}, {"x": x2, "y": y1}, {"x": x2, "y": y2}, {"x": x1, "y": y2}], "bbox": [x1, y1, x2, y2], "area": round((x2-x1)*(y2-y1), 1), "label": label.capitalize()})

    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    logger.info(f"_detect_regions_hf: returning {len(regions)} regions")
    return regions[:10]


def _mask_to_polygon(mask_data, x1, y1, x2, y2, w, h):
    try:
        import cv2, numpy as np, base64
        if isinstance(mask_data, dict):
            mask_data = mask_data.get("mask", mask_data.get("data"))
        if isinstance(mask_data, str):
            mask = np.frombuffer(base64.b64decode(mask_data), dtype=np.uint8).reshape((int(y2-y1), int(x2-x1)))
        elif isinstance(mask_data, list):
            mask_arr = np.array(mask_data, dtype=np.uint8)
            mask = mask_arr.reshape((int(y2-y1), int(x2-x1))) if mask_arr.ndim == 1 else mask_arr
        else:
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(largest, 0.02 * cv2.arcLength(largest, True), True)
        polygon = [{"x": float(pt[0][0] + x1), "y": float(pt[0][1] + y1)} for pt in approx]
        return polygon if len(polygon) >= 3 else None
    except Exception:
        return None


def _detect_regions_local(image_path):
    """Structured facade-based detection with fixed band boundaries."""
    try:
        import cv2, numpy as np
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
    if os.path.exists(local_path):
        logger.info("detect_regions: trying Grounding DINO + SAM2 pipeline")
        hf_result = await _detect_regions_hf(local_path)
        if hf_result:
            logger.info(f"detect_regions: HF pipeline succeeded, {len(hf_result)} regions")
            return {"success": True, "regions": hf_result, "method": "grounding-dino-sam2"}
        logger.warning("detect_regions: HF pipeline returned no results")
    if os.path.exists(local_path):
        logger.info("detect_regions: trying local OpenCV detection")
        regions = _detect_regions_local(local_path)
        if regions:
            logger.info(f"detect_regions: local OpenCV succeeded, {len(regions)} regions")
            return {"success": True, "regions": regions, "method": "local-opencv"}
        logger.warning("detect_regions: local OpenCV returned no results")
    if settings.replicate_api_token and settings.replicate_api_token != "":
        logger.info("detect_regions: trying Replicate SAM-2")
        client = replicate.Client(api_token=settings.replicate_api_token)
        try:
            output = client.run("meta/sam-2", input={"image": image_url, "task": "segment"})
            result = {"success": True, "regions": output if output else [], "method": "replicate-sam2"}
            logger.info("detect_regions: Replicate SAM-2 succeeded")
            return result
        except Exception as e:
            logger.error(f"detect_regions: Replicate SAM-2 failed: {e}")
            return {"success": False, "error": str(e), "regions": []}
    logger.error("detect_regions: no detection method available")
    return {"success": False, "error": "No detection method available", "regions": []}

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

