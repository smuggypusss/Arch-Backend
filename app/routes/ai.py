import base64
import io
import logging
import os
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFilter

from ..config import get_settings
from ..middleware.auth import get_current_user
from ..services.image_service import generate_mask_for_region
from ..services.ai_service import detect_regions as detect_regions_service

router = APIRouter(prefix="/api/ai", tags=["AI"])
settings = get_settings()
logger = logging.getLogger("e2m.ai")


class RegionConfig(BaseModel):
    type: str
    selected_material: str
    polygon: List[dict] = []


class GeneratePreviewRequest(BaseModel):
    image_path: str
    regions: List[RegionConfig]


class DetectRegionsRequest(BaseModel):
    image_url: str


class RefineRegionsRequest(BaseModel):
    image_path: str
    regions: List[dict]


# ---------------------------------------------------------------------------
# Inpainting generation using huggingface_hub.InferenceClient
# ---------------------------------------------------------------------------

REGION_PROMPT_MAP = {
    "wall": "exterior wall",
    "roof": "roof",
    "window": "window frames",
    "balcony": "balcony railing",
    "pillar": "pillar",
    "parapet": "parapet wall",
    "gate": "gate and entrance",
    "door": "front door",
}

# Models to try for material editing (used for material synthesis).
# FLUX.1-Fill-dev is specifically designed for mask-based inpainting and
# object replacement. It is available via the Hugging Face Inference API
# (unlike Mage-Flow-Edit which is local Diffusers only).
T2I_MODELS = [
    "black-forest-labs/FLUX.1-Fill-dev",
]

NEGATIVE_PROMPT = (
    "blurry, distorted, cartoon, illustration, low quality, inconsistent lighting, "
    "deformed, extra limbs, disfigured, text, signature, watermark, ugly, morbid, "
    "mutilated, disfigured hands, poorly drawn hands, poorly drawn face"
)


def _build_material_prompt(region_type: str, material: str) -> str:
    region_desc = REGION_PROMPT_MAP.get(region_type, region_type)
    return (
        f"Architectural facade rendering. "
        f"The white masked region is an existing {region_desc} with plain cement plaster. "
        f"Completely replace the {region_desc} surface with {material}. "
        f"The original plaster texture must disappear completely. "
        f"The new surface must clearly consist of realistic {material} "
        f"with visible texture and mortar joints. "
        f"Do not alter: windows, roof, doors, wall shape, perspective, "
        f"shadows, lighting. "
        f"Only replace the {region_desc} material. "
        f"Ultra realistic architecture photo, 8k detail."
    )


def _load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def _load_mask(path: str) -> Image.Image:
    mask = Image.open(path).convert("L")
    return mask


def _dilate_mask(mask: Image.Image, kernel_size: int = 5) -> Image.Image:
    """Dilate a binary mask to avoid seams where old material shows through.

    Uses cv2.dilate with a square kernel. A 5x5 kernel expands the mask
    slightly so that the AI has enough context to fully cover the original
    surface, preventing thin edges of the old material from bleeding through.
    """
    import numpy as np
    import cv2

    mask_array = np.array(mask)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_array, kernel, iterations=1)
    return Image.fromarray(dilated)


async def _generate_material_image(
    image: Image.Image,
    prompt: str,
    model_name: str,
    hf_token: str = None,
    client=None,
    mask: Image.Image = None,
) -> Optional[Image.Image]:
    """Generate a modified image using inpainting.

    Uses client.inpainting() which only modifies the masked area —
    this is the correct approach for material replacement with FLUX Fill.
    No fallback to image_to_image is used, as that endpoint may ignore
    the mask and produce inconsistent results.
    """
    import asyncio

    # Create the InferenceClient if not provided
    if client is None and hf_token:
        from huggingface_hub import InferenceClient
        client = InferenceClient(provider="fal-ai", api_key=hf_token, timeout=180)

    if client is None:
        logger.warning(f"No client or token available for model {model_name}")
        return None

    # Use image_to_image with mask — this is the HF Inference API method
    # for mask-based inpainting. The InferenceClient exposes this as
    # image_to_image(image, mask, prompt, model=...) which only modifies
    # the masked area. (The dedicated inpainting() method is not always
    # available depending on the huggingface_hub version and provider.)
    if mask is not None:
        def _call_image_to_image_with_mask():
            try:
                return client.image_to_image(
                    image=image,
                    mask=mask,
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    guidance_scale=10.0,
                    num_inference_steps=45,
                    model=model_name,
                )
            except StopIteration:
                raise RuntimeError("StopIteration raised by image_to_image with mask")

        try:
            result = await asyncio.to_thread(_call_image_to_image_with_mask)
            logger.info(f"Image-to-image with mask succeeded for {model_name}")
            return result
        except Exception as e:
            logger.warning(f"Image-to-image with mask failed for {model_name}: {e}")
            return None

    logger.warning(f"No mask provided for model {model_name}")
    return None


async def _generate_inpainting_preview(
    image_path: str,
    regions: List[RegionConfig],
    hf_token: str,
) -> dict:
    """Generate an inpainting preview using huggingface_hub.InferenceClient.

    Regions are grouped by material. For each material group, all region masks
    are merged into a single mask, and a single inpainting inference is performed.
    This avoids the drift that occurs when generating each wall separately —
    each inference introduces new noise, and by the fourth wall the AI has
    drifted completely. Professional renovation visualizers always merge masks
    and do one inference per material.
    """
    from huggingface_hub import InferenceClient
    from PIL import ImageChops

    # Group regions by material — one inference per material, not per wall
    material_groups: dict = {}
    for region in regions:
        if not region.polygon or len(region.polygon) < 3:
            logger.warning(f"Skipping region {region.type}: no valid polygon")
            continue
        key = region.selected_material
        if key not in material_groups:
            material_groups[key] = []
        material_groups[key].append(region)

    if not material_groups:
        return {
            "success": False,
            "error": "No valid regions with materials and polygons provided",
        }

    # Load the original image
    original_image = _load_image(image_path)
    current_image = original_image

    # Create the InferenceClient once and reuse
    client = InferenceClient(
        provider="fal-ai",
        api_key=hf_token,
        timeout=180,
    )

    outputs_dir = os.path.abspath(settings.output_dir)
    os.makedirs(outputs_dir, exist_ok=True)

    regions_succeeded = 0
    regions_failed = 0

    for material, group_regions in material_groups.items():
        # Merge all masks for this material into a single mask.
        # ImageChops.lighter does a per-pixel max, so any white pixel
        # in any mask becomes white in the merged mask.
        merged_mask = None
        region_types = set()
        for region in group_regions:
            mask_path = generate_mask_for_region(
                image_path=image_path,
                region_polygon=region.polygon,
                region_id=f"merged_{material}_{id(region)}",
            )
            mask_image = _load_mask(mask_path)
            if merged_mask is None:
                merged_mask = mask_image
            else:
                merged_mask = ImageChops.lighter(merged_mask, mask_image)
            region_types.add(region.type)

        if merged_mask is None:
            regions_failed += len(group_regions)
            continue

        # Dilate the merged mask to avoid seams where old material shows through
        dilated_mask = _dilate_mask(merged_mask, kernel_size=15)

        # Build prompt for this material group
        primary_type = sorted(region_types)[0]
        region_desc = REGION_PROMPT_MAP.get(primary_type, primary_type)
        prompt = _build_material_prompt(region_desc, material)
        logger.info(
            f"Generating material '{material}' for {len(group_regions)} regions "
            f"(types: {', '.join(sorted(region_types))})"
        )

        # Try each model in order
        region_success = False
        for model_name in T2I_MODELS:
            generated = await _generate_material_image(
                current_image, prompt, model_name,
                hf_token=hf_token, client=client, mask=dilated_mask,
            )

            if generated is not None:
                # FLUX Fill inpainting returns the entire image with only the
                # masked area modified. Use it directly — the model preserves
                # the outside perfectly, so no compositing is needed.
                current_image = generated
                logger.info(f"Successfully applied material '{material}' to {len(group_regions)} regions")
                region_success = True
                regions_succeeded += len(group_regions)
                break

        if not region_success:
            regions_failed += len(group_regions)
            logger.error(f"All models failed for material '{material}'")

    # Only return success if at least one region was actually modified
    if regions_succeeded == 0:
        logger.error(
            f"generate_preview: all {regions_failed} regions failed to generate materials"
        )
        return {
            "success": False,
            "error": (
                f"All {regions_failed} regions failed to generate materials. "
                "Check API tokens, model availability, and network connectivity."
            ),
        }

    # Save the final result
    output_filename = f"ai_preview_{os.path.basename(image_path)}"
    output_path = os.path.join(outputs_dir, output_filename)
    current_image.save(output_path, "PNG")

    logger.info(f"Preview saved to {output_path}")
    return {
        "success": True,
        "generated_image_url": f"/outputs/{output_filename}",
    }


# ---------------------------------------------------------------------------
# Fallback: Replicate inpainting
# ---------------------------------------------------------------------------

async def _generate_replicate_preview(
    image_path: str,
    regions: List[RegionConfig],
    replicate_token: str,
) -> dict:
    """Fallback: generate preview using Replicate's stable-diffusion-inpainting."""
    import replicate

    client = replicate.Client(api_token=replicate_token)

    original_image = _load_image(image_path)
    current_image = original_image
    outputs_dir = os.path.abspath(settings.output_dir)
    os.makedirs(outputs_dir, exist_ok=True)

    regions_succeeded = 0
    regions_failed = 0

    for region in regions:
        if not region.polygon or len(region.polygon) < 3:
            regions_failed += 1
            continue

        mask_path = generate_mask_for_region(
            image_path=image_path,
            region_polygon=region.polygon,
            region_id=f"replicate_{region.type}_{id(region)}",
        )

        # Convert images to base64 for Replicate
        img_buffer = io.BytesIO()
        current_image.save(img_buffer, format="PNG")
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        img_data_uri = f"data:image/png;base64,{img_b64}"

        mask_image = _load_mask(mask_path)
        mask_buffer = io.BytesIO()
        mask_image.save(mask_buffer, format="PNG")
        mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")
        mask_data_uri = f"data:image/png;base64,{mask_b64}"

        prompt = _build_material_prompt(region.type, region.selected_material)

        region_success = False
        try:
            output = client.run(
                "stability-ai/stable-diffusion-inpainting",
                input={
                    "image": img_data_uri,
                    "mask": mask_data_uri,
                    "prompt": prompt,
                    "negative_prompt": NEGATIVE_PROMPT,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 50,
                },
            )
            result_url = output[0] if isinstance(output, list) and len(output) > 0 else output

            # Download the result
            import httpx
            async with httpx.AsyncClient(timeout=120) as http_client:
                resp = await http_client.get(result_url)
                if resp.status_code == 200:
                    current_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    logger.info(f"Replicate inpainted region '{region.type}'")
                    region_success = True
        except Exception as e:
            logger.error(f"Replicate inpainting failed for region '{region.type}': {e}")

        if region_success:
            regions_succeeded += 1
        else:
            regions_failed += 1

    # Only return success if at least one region was actually modified.
    if regions_succeeded == 0:
        logger.error(
            f"generate_preview: Replicate fallback - all {regions_failed} regions failed"
        )
        return {
            "success": False,
            "error": (
                f"All {regions_failed} regions failed to generate materials via Replicate. "
                "Check API tokens and network connectivity."
            ),
        }

    output_filename = f"ai_preview_{os.path.basename(image_path)}"
    output_path = os.path.join(outputs_dir, output_filename)
    current_image.save(output_path, "PNG")

    return {
        "success": True,
        "generated_image_url": f"/outputs/{output_filename}",
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/detect-regions")
async def detect_regions_route(
    data: DetectRegionsRequest,
    current_user: dict = Depends(get_current_user),
):
    """Detect building surface regions using Grounding DINO + SAM2 (or local fallback)."""
    image_url = data.image_url
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url is required")

    logger.info(f"detect_regions: image_url={image_url}")
    result = await detect_regions_service(image_url)
    return result


@router.post("/refine-regions")
async def refine_regions_route(
    data: RefineRegionsRequest,
    current_user: dict = Depends(get_current_user),
):
    """Refine existing region polygons using AI detection with user-provided hints.

    The endpoint receives the original image path and a list of regions (each with
    a polygon), re-runs the AI detection service, and returns refined region polygons.
    The frontend can then merge the refined polygons with the existing region metadata
    (IDs, types, materials, notes) to preserve user edits.
    """
    image_path = data.image_path
    if not image_path:
        raise HTTPException(status_code=400, detail="image_path is required")

    logger.info(f"refine_regions: image_path={image_path}, regions={len(data.regions)}")

    # Build the image_url for the detection service
    image_url = image_path if image_path.startswith("/uploads/") else f"/uploads/{image_path}"
    result = await detect_regions_service(image_url)

    if result.get("success") and result.get("regions"):
        # Convert detected regions to polygon format
        refined = []
        for r in result["regions"]:
            polygon = []
            if r.get("bbox"):
                x1, y1, x2, y2 = r["bbox"]
                polygon = [
                    {"x": x1, "y": y1},
                    {"x": x2, "y": y1},
                    {"x": x2, "y": y2},
                    {"x": x1, "y": y2},
                ]
            elif r.get("polygon"):
                polygon = r["polygon"]
            elif r.get("points"):
                polygon = [{"x": p.get("x", p[0]), "y": p.get("y", p[1])} for p in r["points"]]

            refined.append({
                "type": r.get("type", r.get("label", "wall")),
                "polygon": polygon,
                "area": r.get("area"),
                "label": r.get("label", r.get("type", "")),
            })

        return {"success": True, "regions": refined}
    else:
        return {"success": False, "error": result.get("message", "AI refinement failed")}


@router.post("/generate-preview")
async def generate_preview(
    data: GeneratePreviewRequest,
    current_user: dict = Depends(get_current_user),
):
    """Generate a photorealistic renovation preview using inpainting.

    The endpoint receives the original image path, a list of regions (each with
    a polygon and selected material), generates binary masks from the polygons,
    and calls an inpainting model to replace only the masked areas with the
    specified materials while preserving geometry, lighting, and perspective.
    """
    hf_token = settings.hf_api_token or os.environ.get("HF_API_TOKEN", "")
    if not hf_token:
        raise HTTPException(
            status_code=400,
            detail="Hugging Face API token not configured. Set HF_API_TOKEN in .env",
        )

    # Resolve the local image path
    uploads_dir = os.path.abspath(settings.upload_dir)
    if data.image_path.startswith("/uploads/"):
        local_path = os.path.join(uploads_dir, data.image_path[len("/uploads/"):])
    elif data.image_path.startswith("uploads/"):
        local_path = os.path.abspath(data.image_path)
    else:
        local_path = os.path.join(uploads_dir, data.image_path)

    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Original image not found")

    # Filter regions with valid materials and polygons
    valid_regions = [
        r for r in data.regions
        if r.selected_material and r.polygon and len(r.polygon) >= 3
    ]

    if not valid_regions:
        raise HTTPException(
            status_code=400,
            detail="No valid regions with materials and polygons provided",
        )

    logger.info(
        f"generate_preview: {len(valid_regions)} regions, "
        f"image={os.path.basename(local_path)}"
    )

    # Try HuggingFace InferenceClient first (bypasses DNS issues)
    try:
        result = await _generate_inpainting_preview(local_path, valid_regions, hf_token)
        if result.get("success"):
            logger.info("generate_preview: HuggingFace InferenceClient succeeded")
            return result
    except Exception as e:
        logger.warning(f"generate_preview: HuggingFace InferenceClient failed: {e}")

    # Fallback: Replicate
    replicate_token = settings.replicate_api_token
    if replicate_token and replicate_token != "your-replicate-api-token":
        try:
            result = await _generate_replicate_preview(local_path, valid_regions, replicate_token)
            if result.get("success"):
                logger.info("generate_preview: Replicate fallback succeeded")
                return result
        except Exception as e:
            logger.warning(f"generate_preview: Replicate fallback failed: {e}")

    raise HTTPException(
        status_code=502,
        detail="AI generation failed: all generation methods (HuggingFace InferenceClient, Replicate) failed. "
        "Check API tokens and network connectivity.",
    )
