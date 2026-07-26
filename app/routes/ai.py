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

# Models for material editing/inpainting.
# Note: FLUX.1-Fill-dev is specifically optimized for mask-based inpainting.
MODELS = [
    ("replicate", "black-forest-labs/FLUX.1-Kontext-dev"),
    ("replicate", "Qwen/Qwen-Image-Edit"),
    ("wavespeed", "black-forest-labs/FLUX.1-Kontext-dev"),
    ("fal-ai", "Qwen/Qwen-Image-Edit"),
]

NEGATIVE_PROMPT = (
    "blurry, distorted, cartoon, illustration, low quality, inconsistent lighting, "
    "deformed, extra limbs, disfigured, text, signature, watermark, ugly, morbid, "
    "mutilated, disfigured hands, poorly drawn hands, poorly drawn face"
)


def _build_material_prompt(region_type: str, material: str) -> str:
    region_desc = REGION_PROMPT_MAP.get(region_type, region_type)
    return (
        f"Replace ONLY the masked {region_desc} with {material}. "
        f"Transform the white painted {region_desc} into realistic {material} "
        f"with visible texture and mortar joints. "
        f"The change must be clearly visible. "
        f"Keep the glass unchanged. "
        f"Keep the building geometry identical. "
        f"Do not modify any unmasked pixels. "
        f"Ignore everything outside the mask. "
        f"Ultra realistic architecture photo, 8k detail."
    )


def _load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def _load_mask(path: str) -> Image.Image:
    mask = Image.open(path).convert("L")
    return mask


def _dilate_mask(mask: Image.Image, kernel_size: int = 5) -> Image.Image:
    """Dilate a binary mask to avoid seams where old material shows through."""
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
    provider: str,
    hf_token: str = None,
    mask: Image.Image = None,
) -> Optional[Image.Image]:
    """Generate a modified image using image_to_image with mask."""
    import asyncio
    from huggingface_hub import InferenceClient

    if mask is None:
        logger.warning(f"No mask provided for model {model_name}")
        return None

    if not hf_token:
        logger.warning(f"No HF token available for model {model_name}")
        return None

    try:
        current_client = InferenceClient(
            provider=provider,
            api_key=hf_token,
            timeout=180,
        )

        def _call_image_to_image_with_mask():
            try:
                # Convert source image to bytes
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")
                img_bytes = img_buffer.getvalue()

                # Convert mask to base64 Data URI string so kwargs can be JSON-serialized
                mask_buffer = io.BytesIO()
                mask.save(mask_buffer, format="PNG")
                mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")
                mask_data_uri = f"data:image/png;base64,{mask_b64}"

                return current_client.image_to_image(
                    image=img_bytes,
                    mask=mask_data_uri,
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    guidance_scale=10.0,
                    num_inference_steps=45,
                    model=model_name,
                )
            except StopIteration:
                raise RuntimeError("StopIteration raised by image_to_image with mask")

        result = await asyncio.to_thread(_call_image_to_image_with_mask)
        logger.info(f"Image-to-image with mask succeeded for {model_name} via {provider}")
        return result
    except Exception as e:
        logger.warning(f"Image-to-image with mask failed for {model_name} via {provider}: {e}")
        return None


async def _generate_inpainting_preview(
    image_path: str,
    regions: List[RegionConfig],
    hf_token: str,
) -> dict:
    """Generate an inpainting preview using huggingface_hub.InferenceClient."""
    from PIL import ImageChops

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

    original_image = _load_image(image_path)
    current_image = original_image

    outputs_dir = os.path.abspath(settings.output_dir)
    os.makedirs(outputs_dir, exist_ok=True)

    regions_succeeded = 0
    regions_failed = 0

    for material, group_regions in material_groups.items():
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

        dilated_mask = _dilate_mask(merged_mask, kernel_size=15)

        # Save debug mask for inspection
        debug_mask_path = os.path.join(outputs_dir, f"debug_mask_{material}.png")
        dilated_mask.save(debug_mask_path, "PNG")
        logger.info(f"Saved debug mask to {debug_mask_path}")

        primary_type = sorted(region_types)[0]
        region_desc = REGION_PROMPT_MAP.get(primary_type, primary_type)
        prompt = _build_material_prompt(region_desc, material)
        logger.info(
            f"Generating material '{material}' for {len(group_regions)} regions "
            f"(types: {', '.join(sorted(region_types))})"
        )

        region_success = False
        for provider, model_name in MODELS:
            generated = await _generate_material_image(
                current_image, prompt, model_name,
                provider=provider, hf_token=hf_token, mask=dilated_mask,
            )

            if generated is not None:
                # Verify the model actually changed something — some providers
                # silently ignore the mask and return an identical image.
                diff = ImageChops.difference(
                    current_image.resize(generated.size),
                    generated,
                )
                diff_bbox = diff.getbbox()
                if diff_bbox is None:
                    logger.warning(
                        f"Model {model_name} via {provider} returned an identical image "
                        f"(mask may have been ignored). Trying next model."
                    )
                    continue

                # Save debug difference image
                debug_diff_path = os.path.join(
                    outputs_dir, f"debug_diff_{material}_{provider}.png"
                )
                diff.save(debug_diff_path, "PNG")
                logger.info(f"Saved debug difference to {debug_diff_path}")

                current_image = generated
                logger.info(f"Successfully applied material '{material}' to {len(group_regions)} regions")
                region_success = True
                regions_succeeded += len(group_regions)
                break

        if not region_success:
            regions_failed += len(group_regions)
            logger.error(f"All models failed for material '{material}'")

    if regions_succeeded == 0:
        logger.error(f"generate_preview: all {regions_failed} regions failed to generate materials")
        return {
            "success": False,
            "error": (
                f"All {regions_failed} regions failed to generate materials. "
                "Check API tokens, model availability, and network connectivity."
            ),
        }

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

    if regions_succeeded == 0:
        logger.error(f"generate_preview: Replicate fallback - all {regions_failed} regions failed")
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
    """Refine existing region polygons using AI detection with user-provided hints."""
    image_path = data.image_path
    if not image_path:
        raise HTTPException(status_code=400, detail="image_path is required")

    logger.info(f"refine_regions: image_path={image_path}, regions={len(data.regions)}")

    image_url = image_path if image_path.startswith("/uploads/") else f"/uploads/{image_path}"
    result = await detect_regions_service(image_url)

    if result.get("success") and result.get("regions"):
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
    """Generate a photorealistic renovation preview using inpainting."""
    hf_token = settings.hf_api_token or os.environ.get("HF_API_TOKEN", "")
    if not hf_token:
        raise HTTPException(
            status_code=400,
            detail="Hugging Face API token not configured. Set HF_API_TOKEN in .env",
        )

    uploads_dir = os.path.abspath(settings.upload_dir)
    if data.image_path.startswith("/uploads/"):
        local_path = os.path.join(uploads_dir, data.image_path[len("/uploads/"):])
    elif data.image_path.startswith("uploads/"):
        local_path = os.path.abspath(data.image_path)
    else:
        local_path = os.path.join(uploads_dir, data.image_path)

    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Original image not found")

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

    try:
        result = await _generate_inpainting_preview(local_path, valid_regions, hf_token)
        if result.get("success"):
            logger.info("generate_preview: HuggingFace InferenceClient succeeded")
            return result
    except Exception as e:
        logger.warning(f"generate_preview: HuggingFace InferenceClient failed: {e}")

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