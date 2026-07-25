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

# Models to try for image-to-image generation (used for material synthesis).
# These models support the image-to-image task on HuggingFace Inference API.
T2I_MODELS = [
    "black-forest-labs/FLUX.1-Kontext-dev",
]

NEGATIVE_PROMPT = (
    "blurry, distorted, cartoon, illustration, low quality, inconsistent lighting, "
    "deformed, extra limbs, disfigured, text, signature, watermark, ugly, morbid, "
    "mutilated, disfigured hands, poorly drawn hands, poorly drawn face"
)


def _build_material_prompt(region_type: str, material: str) -> str:
    region_desc = REGION_PROMPT_MAP.get(region_type, region_type)
    return (
        f"An ultra-photorealistic, high-end professional architectural photograph of building exterior {region_desc} "
        f"renovated with premium, beautifully textured {material}. "
        f"Seamless design integration, realistic texture detailing, crisp material seams, and natural shadows. "
        f"8k resolution, modern facade design, perfect daylight lighting, crisp detail, realistic depth."
    )


def _load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def _load_mask(path: str) -> Image.Image:
    mask = Image.open(path).convert("L")
    return mask


def _composite_region(original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    """Composite a generated material image onto the original using a mask.

    The mask defines which pixels to replace (white = replace, black = keep).
    The generated image is resized to match the original and blended using
    a soft mask to avoid hard edges.
    """
    # Resize generated image to match original
    gen_resized = generated.resize(original.size, Image.LANCZOS)

    # Apply Gaussian blur to the mask for soft edges
    mask_blurred = mask.filter(ImageFilter.GaussianBlur(radius=3))

    # Composite: where mask is white, use generated; where black, use original
    result = Image.composite(gen_resized, original, mask_blurred)
    return result


# Module-level cache for the Mage-Flow-Edit-Turbo pipeline (local inference)
_mage_pipe = None


def _get_mage_pipe(hf_token: str):
    """Load and cache the microsoft/Mage-Flow-Edit-Turbo DiffusionPipeline.

    Returns the cached pipeline if already loaded, or None if loading fails.
    The pipeline is loaded once and reused across all generation calls to
    avoid the overhead of re-downloading model weights.
    """
    global _mage_pipe
    if _mage_pipe is not None:
        return _mage_pipe

    try:
        import torch
        from diffusers import DiffusionPipeline

        logger.info("_get_mage_pipe: loading microsoft/Mage-Flow-Edit-Turbo...")
        _mage_pipe = DiffusionPipeline.from_pretrained(
            "microsoft/Mage-Flow-Edit-Turbo",
            torch_dtype=torch.float32,
            token=hf_token,
        )
        logger.info("_get_mage_pipe: pipeline loaded successfully")
        return _mage_pipe
    except Exception as e:
        logger.warning(f"_get_mage_pipe: failed to load pipeline: {e}")
        _mage_pipe = None
        return None


async def _generate_material_image(
    image: Image.Image,
    prompt: str,
    model_name: str,
    hf_token: str = None,
    client=None,
) -> Optional[Image.Image]:
    """Generate a modified image using image-to-image.

    First tries local inference with diffusers DiffusionPipeline
    (for microsoft/Mage-Flow-Edit-Turbo), then falls back to
    HuggingFace InferenceClient. The result preserves the original
    structure, perspective, and lighting while changing only the
    materials as described in the prompt.
    """
    import asyncio

    # --- Local diffusers inference (primary method) ---
    if model_name == "microsoft/Mage-Flow-Edit-Turbo":
        pipe = _get_mage_pipe(hf_token) if hf_token else None
        if pipe is not None:
            def _call_diffusers():
                try:
                    result = pipe(
                        image=image,
                        prompt=prompt,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                    )
                    if hasattr(result, "images") and result.images:
                        return result.images[0]
                    elif isinstance(result, list) and len(result) > 0:
                        return result[0]
                    return result
                except StopIteration:
                    raise RuntimeError("StopIteration raised by diffusers pipeline")

            try:
                generated = await asyncio.to_thread(_call_diffusers)
                if generated is not None:
                    logger.info(f"Local diffusers inference succeeded for {model_name}")
                    return generated
            except Exception as e:
                logger.warning(f"Local diffusers inference failed for {model_name}: {e}")
                # Reset cached pipe so it can be retried on next call
                _mage_pipe = None

    # --- Fallback: HuggingFace InferenceClient ---
    if client is None and hf_token:
        from huggingface_hub import InferenceClient
        client = InferenceClient(provider="fal-ai", api_key=hf_token, timeout=180)

    if client is None:
        logger.warning(f"No client or token available for model {model_name}")
        return None

    def _call_image_to_image():
        try:
            return client.image_to_image(
                image=image,
                prompt=prompt,
                model=model_name,
                guidance_scale=7.5,
                num_inference_steps=30,
            )
        except StopIteration:
            # StopIteration can be raised by generators inside the HuggingFace
            # client when used with asyncio.to_thread. Convert to RuntimeError
            # so it doesn't break the asyncio event loop.
            raise RuntimeError("StopIteration raised by image_to_image")

    try:
        result = await asyncio.to_thread(_call_image_to_image)
        return result
    except Exception as e:
        logger.warning(f"Model {model_name} failed: {e}")
        return None


async def _generate_inpainting_preview(
    image_path: str,
    regions: List[RegionConfig],
    hf_token: str,
) -> dict:
    """Generate an inpainting preview using huggingface_hub.InferenceClient.

    For each region, a binary mask is generated from the polygon. The current
    image is modified using image_to_image with a prompt describing the desired
    material, then composited back onto the original using the mask. This
    preserves the original geometry, lighting, and perspective while changing
    only the masked materials.
    """
    from huggingface_hub import InferenceClient
    from PIL import ImageFilter

    # Sort regions by polygon vertex count (proxy for area)
    sorted_regions = sorted(regions, key=lambda r: len(r.polygon), reverse=True)

    # Load the original image
    original_image = _load_image(image_path)
    current_image = original_image

    # Create the InferenceClient once and reuse.
    # Use fal-ai provider as configured for black-forest-labs/FLUX.1-Kontext-dev
    client = InferenceClient(
        provider="fal-ai",
        api_key=hf_token,
        timeout=180,
    )

    outputs_dir = os.path.abspath(settings.output_dir)
    os.makedirs(outputs_dir, exist_ok=True)

    regions_succeeded = 0
    regions_failed = 0

    for region in sorted_regions:
        if not region.polygon or len(region.polygon) < 3:
            logger.warning(f"Skipping region {region.type}: no valid polygon")
            regions_failed += 1
            continue

        # Generate mask from polygon
        mask_path = generate_mask_for_region(
            image_path=image_path,
            region_polygon=region.polygon,
            region_id=f"preview_{region.type}_{id(region)}",
        )
        mask_image = _load_mask(mask_path)

        prompt = _build_material_prompt(region.type, region.selected_material)
        logger.info(f"Generating material for region '{region.type}' with '{region.selected_material}'")

        # Try each image-to-image model in order.
        # _generate_material_image returns None on failure (exceptions are
        # caught internally), so we track success via the return value.
        region_success = False
        for model_name in T2I_MODELS:
            generated = await _generate_material_image(current_image, prompt, model_name, hf_token=hf_token, client=client)

            if generated is not None:
                # Composite the generated material onto the current image
                current_image = _composite_region(current_image, generated, mask_image)
                logger.info(f"Successfully applied material for region '{region.type}' with model {model_name}")
                region_success = True
                break

        if region_success:
            regions_succeeded += 1
        else:
            regions_failed += 1
            logger.error(f"All models failed for region '{region.type}'")

    # Only return success if at least one region was actually modified.
    # Returning success with the unmodified original image is misleading.
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
