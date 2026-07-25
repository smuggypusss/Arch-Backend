import os
import uuid
from PIL import Image
import io
from ..config import get_settings

settings = get_settings()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_DIMENSIONS = (480, 320)


def validate_image(file_path: str) -> tuple[bool, str]:
    """Validate an uploaded image for quality and dimensions."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type {ext} not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}"

    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large ({file_size / 1024 / 1024:.1f}MB). Max: 10MB"

    try:
        with Image.open(file_path) as img:
            width, height = img.size
            if width < MIN_DIMENSIONS[0] or height < MIN_DIMENSIONS[1]:
                return False, f"Image too small ({width}x{height}). Minimum: 480x320"

            if width / height > 4 or height / width > 4:
                return False, "Image aspect ratio too extreme. Use a standard photo."

        return True, "Image valid"
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def resize_image(input_path: str, max_dimension: int = 1920) -> str:
    """Resize image if too large, preserving aspect ratio."""
    save_dir = os.path.join(settings.upload_dir, "resized")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"resized_{uuid.uuid4().hex}{os.path.splitext(input_path)[1]}"
    output_path = os.path.join(save_dir, filename)

    try:
        with Image.open(input_path) as img:
            width, height = img.size
            if max(width, height) <= max_dimension:
                return input_path

            ratio = max_dimension / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            img_resized = img.resize(new_size, Image.LANCZOS)
            img_resized.save(output_path, optimize=True, quality=85)

        return output_path
    except Exception:
        return input_path


def generate_mask_for_region(
    image_path: str,
    region_polygon: list,
    region_id: str,
) -> str:
    """Generate a binary mask image for a given polygon region."""
    os.makedirs(os.path.join(settings.output_dir, "masks"), exist_ok=True)
    mask_path = os.path.join(settings.output_dir, "masks", f"mask_{region_id}.png")

    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            mask = Image.new("L", (img_width, img_height), 0)

            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)

            polygon_tuples = [(p["x"], p["y"]) for p in region_polygon]
            if len(polygon_tuples) >= 3:
                draw.polygon(polygon_tuples, fill=255)
            else:
                min_x = min(p["x"] for p in region_polygon)
                min_y = min(p["y"] for p in region_polygon)
                max_x = max(p["x"] for p in region_polygon)
                max_y = max(p["y"] for p in region_polygon)
                draw.rectangle([min_x, min_y, max_x, max_y], fill=255)

            mask.save(mask_path)
        return mask_path
    except Exception as e:
        raise ValueError(f"Failed to generate mask: {str(e)}")