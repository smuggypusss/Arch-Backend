import logging
import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from ..middleware.auth import get_current_user
from ..services.image_service import validate_image, resize_image
from ..config import get_settings

router = APIRouter(prefix="/api/upload", tags=["Upload"])
settings = get_settings()
logger = logging.getLogger("e2m.upload")


@router.post("")
async def upload_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    logger.info(f"upload_image: user_id={current_user['id']}, filename={file.filename}")
    ext = os.path.splitext(file.filename or "image.jpg")[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        logger.warning(f"upload_image: unsupported format {ext}")
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    filename = f"{uuid.uuid4().hex}{ext}"
    user_dir = os.path.join(settings.upload_dir, str(current_user["id"]))
    os.makedirs(user_dir, exist_ok=True)
    file_path = os.path.join(user_dir, filename)
    logger.info(f"upload_image: saving to {file_path}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save file")

    is_valid, message = validate_image(file_path)
    if not is_valid:
        logger.warning(f"upload_image: validation failed: {message}")
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=message)

    optimized_path = resize_image(file_path)
    logger.info(f"upload_image: optimized to {optimized_path}")

    # Return relative path including user_id subdirectory so frontend can serve it
    relative_path = os.path.join(str(current_user["id"]), os.path.basename(optimized_path))
    logger.info(f"upload_image: success, relative_path={relative_path}")

    return {
        "success": True,
        "filename": relative_path,
        "path": optimized_path,
        "message": message,
    }
