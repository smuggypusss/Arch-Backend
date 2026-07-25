import json
from fastapi import APIRouter, Depends
import aiosqlite

from ..models.project import EstimateRequest
from ..services.estimation_service import estimate_all_regions
from ..services.cost_service import calculate_cost
from ..database.connection import get_db
from ..middleware.auth import get_current_user

router = APIRouter(prefix="/api/estimate", tags=["Estimation"])


@router.post("/calculate")
async def calculate_estimate(
    data: EstimateRequest,
    current_user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    area_map = estimate_all_regions(
        regions=data.regions,
        reference_height_ft=data.reference_height_ft,
        reference_pixels=data.reference_pixels,
    )

    cost_estimate = await calculate_cost(
        db=db,
        regions=data.regions,
        area_map=area_map,
    )

    return {
        "success": True,
        "areas": area_map,
        "cost": cost_estimate.model_dump(),
    }