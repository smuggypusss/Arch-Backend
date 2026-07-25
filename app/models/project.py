from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime


class RegionPoint(BaseModel):
    x: float
    y: float


class Region(BaseModel):
    id: str
    type: Literal["wall", "window", "balcony", "pillar", "parapet", "gate", "roof"]
    polygon: List[RegionPoint] = []
    area_sqft: Optional[float] = None
    selected_material: Optional[str] = None
    notes: str = ""


class MaterialCost(BaseModel):
    material_id: str
    name: str
    quantity: float
    unit: str
    rate: float
    total: float


class LaborCost(BaseModel):
    category: str
    rate: float
    total: float


class CostEstimate(BaseModel):
    materials: List[MaterialCost] = []
    labor: List[LaborCost] = []
    wastage: float = 0
    grand_total: float = 0
    rates_editable: bool = True


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    original_image: str
    status: str = "draft"


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    regions: Optional[List[Region]] = None
    generated_image: Optional[str] = None
    cost_estimate: Optional[CostEstimate] = None
    status: Optional[Literal["draft", "regions_mapped", "materials_selected", "visualized", "completed"]] = None


class ProjectResponse(BaseModel):
    id: int
    user_id: str
    name: str
    original_image: str
    regions: List[Region] = []
    generated_image: Optional[str] = None
    cost_estimate: Optional[CostEstimate] = None
    status: str = "draft"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class EstimateRequest(BaseModel):
    regions: List[Region]
    reference_height_ft: float = 7.0
    reference_pixels: float = 100.0