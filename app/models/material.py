from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from datetime import datetime


class MaterialBase(BaseModel):
    name: str = Field(..., min_length=2)
    category: Literal["paint", "cladding", "tile", "texture", "railing", "panel"]
    applicable_regions: List[Literal["wall", "window", "balcony", "pillar", "parapet", "gate", "roof"]] = []
    image: Optional[str] = None
    coverage_value: float
    coverage_unit: str = "sqft"
    material_rate: float
    labor_rate: float
    wastage_percent: float = 5.0
    durability: str = "Standard"
    maintenance: str = "Low"
    description: str = ""


class MaterialCreate(MaterialBase):
    pass


class MaterialResponse(MaterialBase):
    id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MaterialUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    applicable_regions: Optional[List[str]] = None
    image: Optional[str] = None
    coverage_value: Optional[float] = None
    coverage_unit: Optional[str] = None
    material_rate: Optional[float] = None
    labor_rate: Optional[float] = None
    wastage_percent: Optional[float] = None
    durability: Optional[str] = None
    maintenance: Optional[str] = None
    description: Optional[str] = None