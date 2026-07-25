import json
from fastapi import APIRouter, Depends, HTTPException
import aiosqlite
from typing import List

from ..models.material import MaterialCreate, MaterialResponse, MaterialUpdate
from ..database.connection import get_db
from ..middleware.auth import get_current_user

router = APIRouter(prefix="/api/materials", tags=["Materials"])


def row_to_material(row: aiosqlite.Row) -> dict:
    d = dict(row)
    d["applicable_regions"] = json.loads(d.get("applicable_regions", "[]"))
    return d


@router.get("", response_model=List[MaterialResponse])
async def get_materials(category: str = None, db: aiosqlite.Connection = Depends(get_db)):
    if category:
        query = "SELECT * FROM materials WHERE category = ? ORDER BY name"
        params = (category,)
    else:
        query = "SELECT * FROM materials ORDER BY name"
        params = ()

    async with db.execute(query, params) as cursor:
        rows = await cursor.fetchall()

    return [MaterialResponse(**row_to_material(r)) for r in rows]


@router.get("/{material_id}")
async def get_material(material_id: int, db: aiosqlite.Connection = Depends(get_db)):
    async with db.execute("SELECT * FROM materials WHERE id = ?", (material_id,)) as cursor:
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Material not found")
    return MaterialResponse(**row_to_material(row))


@router.post("", response_model=MaterialResponse, status_code=201)
async def create_material(
    data: MaterialCreate,
    current_user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    async with db.execute(
        "INSERT INTO materials (name, category, applicable_regions, image, coverage_value, coverage_unit, material_rate, labor_rate, wastage_percent, durability, maintenance, description) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (data.name, data.category, json.dumps(data.applicable_regions), data.image, data.coverage_value, data.coverage_unit, data.material_rate, data.labor_rate, data.wastage_percent, data.durability, data.maintenance, data.description),
    ) as cursor:
        mat_id = cursor.lastrowid
    await db.commit()

    async with db.execute("SELECT * FROM materials WHERE id = ?", (mat_id,)) as cursor:
        row = await cursor.fetchone()
    return MaterialResponse(**row_to_material(row))


@router.put("/{material_id}")
async def update_material(
    material_id: int,
    data: MaterialUpdate,
    current_user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    if "applicable_regions" in updates:
        updates["applicable_regions"] = json.dumps(updates["applicable_regions"])

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [material_id]
    await db.execute(f"UPDATE materials SET {set_clause} WHERE id = ?", values)
    await db.commit()

    async with db.execute("SELECT * FROM materials WHERE id = ?", (material_id,)) as cursor:
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Material not found")
    return MaterialResponse(**row_to_material(row))