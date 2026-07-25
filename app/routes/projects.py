import json
from fastapi import APIRouter, Depends, HTTPException, status
import aiosqlite
from datetime import datetime
from typing import List

from ..models.project import ProjectCreate, ProjectUpdate, ProjectResponse
from ..database.connection import get_db
from ..middleware.auth import get_current_user

router = APIRouter(prefix="/api/projects", tags=["Projects"])


def row_to_project(row: dict) -> dict:
    """Convert SQLite row to project dict with parsed JSON fields."""
    d = dict(row)
    d["user_id"] = str(d["user_id"]) if d.get("user_id") else ""
    d["regions"] = json.loads(d.get("regions", "[]"))
    d["cost_estimate"] = json.loads(d.get("cost_estimate", "null")) if d.get("cost_estimate") else None
    return d


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project(
    data: ProjectCreate,
    current_user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    async with db.execute(
        "INSERT INTO projects (user_id, name, original_image, status) VALUES (?, ?, ?, ?)",
        (int(current_user["id"]), data.name, data.original_image, data.status),
    ) as cursor:
        pid = cursor.lastrowid
    await db.commit()

    async with db.execute("SELECT * FROM projects WHERE id = ?", (pid,)) as cursor:
        row = await cursor.fetchone()
    return ProjectResponse(**row_to_project(row))


@router.get("", response_model=List[ProjectResponse])
async def list_projects(
    current_user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    async with db.execute(
        "SELECT * FROM projects WHERE user_id = ? ORDER BY updated_at DESC",
        (int(current_user["id"]),),
    ) as cursor:
        rows = await cursor.fetchall()
    return [ProjectResponse(**row_to_project(r)) for r in rows]


@router.get("/{project_id}")
async def get_project(
    project_id: int,
    current_user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    async with db.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?",
        (project_id, int(current_user["id"])),
    ) as cursor:
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectResponse(**row_to_project(row))


@router.put("/{project_id}")
async def update_project(
    project_id: int,
    data: ProjectUpdate,
    current_user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    updates = {}
    if data.name is not None:
        updates["name"] = data.name
    if data.regions is not None:
        updates["regions"] = json.dumps([r.model_dump() for r in data.regions])
    if data.generated_image is not None:
        updates["generated_image"] = data.generated_image
    if data.cost_estimate is not None:
        updates["cost_estimate"] = json.dumps(data.cost_estimate.model_dump())
    if data.status is not None:
        updates["status"] = data.status

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    updates["updated_at"] = datetime.utcnow().isoformat()

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [project_id, int(current_user["id"])]
    cursor = await db.execute(
        f"UPDATE projects SET {set_clause} WHERE id = ? AND user_id = ?", values
    )
    await db.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Project not found")

    async with db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)) as cursor:
        row = await cursor.fetchone()
    return ProjectResponse(**row_to_project(row))


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: int,
    current_user: dict = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    cursor = await db.execute(
        "DELETE FROM projects WHERE id = ? AND user_id = ?",
        (project_id, int(current_user["id"])),
    )
    await db.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Project not found")