from typing import List, Optional
import aiosqlite
from ..models.project import Region, MaterialCost, LaborCost, CostEstimate


async def calculate_cost(
    db: aiosqlite.Connection,
    regions: List[Region],
    area_map: dict,
    custom_material_rates: Optional[dict] = None,
    custom_labor_rates: Optional[dict] = None,
) -> CostEstimate:
    material_costs: List[MaterialCost] = []
    labor_costs: List[LaborCost] = []
    total_wastage = 0.0
    grand_total = 0.0

    for region in regions:
        region_area = area_map.get(region.id, {}).get("area_sqft", 0)
        if region_area <= 0 or not region.selected_material:
            continue

        async with db.execute(
            "SELECT * FROM materials WHERE id = ?",
            (int(region.selected_material),),
        ) as cursor:
            material = await cursor.fetchone()

        if not material:
            continue

        mat = dict(material)
        mat_rate = mat["material_rate"]
        lab_rate = mat["labor_rate"]
        wastage_pct = mat.get("wastage_percent", 5.0)

        if custom_material_rates and region.selected_material in custom_material_rates:
            mat_rate = custom_material_rates[region.selected_material]
        if custom_labor_rates and region.selected_material in custom_labor_rates:
            lab_rate = custom_labor_rates[region.selected_material]

        coverage = mat.get("coverage_value", 1)
        quantity = region_area / coverage if coverage > 0 else region_area
        wastage_qty = quantity * (wastage_pct / 100)
        total_qty = quantity + wastage_qty

        mat_total = round(total_qty * mat_rate, 2)
        lab_total = round(region_area * lab_rate, 2)
        wastage_cost = round(wastage_qty * mat_rate, 2)

        material_costs.append(
            MaterialCost(
                material_id=str(mat["id"]),
                name=mat["name"],
                quantity=round(total_qty, 2),
                unit=mat.get("coverage_unit", "sqft"),
                rate=mat_rate,
                total=mat_total,
            )
        )

        labor_costs.append(
            LaborCost(
                category=f"{region.type} - {mat['name']}",
                rate=lab_rate,
                total=lab_total,
            )
        )

        total_wastage += wastage_cost
        grand_total += mat_total + lab_total

    grand_total = round(grand_total, 2)
    total_wastage = round(total_wastage, 2)

    return CostEstimate(
        materials=material_costs,
        labor=labor_costs,
        wastage=total_wastage,
        grand_total=grand_total,
    )