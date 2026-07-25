import json
import asyncio
import aiosqlite
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.config import get_settings

settings = get_settings()
DB_PATH = os.path.abspath(settings.db_path)


async def seed():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row

    await db.executescript("""
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            applicable_regions TEXT DEFAULT '[]',
            image TEXT,
            coverage_value REAL,
            coverage_unit TEXT DEFAULT 'sqft',
            material_rate REAL,
            labor_rate REAL,
            wastage_percent REAL DEFAULT 5.0,
            durability TEXT DEFAULT 'Standard',
            maintenance TEXT DEFAULT 'Low',
            description TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    async with db.execute("SELECT COUNT(*) as cnt FROM materials") as cursor:
        row = await cursor.fetchone()
    if row["cnt"] > 0:
        async with db.execute("SELECT applicable_regions FROM materials LIMIT 1") as cursor:
            sample = await cursor.fetchone()
        if sample and sample["applicable_regions"]:
            try:
                ar = json.loads(sample["applicable_regions"])
                if isinstance(ar, list) and len(ar) > 0:
                    print(f"Database already has {row['cnt']} materials with applicable_regions. Skipping seed.")
                    await db.close()
                    return
            except json.JSONDecodeError:
                pass
        print("Existing materials missing applicable_regions. Reseeding...")
        await db.execute("DELETE FROM materials")

    seed_path = os.path.join(os.path.dirname(__file__), "materials.json")
    with open(seed_path, "r") as f:
        materials = json.load(f)

    for mat in materials:
        await db.execute(
            "INSERT INTO materials (name, category, applicable_regions, image, coverage_value, coverage_unit, material_rate, labor_rate, wastage_percent, durability, maintenance, description) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                mat["name"], mat["category"], json.dumps(mat.get("applicable_regions", [])),
                mat.get("image"),
                mat["coverage_value"], mat["coverage_unit"],
                mat["material_rate"], mat["labor_rate"],
                mat.get("wastage_percent", 5.0),
                mat.get("durability", "Standard"),
                mat.get("maintenance", "Low"),
                mat.get("description", ""),
            ),
        )

    await db.commit()
    print(f"Seeded {len(materials)} materials successfully.")
    await db.close()


if __name__ == "__main__":
    asyncio.run(seed())