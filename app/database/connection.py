import aiosqlite
import json
import os
from typing import AsyncGenerator
from ..config import get_settings

settings = get_settings()
DB_PATH = os.path.abspath(settings.db_path)
SEED_PATH = os.path.join(os.path.dirname(__file__), "..", "seed", "materials.json")


async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    try:
        yield db
    finally:
        await db.close()


async def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'homeowner',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

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

        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            original_image TEXT,
            regions TEXT DEFAULT '[]',
            generated_image TEXT,
            cost_estimate TEXT,
            status TEXT DEFAULT 'draft',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    # Auto-seed materials if empty or schema was just created
    async with db.execute("SELECT COUNT(*) as cnt FROM materials") as cursor:
        row = await cursor.fetchone()
    if row["cnt"] == 0 and os.path.exists(SEED_PATH):
        with open(SEED_PATH, "r") as f:
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
        print(f"Auto-seeded {len(materials)} materials on startup.")

    await db.commit()
    await db.close()
