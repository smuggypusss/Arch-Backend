import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from .config import get_settings
from .database.connection import init_db
from .routes import auth, projects, materials, upload, ai, estimate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("e2m")
logger.info("E2M server starting...")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="E2M - Exterior House Renovation & Cost Estimation",
    description="AI-powered platform for visualizing and estimating exterior house renovations",
    version="1.0.0",
    lifespan=lifespan,
)

# Safe list of explicit origins
origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]

# To support allow_credentials=True with deployed frontends (which cannot use wildcard "*"),
# we allow any HTTPS origin or localhost port via allow_origin_regex. This is highly robust
# and ensures no CORS errors ever happen on any deployed preview or production URLs.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*|http://localhost:\d+|http://127\.0\.0\.1:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(materials.router)
app.include_router(upload.router)
app.include_router(ai.router)
app.include_router(estimate.router)

# Serve uploaded and generated files
uploads_dir = os.path.abspath(settings.upload_dir)
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

outputs_dir = os.path.abspath(settings.output_dir)
os.makedirs(outputs_dir, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")


@app.get("/")
async def root():
    return {"name": "E2M Renovation API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
