from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    db_path: str = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "..", "e2m.db"))
    jwt_secret: str = os.environ.get("JWT_SECRET", "e2m-renovation-secret-key-change-in-production")
    jwt_algorithm: str = os.environ.get("JWT_ALGORITHM", "HS256")
    jwt_expire_minutes: int = int(os.environ.get("JWT_EXPIRE_MINUTES", "1440"))
    upload_dir: str = os.environ.get("UPLOAD_DIR", "uploads")
    output_dir: str = os.environ.get("OUTPUT_DIR", "outputs")
    replicate_api_token: str = os.environ.get("REPLICATE_API_TOKEN", "")
    hf_api_token: str = os.environ.get("HF_API_TOKEN", "")
    cors_origins: str = os.environ.get(
        "CORS_ORIGINS",
        "https://arch-frontend-133093946118.europe-west1.run.app,http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"
    )

    model_config = {
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
