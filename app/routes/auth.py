import logging
from fastapi import APIRouter, Depends, HTTPException, status
import aiosqlite
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta

from ..models.user import UserCreate, UserLogin, TokenResponse, UserResponse
from ..config import get_settings
from ..database.connection import get_db
from ..middleware.auth import get_current_user

router = APIRouter(prefix="/api/auth", tags=["Auth"])
import hashlib

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
settings = get_settings()
logger = logging.getLogger("e2m.auth")


def _hash_password(password: str) -> str:
    """Pre-hash with SHA-256 to bypass bcrypt's 72-byte limit, then bcrypt."""
    sha256_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return pwd_context.hash(sha256_hash)


def _verify_password(password: str, hashed: str) -> bool:
    sha256_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return pwd_context.verify(sha256_hash, hashed)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: aiosqlite.Connection = Depends(get_db)):
    logger.info(f"register: email={user_data.email}, name={user_data.name}")
    async with db.execute("SELECT id FROM users WHERE email = ?", (user_data.email,)) as cursor:
        if await cursor.fetchone():
            logger.warning(f"register: email already registered: {user_data.email}")
            raise HTTPException(status_code=400, detail="Email already registered")

    hashed = _hash_password(user_data.password)
    async with db.execute(
        "INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
        (user_data.name, user_data.email, hashed, user_data.role),
    ) as cursor:
        user_id = cursor.lastrowid
    await db.commit()

    async with db.execute("SELECT id, name, email, role, created_at FROM users WHERE id = ?", (user_id,)) as cursor:
        row = await cursor.fetchone()

    user = dict(row)
    token = create_access_token({"sub": str(user["id"])})
    logger.info(f"register: success, user_id={user_id}")
    return TokenResponse(
        access_token=token,
        user=UserResponse(**user),
    )


@router.post("/login")
async def login(credentials: UserLogin, db: aiosqlite.Connection = Depends(get_db)):
    logger.info(f"login: email={credentials.email}")
    async with db.execute("SELECT * FROM users WHERE email = ?", (credentials.email,)) as cursor:
        row = await cursor.fetchone()

    if not row or not _verify_password(credentials.password, row["password"]):
        logger.warning(f"login: failed for email={credentials.email}")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user = dict(row)
    user.pop("password", None)
    token = create_access_token({"sub": str(user["id"])})
    logger.info(f"login: success, user_id={user['id']}")
    return TokenResponse(access_token=token, user=UserResponse(**user))


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(**current_user)