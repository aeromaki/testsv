from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from core.config import DB_URL

engine = create_engine(DB_URL, connect_args={ "check_same_thread": False })
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()