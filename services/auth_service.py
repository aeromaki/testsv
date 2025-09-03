from sqlalchemy.orm import Session
from core.security import verify_password
from repositories.user_repo import get_user_by_email
from models import User

def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user