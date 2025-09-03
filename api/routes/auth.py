from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from db.session import get_db
from schemas.auth import Token, UserLoginResponse
from services.auth_service import authenticate_user
from core.security import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/token", response_model=UserLoginResponse, summary="로그인 후 액세스 토큰 발급")
def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[Session, Depends(get_db)],
) -> UserLoginResponse:
    email = form.username
    password = form.password

    user = authenticate_user(db, email, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(subject=user.email)
    print(token)
    return UserLoginResponse(
        token=token, #Token(access_token=token).,
        email=user.email,
        userName=user.username
    )
