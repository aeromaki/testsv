from fastapi import APIRouter, Depends
from glue import require_admin
from models import User
from sqlalchemy.orm import Session
from sqlalchemy import select
from db.session import get_db
from typing import Annotated
from pydantic import BaseModel
from core.security import hash_password
from sqlalchemy.exc import NoResultFound



class User_(BaseModel):
    email: str
    userName: str
    phoneNumber: str

class NewUser(User_):
    password: str

class AddUserRequest(BaseModel):
    user: NewUser

class RemoveUserRequest(BaseModel):
    user: User_

class ChangePasswordRequest(BaseModel):
    password: str


router = APIRouter(prefix="/admin", tags=["admin"])


@router.post('/changepassword')
def change_password(
    request: ChangePasswordRequest,
    user: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    user_ = db.query(User).filter(User.admin == True).first()
    assert user_ is not None
    user_.hashed_password = hash_password(request.password)
    db.commit()
    db.refresh(user)


@router.get('/load')
def load_users(
    user: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    users = db.execute(select(User).where(User.admin != True)).scalars().all()
    return { 'users': [*map(lambda x: {
        'email': x.email,
        'phoneNumber': x.phone_number,
        'userName': x.username
    }, users)] }


@router.post('/add')
def add_user(
    request: AddUserRequest,
    user: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    user_info = request.user
    new_user = User(
        email=user_info.email,
        username=user_info.userName,
        phone_number=user_info.phoneNumber,
        hashed_password=hash_password(user_info.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)


@router.post('/remove')
def remove_user(
    request: RemoveUserRequest,
    user: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    user_ = db.query(User).filter(User.email == request.user.email).first()
    if user_ is None:
        raise NoResultFound('해당 유저는 존재하지 않습니다.')

    db.delete(user_)
    db.commit()
