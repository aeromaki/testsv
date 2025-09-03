from sqlalchemy.orm import Session
from models import User
from fastapi import Depends, HTTPException
from db.session import get_db
from models import User
from core.security import decode_access_token, oauth2_scheme
from typing import Annotated, Optional


def get_current_user(
    token: Annotated[Optional[str], Depends(oauth2_scheme)],
    db: Annotated[Session, Depends(get_db)]
) -> User:
    if token is None:
        raise HTTPException(status_code=401, detail='Invalid authentication credentials')

    payload = decode_access_token(token)
    email = payload.get('sub')

    if email is None:
        raise HTTPException(status_code=401, detail='Invalid authentication credentials')

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if not user.admin:
        raise HTTPException(status_code=403, detail='Admin privileges required')
    return user





from solapi import SolapiMessageService
from solapi.model import RequestMessage
from core.config import SOLAPI_KEY, SOLAPI_SECRET
from pydantic import BaseModel


message_service = SolapiMessageService(
api_key=SOLAPI_KEY,
api_secret=SOLAPI_SECRET
)

class SendSMSRequest(BaseModel):
    to: str
    content: str


def send_message(request: SendSMSRequest, from_: str):
    message = RequestMessage(
        subject='★ AI 전화노래방 분석 결과 ★',
        from_=from_,
        to=request.to,
        text=request.content
    )
    try:
        response = message_service.send(message)
    except Exception as e:
        raise HTTPException(status_code=400, detail='메시지 전송에 실패했습니다.')
