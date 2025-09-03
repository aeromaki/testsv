from fastapi import APIRouter, Depends, UploadFile, File
from glue import get_current_user, send_message, SendSMSRequest
from models import User
from typing import Annotated
from core.config import SOLAPI_NUMBER


router = APIRouter(prefix="/api", tags=["api"])


@router.post('/analyze')
def upload_webm(
    file: Annotated[UploadFile, File(description='WebM')],
    user: Annotated[User, Depends(get_current_user)]
):
    return {
        "pitch": 85,
        "rhythm": 75,
        "emotion": 3,
        "total": 75,
        "content": "피치가 제법 정확합니다."
    }


@router.post('/sendsms')
def send_sms(
    request: SendSMSRequest,
    user: Annotated[User, Depends(get_current_user)]
):
    send_message(
        request,
        SOLAPI_NUMBER #user.phone_number
    )