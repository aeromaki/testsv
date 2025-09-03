from fastapi import APIRouter, Depends, UploadFile, File
from glue import get_current_user, send_message, SendSMSRequest
from models import User
from typing import Annotated
from core.config import SOLAPI_NUMBER

from concurrent.futures import ProcessPoolExecutor
import asyncio
from glue.analyze import analyze
import tempfile
import os

executor = ProcessPoolExecutor(max_workers=4)


router = APIRouter(prefix="/api", tags=["api"])


@router.post('/analyze')
async def upload_webm(
    file: Annotated[UploadFile, File(description='WebM')],
    user: Annotated[User, Depends(get_current_user)]
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, analyze, tmp_path)

    os.remove(tmp_path)
    return result


@router.post('/sendsms')
def send_sms(
    request: SendSMSRequest,
    user: Annotated[User, Depends(get_current_user)]
):
    send_message(
        request,
        SOLAPI_NUMBER #user.phone_number
    )