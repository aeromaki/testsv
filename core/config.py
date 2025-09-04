from dotenv import load_dotenv
import os
from datetime import timedelta

if not load_dotenv():
    raise Exception("dotenv file not found")

PORT: int = int(os.environ["PORT"])
SECRET_KEY: str = os.environ["SECRET_KEY"]
DB_URL: str = os.environ["DB_URL"]

ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE: timedelta = timedelta(days=7)

SOLAPI_KEY: str = os.environ['SOLAPI_KEY']
SOLAPI_SECRET: str = os.environ['SOLAPI_SECRET']
SOLAPI_NUMBER: str = os.environ['SOLAPI_NUMBER']

SR = 8000