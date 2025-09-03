from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from api.routes import *

from core.config import PORT, SOLAPI_NUMBER

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(api_router)
app.mount('/', StaticFiles(directory='dist', html=True))


from sqlalchemy import select
from db.session import engine, SessionLocal
from models.base import Base
from models.user import User
from core.security import hash_password

@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        exists = db.scalar(select(User).where(User.admin))
        if not exists:
            db.add(User(
                username='admin',
                email='admin',
                phone_number=SOLAPI_NUMBER,
                hashed_password=hash_password('admin12345'),
                admin=True
            ))
            db.commit()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

