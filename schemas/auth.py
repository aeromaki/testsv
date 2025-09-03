from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserLoginResponse(BaseModel):
    token: str #Token
    email: str
    userName: str