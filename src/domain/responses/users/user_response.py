from datetime import datetime
from uuid import UUID
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class UserResponse(BaseModel):
    """
    Modelo de resposta para dados de usuário.
    """
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "email": "usuario@exemplo.com",
                "user_type": "user",
                "active": True,
                "email_verified": False,
                "last_login_at": None,
                "created_at": "2026-01-10T10:00:00",
                "updated_at": "2026-01-10T10:00:00"
            }
        }
    )
    
    id: int = Field(
        ...,
        description="ID do usuário",
        examples=[1]
    )
    
    uuid: UUID = Field(
        ...,
        description="UUID do usuário",
        examples=["123e4567-e89b-12d3-a456-426614174000"]
    )
    
    email: str = Field(
        ...,
        description="Email do usuário",
        examples=["usuario@exemplo.com"]
    )
    
    user_type: str = Field(
        ...,
        description="Tipo de usuário",
        examples=["user", "admin"]
    )
    
    active: bool = Field(
        ...,
        description="Se o usuário está ativo",
        examples=[True]
    )
    
    email_verified: bool = Field(
        ...,
        description="Se o email foi verificado",
        examples=[False]
    )
    
    last_login_at: Optional[datetime] = Field(
        default=None,
        description="Data e hora do último login",
        examples=["2026-01-10T10:30:00"]
    )
    
    created_at: datetime = Field(
        ...,
        description="Data e hora de criação",
        examples=["2026-01-10T10:00:00"]
    )
    
    updated_at: datetime = Field(
        ...,
        description="Data e hora da última atualização",
        examples=["2026-01-10T10:00:00"]
    )
