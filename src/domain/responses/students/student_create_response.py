from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict

class StudentCreateResponse(BaseModel):
    """
    Modelo de requisição para criação de aluno.
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "created_at": "2026-01-10T10:00:00"
            }
        }
    )
    
    uuid: UUID = Field(
        ...,
        description="UUID do aluno",
        examples=["123e4567-e89b-12d3-a456-426614174000"]
    )
    
    created_at: datetime = Field(
        ...,
        description="Data e hora de criação do aluno",
        examples=["2026-01-10T10:00:00"]
    )
