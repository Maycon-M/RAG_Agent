from pydantic import BaseModel, EmailStr, Field, field_validator

class StudentCreateRequest(BaseModel):
    """
    Modelo de requisição para criação de aluno.
    """
    
    full_name: str = Field(
        ...,
        min_length=1,
        max_length=250,
        description="Nome completo do aluno",
        examples=["João Silva"]
    )
    
    email: EmailStr | None = Field(
        default=None,
        description="Email do aluno",
        examples=["aluno@exemplo.com"]
    )
    
    
    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v: str) -> str:
        """
        Valida o nome completo do aluno.
        
        Args:
            v: Nome completo a ser validado
            
        Returns:
            str: Nome completo validado
            
        Raises:
            ValueError: Se o nome completo não atender aos requisitos
        """
        if not v.replace(" ", "").isalpha():
            raise ValueError("O nome completo deve conter apenas letras e espaços.")
        return v
