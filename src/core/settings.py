from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Classe responsável por armazenar as configurações do sistema."""
    
    # === AMBIENTE ===
    ENV: str = Field(default="dev", description="Ambiente de execução")
    
    # === LOGS ===
    LOG_LEVEL: str = Field(default="INFO", description="Nível de log")
    LOG_JSON: bool = Field(default=False, description="Logs em formato JSON")
    
    # === CORS ===
    ALLOW_ORIGINS: list[str] = [
        "http://localhost:3000", 
        "http://localhost:5173", 
        "http://localhost:8000"
    ]
    ALLOW_METHODS: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CSP_REPORT_ONLY: bool = False
    
    # === DATABASE (POSTGRES/SUPABASE) ===
    SUPABASE_CONN_STRING: str = Field(..., description="Connection string do PostgreSQL/Supabase")
    
    # === OPENAI ===
    OPENAI_API_KEY: str = Field(..., description="Chave da API OpenAI")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", description="Modelo padrão")
    OPENAI_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    
    # === CHROMADB ===
    CHROMA_PERSIST_DIRECTORY: str = Field(default="./chroma_db", description="Diretório do ChromaDB")
    CHROMA_COLLECTION_NAME: str = Field(default="documents", description="Nome da coleção")
    
    @field_validator("ALLOW_ORIGINS")
    @classmethod
    def validate_cors_in_production(cls, v: list[str], info) -> list[str]:
        """Previne CORS aberto em produção."""
        if info.data.get("ENV") == "prd" and "*" in v:
            raise ValueError("CORS wildcard não permitido em produção")
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignora variáveis extras do .env
    )

settings = Settings()
