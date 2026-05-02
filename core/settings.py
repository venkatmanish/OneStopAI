import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:
    from pydantic import BaseModel as BaseSettings

    def SettingsConfigDict(**kwargs):  # type: ignore
        return kwargs


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")
    groq_analysis_model: str = Field(
        default="openai/gpt-oss-120b",
        alias="GROQ_ANALYSIS_MODEL",
    )
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1", alias="OLLAMA_MODEL")

    database_url: str = Field(
        default="postgresql+psycopg://rag_user:rag_password@localhost:5432/agentic_rag",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="agentic_rag_chunks", alias="QDRANT_COLLECTION")
    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        alias="EMBEDDING_MODEL",
    )
    reranker_model: str | None = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL",
    )

    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    openweather_api_key: str | None = Field(default=None, alias="OPENWEATHER_API_KEY")
    local_storage_dir: Path = Field(default=Path(".local_storage"), alias="LOCAL_STORAGE_DIR")

    backend_url: str = Field(default="http://localhost:8000", alias="BACKEND_URL")
    google_drive_folder_id: str | None = Field(default=None, alias="GOOGLE_DRIVE_FOLDER_ID")
    google_service_account_file: str | None = Field(
        default=None,
        alias="GOOGLE_SERVICE_ACCOUNT_FILE",
    )
    google_drive_shared_drive_id: str | None = Field(
        default=None,
        alias="GOOGLE_DRIVE_SHARED_DRIVE_ID",
    )

    def __init__(self, **data):
        if BaseSettings.__name__ == "BaseModel":
            for field_name, field in self.__class__.model_fields.items():
                alias = field.alias or field_name
                if alias in os.environ and field_name not in data:
                    data[field_name] = os.environ[alias]
        super().__init__(**data)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.local_storage_dir.mkdir(parents=True, exist_ok=True)
    (settings.local_storage_dir / "uploads").mkdir(parents=True, exist_ok=True)
    return settings
