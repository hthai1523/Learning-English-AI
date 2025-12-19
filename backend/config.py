"""
Configuration settings for the English Learning App backend
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model_name: str = "gpt-4o-mini"
    openai_tts_model: str = "tts-1"
    openai_tts_voice: str = "nova"  # Options: alloy, echo, fable, onyx, nova, shimmer

    # Gemini Configuration
    gemini_api_key: Optional[str] = None
    gemini_model_name: str = "gemini-pro"

    # AI Provider (openai | gemini)
    ai_provider: str = "gemini"

    # Server Configuration
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:5173"

    # Environment
    env: str = "development"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
