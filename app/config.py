from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Chatwoot
    chatwoot_base_url: str = "http://localhost:3000"
    chatwoot_api_access_token: str = ""
    chatwoot_account_id: int = 1
    chatwoot_hmac_secret: str = ""

    # LLM
    llm_provider: str = "openai_compatible"  # "anthropic" or "openai_compatible" (Groq)
    anthropic_api_key: str = ""
    agent_model: str = "claude-sonnet-4-6"
    openai_compatible_base_url: str = "https://api.groq.com/openai/v1"
    openai_compatible_api_key: str = ""
    openai_compatible_model: str = "llama-3.3-70b-versatile"

    # RAG
    chroma_persist_dir: str = "/data/chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 4

    class Config:
        env_file = ".env"


settings = Settings()