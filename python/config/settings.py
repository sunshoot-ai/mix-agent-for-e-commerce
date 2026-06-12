from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Multi-Agent E-Commerce System"
    debug: bool = False

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen3.7-plus"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    feature_ttl_seconds: int = 86400

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "product_embeddings"

    # Database
    database_url: str = "sqlite:///./ecommerce.db"

    # A/B Testing
    ab_test_enabled: bool = True
    ab_test_default_bucket_count: int = 100

    # Agent timeouts (seconds)
    agent_timeout_user_profile: float = 5.0
    agent_timeout_product_rec: float = 8.0
    agent_timeout_marketing_copy: float = 10.0
    agent_timeout_inventory: float = 5.0

    # Alert thresholds
    alert_workflow_latency_warning_ms: float = 3000.0
    alert_workflow_latency_critical_ms: float = 5000.0
    alert_agent_latency_warning_ms: float = 1500.0
    alert_agent_latency_critical_ms: float = 3000.0
    alert_agent_fallback_warning_count: float = 1.0
    alert_agent_fallback_critical_count: float = 5.0
    alert_llm_tokens_warning_count: float = 8000.0
    alert_llm_tokens_critical_count: float = 20000.0
    alert_inventory_critical_warning_count: float = 1.0
    alert_inventory_critical_critical_count: float = 3.0
    alert_webhook_url: str = ""
    alert_webhook_timeout_seconds: float = 2.0

    # Resolve .env relative to the python package directory so running from
    # repo root (`uvicorn python.main:app`) still loads `python/.env`.
    model_config = {
        "env_file": str(Path(__file__).resolve().parents[1] / ".env"),
        "env_prefix": "ECOM_",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
