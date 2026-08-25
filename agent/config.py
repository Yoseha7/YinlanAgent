"""Mineradio AI Agent — Configuration via Pydantic BaseSettings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# .env 路径相对于 config.py 自身，不受工作目录影响
_ENV_FILE = str(Path(__file__).parent / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Provider ---
    llm_provider: str = Field(default="deepseek", alias="LLM_PROVIDER")

    # --- DeepSeek (OpenAI-compatible) ---
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_api_base: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_API_BASE")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    # --- OpenAI ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_api_base: str = Field(default="https://api.openai.com/v1", alias="OPENAI_API_BASE")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    # --- Ollama ---
    ollama_api_base: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_API_BASE")
    ollama_model: str = Field(default="qwen2.5:7b", alias="OLLAMA_MODEL")

    # --- Web Search (Tavily) ---
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    """Tavily AI Search API Key (https://tavily.com)"""
    search_max_results: int = Field(default=5, alias="SEARCH_MAX_RESULTS")
    """每次搜索默认返回结果数"""

    # --- General ---
    node_server_base: str = Field(default="http://127.0.0.1:3000", alias="NODE_SERVER_BASE")
    agent_port: int = Field(default=3001, alias="AGENT_PORT")

    @property
    def effective_model(self) -> str:
        """Return the model name for the active provider."""
        if self.llm_provider == "deepseek":
            return self.deepseek_model
        elif self.llm_provider == "openai":
            return self.openai_model
        elif self.llm_provider == "ollama":
            return self.ollama_model
        return "unknown"

    @property
    def active_api_key(self) -> str:
        if self.llm_provider == "deepseek":
            return self.deepseek_api_key
        elif self.llm_provider == "openai":
            return self.openai_api_key
        return ""

    @property
    def active_api_base(self) -> str:
        if self.llm_provider == "deepseek":
            return self.deepseek_api_base
        elif self.llm_provider == "openai":
            return self.openai_api_base
        elif self.llm_provider == "ollama":
            return self.ollama_api_base
        return ""


settings = Settings()
