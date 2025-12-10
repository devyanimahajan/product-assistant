import os
from typing import Any

from langchain_openai import ChatOpenAI
# You can add other providers under this factory if you want.


def get_llm() -> Any:
    """
    Return a chat model instance based on environment configuration.

    Environment variables:
      LLM_PROVIDER: "openai" (default) or another provider you add
      LLM_MODEL: model name (for example "gpt-4o-mini")
      LLM_TEMPERATURE: float (default 0.2)
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
