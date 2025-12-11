import os
from typing import Any

from langchain_openai import ChatOpenAI
# You can add other providers under this factory if you want.


def get_llm() -> Any:
    """
    Return a chat model instance based on environment configuration.

    Environment variables:
      LLM_PROVIDER: "openai" (default) or another provider you add
      LLM_MODEL: model name (for example "gpt-5-nano")
      LLM_TEMPERATURE: float (default 0.2)
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL", "gpt-5-nano")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    if provider == "openai":
        # For GPT-5 models, set reasoning_effort to minimal for faster responses
        # GPT-5 defaults to "medium" reasoning which is very slow
        model_config = {
            "model": model,
            "temperature": temperature
        }
        
        if "gpt-5" in model.lower():
            model_config["model_kwargs"] = {"reasoning_effort": "minimal"}
        
        return ChatOpenAI(**model_config)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
