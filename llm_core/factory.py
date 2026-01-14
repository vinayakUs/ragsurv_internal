import os
from typing import Optional
from llm_core.interface import LLMProvider
from llm_core.providers.Ollama import OllamaProvider

class LLMFactory:
    """
    Factory class to create LLMProvider instances based on configuration.
    """

    @staticmethod
    def get_provider(provider_name: Optional[str] = None) -> LLMProvider:
        """
        Returns an instance of the requested LLM provider.
        If provider_name is not specified, it reads from the LLM_PROVIDER env var.
        Defaults to 'ollama'.
        """
        if not provider_name:
            provider_name = os.environ.get("LLM_PROVIDER", "ollama").lower()

        if provider_name == "ollama":
            return OllamaProvider()
        
        # Future providers can be added here
        # elif provider_name == "openai":
        #     return OpenAIProvider()
        
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional