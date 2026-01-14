
from typing import Any
import os
from langchain_community.chat_models import ChatOllama
from llm_core.interface import LLMProvider

class OllamaProvider(LLMProvider):
    """
    Concrete implementation of LLMProvider for local Ollama models.
    """


# === Helpers ===
    def __init__(self):
        # Allow overriding model via env var, default to user request
        self.model_name = os.environ.get("OLLAMA_MODEL", "mistral:7b-instruct")
        # Base URL can also be configured if not localhost:11434
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://172.17.177.186:11434")
        
        self._llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0
        )

    def invoke(self, prompt: Any, **kwargs) -> Any:
        """
        Invokes importance Ollama LLM.
        """
        if hasattr(prompt, 'invoke'):
            return prompt.invoke(kwargs)
        
        return self._llm.invoke(prompt, **kwargs)

    def get_base_model(self) -> Any:
        return self._llm
