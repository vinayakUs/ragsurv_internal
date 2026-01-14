from abc import ABC,abstractmethod
from typing import Any , Dict , Optional

class LLMProvider(ABC):
    """
    Abstract Base Class for LLM Providers.
    Defines the contract that all concrete LLM implementations must follow.
    """

    @abstractmethod
    def invoke(self, prompt: Any, **kwargs) -> Any:
        """
        Invokes the LLM with the given prompt.
        
        Args:
            prompt: The input prompt (string or LangChain prompt object).
            **kwargs: Additional arguments specific to the provider.
            
        Returns:
            The response from the LLM.
        """
        pass

    @abstractmethod
    def get_base_model(self) -> Any:
        """
        Returns the underlying model object (e.g., for LangChain compatibility).
        """
        pass

