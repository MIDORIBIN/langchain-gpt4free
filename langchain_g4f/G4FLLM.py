from types import ModuleType
from typing import Any, List, Mapping, Optional, Union

from g4f import ChatCompletion
from g4f.models import Model
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens


class G4FLLM(LLM):
    # Model.model or str
    model: Union[Model, str]
    # Provider.Provider
    provider: Optional[ModuleType] = None
    auth: Optional[Union[str, bool]] = None
    create_kwargs: Optional[dict[str, Any]] = None

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        create_kwargs = {} if self.create_kwargs is None else self.create_kwargs.copy()
        create_kwargs["model"] = self.model
        if self.provider is not None:
            create_kwargs["provider"] = self.provider
        if self.auth is not None:
            create_kwargs["auth"] = self.auth

        text = ChatCompletion.create(  # type: ignore
            messages=[{"role": "user", "content": prompt}],
            **create_kwargs,
        )
        if stop is not None and type(stop) is str:
            text = enforce_stop_tokens(text, stop)  # type: ignore
        return text  # type: ignore

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "provider": self.provider,
            "auth": self.auth,
            "create_kwargs": self.create_kwargs,
        }
