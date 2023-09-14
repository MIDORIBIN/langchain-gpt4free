from typing import Any, List, Mapping, Optional, Union
from functools import partial

from g4f import ChatCompletion
from g4f.models import Model
from g4f.Provider.base_provider import BaseProvider
from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

MAX_TRIES = 5
class G4FLLM(LLM):
    model: Union[Model, str]
    provider: Optional[type[BaseProvider]] = None
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

        for i in range(MAX_TRIES):
            try:
                text = ChatCompletion.create(
                    messages=[{"role": "user", "content": prompt}],
                    **create_kwargs,
                )

                # Generator -> str
                text = text if type(text) is str else "".join(text)
                if stop is not None:
                    text = enforce_stop_tokens(text, stop)
                if text:
                    return text
                print(f"Empty response, trying {i+1} of {MAX_TRIES}")
            except Exception as e:
                print(f"Error in G4FLLM._call: {e}, trying {i+1} of {MAX_TRIES}")
        return ""
    

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        create_kwargs = {} if self.create_kwargs is None else self.create_kwargs.copy()
        create_kwargs["model"] = self.model
        if self.provider is not None:
            create_kwargs["provider"] = self.provider
        if self.auth is not None:
            create_kwargs["auth"] = self.auth

        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token)
        
        text = ""
        for token in ChatCompletion.create(messages=[{"role": "user", "content": prompt}], stream=True, **create_kwargs):
            if text_callback:
                await text_callback(token)
            text += token
        return text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "provider": self.provider,
            "auth": self.auth,
            "create_kwargs": self.create_kwargs,
        }
