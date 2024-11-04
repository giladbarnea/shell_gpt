import json
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

from ..cache import Cache
from ..config import USE_ANTHROPIC, USE_OPENAI, cfg
from ..function import get_function
from ..printer import MarkdownPrinter, Printer, TextPrinter
from ..role import DefaultRoles, SystemRole

completion: Callable[..., Any] = lambda *args, **kwargs: Generator[Any, None, None]

base_url = cfg.get("API_BASE_URL")
# use_litellm = cfg.get("USE_LITELLM") == "true"
additional_kwargs = {
    "timeout": int(cfg.get("REQUEST_TIMEOUT")),
    # "api_key": cfg.get("OPENAI_API_KEY"),
    "base_url": None if base_url == "default" else base_url,
}

if cfg.get("USE_LITELLM") == "true":
    import litellm  # type: ignore

    completion = litellm.completion
    litellm.suppress_debug_info = True
    # additional_kwargs.pop("api_key")
elif USE_ANTHROPIC:
    import anthropic

    client = anthropic.Anthropic(api_key=cfg.get("ANTHROPIC_API_KEY"))

    def _adapt_parameters_to_anthropic(*args, **kwargs):
        if 'system' in kwargs:
            return args, kwargs
        if any(
            isinstance(arg,dict) and arg.get("role") == "system" for arg in args
        ):
            return args, kwargs
        messages = kwargs.pop("messages", None)
        args_without_messages = []
        if messages is None:
            for arg in args:
                if isinstance(arg, list) and arg and "role" in arg[0]:
                    messages = arg
                else:
                    args_without_messages.append(arg)
        else:
            args_without_messages = args
        messages_without_system = []
        system_message = None
        for message in messages:
            if message["role"] == "system":
                system_message = message
            else:
                message["content"] = [{
                    "type": "text",
                    "text": message["content"],
                }]
                messages_without_system.append(message)
        kwargs["messages"] = messages_without_system
        kwargs["system"] = system_message["content"]
        assert kwargs["model"] == "claude-3-5-sonnet-20241022", kwargs["model"]
        return args_without_messages, kwargs

    def completion(*args, **kwargs):
        args, kwargs = _adapt_parameters_to_anthropic(*args, **kwargs)
        return client.messages.create(*args, **kwargs)
    additional_kwargs = {}
elif USE_OPENAI:
    from openai import OpenAI

    client = OpenAI(**additional_kwargs, api_key=cfg.get("OPENAI_API_KEY"))  # type: ignore
    completion = client.chat.completions.create
    additional_kwargs = {}


class Handler:
    cache = Cache(int(cfg.get("CACHE_LENGTH")), Path(cfg.get("CACHE_PATH")))

    def __init__(self, role: SystemRole, markdown: bool) -> None:
        self.role = role

        api_base_url = cfg.get("API_BASE_URL")
        self.base_url = None if api_base_url == "default" else api_base_url
        self.timeout = int(cfg.get("REQUEST_TIMEOUT"))

        self.markdown = "APPLY MARKDOWN" in self.role.role or markdown
        self.code_theme, self.color = cfg.get("CODE_THEME"), cfg.get("DEFAULT_COLOR")

    @property
    def printer(self) -> Printer:
        return (
            MarkdownPrinter(self.code_theme)
            if self.markdown
            else TextPrinter(self.color)
        )

    def make_messages(self, prompt: str) -> List[Dict[str, str]]:
        raise NotImplementedError

    def handle_function_call(
        self,
        messages: List[dict[str, Any]],
        name: str,
        arguments: str,
    ) -> Generator[str, None, None]:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "function_call": {"name": name, "arguments": arguments},
            }
        )

        if messages and messages[-1]["role"] == "assistant":
            yield "\n"

        dict_args = json.loads(arguments)
        # joined_args = ", ".join(f'{k}="{v}"' for k, v in dict_args.items())
        # yield f"> @FunctionCall `{name}({joined_args})` \n\n"

        result = get_function(name)(**dict_args)
        if cfg.get("SHOW_FUNCTIONS_OUTPUT") == "true":
            yield f"```text\n{result}\n```\n"
        messages.append({"role": "function", "content": result, "name": name})

    @cache
    def get_completion(
        self,
        model: str,
        temperature: float,
        top_p: float,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Dict[str, str]]],
    ) -> Generator[str, None, None]:
        name = arguments = ""
        is_shell_role = self.role.name == DefaultRoles.SHELL.value
        is_code_role = self.role.name == DefaultRoles.CODE.value
        is_dsc_shell_role = self.role.name == DefaultRoles.DESCRIBE_SHELL.value
        if is_shell_role or is_code_role or is_dsc_shell_role:
            functions = None

        if functions:
            additional_kwargs["tool_choice"] = "auto"
            additional_kwargs["tools"] = functions
            additional_kwargs["parallel_tool_calls"] = False

        response = completion(
            model=model,
            temperature=temperature,
            top_p=top_p,
            messages=messages,
            stream=True,
            **additional_kwargs,
        )

        try:
            for chunk in response:
                delta = chunk.choices[0].delta

                # LiteLLM uses dict instead of Pydantic object like OpenAI does.
                tool_calls = (
                    delta.get("tool_calls") if use_litellm else delta.tool_calls
                )
                if tool_calls:
                    for tool_call in tool_calls:
                        if tool_call.function.name:
                            name = tool_call.function.name
                        if tool_call.function.arguments:
                            arguments += tool_call.function.arguments
                if chunk.choices[0].finish_reason == "tool_calls":
                    yield from self.handle_function_call(messages, name, arguments)
                    yield from self.get_completion(
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        messages=messages,
                        functions=functions,
                        caching=False,
                    )
                    return

                yield delta.content or ""
        except KeyboardInterrupt:
            response.close()

    def handle(
        self,
        prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        caching: bool,
        functions: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        disable_stream = cfg.get("DISABLE_STREAMING") == "true"
        messages = self.make_messages(prompt.strip())
        generator = self.get_completion(
            model=model,
            temperature=temperature,
            top_p=top_p,
            messages=messages,
            functions=functions,
            caching=caching,
            **kwargs,
        )
        return self.printer(generator, not disable_stream)
