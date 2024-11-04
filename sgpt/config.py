import os
from getpass import getpass
from pathlib import Path
from tempfile import gettempdir
from typing import Any, List

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


from click import UsageError


class SupportedVendor(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


VENDOR_KEYWORDS = {
    SupportedVendor.ANTHROPIC: ["claude"],
    SupportedVendor.OPENAI: ["gpt"],
}


def infer_vendor_from_precedence(model_precedence: List[str]) -> SupportedVendor:
    for model_name in model_precedence:
        model_name = model_name.lower()
        for vendor, keywords in VENDOR_KEYWORDS.items():
            if any(keyword in model_name for keyword in keywords):
                return vendor

    recognized_keywords = [
        f"'{keyword}' for {vendor.lower()}"
        for vendor, keywords in VENDOR_KEYWORDS.items()
        for keyword in keywords
    ]
    raise ValueError(
        "Unable to infer vendor from the provided model precedence list. "
        "Please ensure that the list contains at least one model name "
        f"with a recognized vendor keyword ({', '.join(recognized_keywords)})."
    )


CONFIG_FOLDER = os.path.expanduser("~/.config")
SHELL_GPT_CONFIG_FOLDER = Path(CONFIG_FOLDER) / "shell_gpt"
SHELL_GPT_CONFIG_PATH = SHELL_GPT_CONFIG_FOLDER / ".sgptrc"
ROLE_STORAGE_PATH = SHELL_GPT_CONFIG_FOLDER / "roles"
FUNCTIONS_PATH = SHELL_GPT_CONFIG_FOLDER / "functions"
CHAT_CACHE_PATH = Path(gettempdir()) / "chat_cache"
CACHE_PATH = Path(gettempdir()) / "cache"

# TODO: Refactor ENV variables with SGPT_ prefix.
DEFAULT_CONFIG = {
    # TODO: Refactor it to CHAT_STORAGE_PATH.
    "CHAT_CACHE_PATH": os.getenv("CHAT_CACHE_PATH", str(CHAT_CACHE_PATH)),
    "CACHE_PATH": os.getenv("CACHE_PATH", str(CACHE_PATH)),
    "CHAT_CACHE_LENGTH": int(os.getenv("CHAT_CACHE_LENGTH", "100")),
    "CACHE_LENGTH": int(os.getenv("CHAT_CACHE_LENGTH", "100")),
    "REQUEST_TIMEOUT": int(os.getenv("REQUEST_TIMEOUT", "60")),
    "DEFAULT_COLOR": os.getenv("DEFAULT_COLOR", "magenta"),
    "ROLE_STORAGE_PATH": os.getenv("ROLE_STORAGE_PATH", str(ROLE_STORAGE_PATH)),
    "DEFAULT_EXECUTE_SHELL_CMD": os.getenv("DEFAULT_EXECUTE_SHELL_CMD", "false"),
    "DISABLE_STREAMING": os.getenv("DISABLE_STREAMING", "false"),
    "CODE_THEME": os.getenv("CODE_THEME", "dracula"),
    "OPENAI_FUNCTIONS_PATH": os.getenv("OPENAI_FUNCTIONS_PATH", str(FUNCTIONS_PATH)),
    "OPENAI_USE_FUNCTIONS": os.getenv("OPENAI_USE_FUNCTIONS", "true"),
    "SHOW_FUNCTIONS_OUTPUT": os.getenv("SHOW_FUNCTIONS_OUTPUT", "false"),
    "API_BASE_URL": os.getenv("API_BASE_URL", "default"),
    "PRETTIFY_MARKDOWN": os.getenv("PRETTIFY_MARKDOWN", "true"),
    "USE_LITELLM": os.getenv("USE_LITELLM", "false"),
    "SHELL_INTERACTION": os.getenv("SHELL_INTERACTION ", "true"),
    "OS_NAME": os.getenv("OS_NAME", "auto"),
    "SHELL_NAME": os.getenv("SHELL_NAME", "auto"),
    "MODEL_PRECEDENCE": os.getenv(
        "MODEL_PRECEDENCE", "claude-3-5-sonnet-20241022, gpt-4o"
    ),
    # New features might add their own config variables here.
}


MODEL_PRECEDENCE: List[str] = [
    model.strip() for model in DEFAULT_CONFIG["MODEL_PRECEDENCE"].split(",")
]
DEFAULT_CONFIG["MODEL_PRECEDENCE"] = MODEL_PRECEDENCE
MODEL = MODEL_PRECEDENCE[0]

VENDOR: SupportedVendor = infer_vendor_from_precedence(MODEL_PRECEDENCE)
USE_ANTHROPIC: bool = VENDOR == SupportedVendor.ANTHROPIC
USE_OPENAI: bool = VENDOR == SupportedVendor.OPENAI

POTENTIAL_PATH_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


class Config(dict):  # type: ignore
    def __init__(self, config_path: Path, **defaults: Any):
        self.config_path = config_path

        if self._exists:
            self._read()
            has_new_config = False
            for key, value in defaults.items():
                if key not in self:
                    has_new_config = True
                    self[key] = value
            if has_new_config:
                self._write()
        else:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            # Don't write API key to config file if it is in the environment.
            if not defaults.get("ANTHROPIC_API_KEY") and not os.getenv(
                "ANTHROPIC_API_KEY"
            ):
                __anthropic_key = getpass(
                    prompt="Please enter your Anthropic API key: "
                )
                defaults["ANTHROPIC_API_KEY"] = __anthropic_key
            if not defaults.get("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
                __api_key = getpass(prompt="Please enter your OpenAI API key: ")
                defaults["OPENAI_API_KEY"] = __api_key
            super().__init__(**defaults)
            self._write()
        self['MODEL_PRECEDENCE'] = [model.strip() for model in self['MODEL_PRECEDENCE'].split(",")]

    @property
    def _exists(self) -> bool:
        return self.config_path.exists()

    def _write(self) -> None:
        with open(self.config_path, "w", encoding="utf-8") as file:
            string_config = ""
            for key, value in self.items():
                string_config += f"{key}={value}\n"
            file.write(string_config)

    def _read(self) -> None:
        with open(self.config_path, "r", encoding="utf-8") as file:
            for line in file:
                self._read_from_line(line)

    def _read_from_line(self, line: str) -> None:
        line = line.strip()
        if not line or line.startswith("#"):
            return
        key, value = line.split("=", 1)
        if key in POTENTIAL_PATH_KEYS and (path := Path(value).expanduser()).is_file():
            value = str(path.read_text().strip())
        self[key] = value

    def get(self, key: str) -> str:  # type: ignore
        # Prioritize environment variables over config file.
        value = os.getenv(key) or super().get(key)
        if not value:
            raise UsageError(f"Missing config key: {key}")
        return value

    @property
    def model(self):
        model_precedence: List[str] = self['MODEL_PRECEDENCE']  # Not using get() because of the list type.
        return model_precedence[0]


cfg = Config(SHELL_GPT_CONFIG_PATH, **DEFAULT_CONFIG)
