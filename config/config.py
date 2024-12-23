import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import List
from typing import Literal
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


from .utils import get_env_variable

MAX_MESSAGE_LEN: int = 4095
logger: logging.Logger = logging.getLogger(__name__)

DB_SETTINGS_ERROR_MESSAGE = "Database settings are required to build the database URI"


class FaissDBConfig(BaseModel):
    openai_key: str

    @field_validator("openai_key")
    def check_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field cannot be empty")
        return value


class ModelConfig(BaseModel):
    model: Literal["gpt4-small"]
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    max_tokens: int = Field(..., gt=0, lt=MAX_MESSAGE_LEN)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stop: Optional[List[str]] = Field(default_factory=list, max_items=4)


class AiChatbotConfig(BaseModel):
    description: str = Field(..., max_length=250)
    max_context_len: int = Field(..., gt=0, lt=5e5)
    max_free_context_len: int = Field(..., gt=0, lt=5e5)


class AiChatModel(BaseModel):
    chat_model: ModelConfig
    chatbot: AiChatbotConfig = Field(default_factory=AiChatbotConfig)

    @classmethod
    def load_from_yaml_file(cls, file_path: str, name_of_model: str) -> "AiChatModel":
        logger.debug(f"Current working directory: {Path.cwd()}")
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"The model configuration file isn't found by {file_path}.")
        with file_path.open() as file:
            model_configs = yaml.safe_load(file).get("models", {})
            model_config = model_configs.get(name_of_model, {})
            if not model_config:
                logger.warning(
                    f"The model configuration '{name_of_model}' isn't found. Load default model."
                )

            return cls(**model_config)

    def get_genai_chat_params(self) -> dict[str, Any]:
        return self.chat_model.model_dump()

    def get_genai_speech_params(self) -> dict[str, str | float]:
        return self.voice.model_dump() if self.voice else {}


@dataclass
class Config:
    chat_model: AiChatModel
    voices_directory: Path
    faiss_config: FaissDBConfig
    debug_mode: str


def load_config() -> Config:
    load_dotenv()

    base_dir = Path(__file__).resolve().parent.parent

    model_config_path = base_dir / get_env_variable("MODEL_CONFIG_PATH")
    model_config_name = get_env_variable("MODEL_CONFIG_NAME")
    chat_model = AiChatModel.load_from_yaml_file(model_config_path, model_config_name)
    logger.info(f"The chat model loaded: {chat_model}")

    voices_directory = base_dir / "temp"
    voices_directory.mkdir(exist_ok=True)

    faiss_config = FaissDBConfig(
        openai_key=get_env_variable("AZURE_OPENAI_API_KEY"),
    )

    debug_mode = get_env_variable("DEBUG")

    return Config(
        chat_model=chat_model,
        voices_directory=voices_directory,
        faiss_config=faiss_config,
        debug_mode=debug_mode,
    )
