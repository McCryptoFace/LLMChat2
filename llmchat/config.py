import configparser
from typing import List, Union
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    path: Union[str, Path] = Path("config.ini")
    
    def __post_init__(self):
        self.path = Path(self.path)  # Ensure path is a Path object
        self._config = configparser.ConfigParser(comment_prefixes='/', allow_no_value=True)
        self.load()

    def load(self):
        try:
            self._config.read(self.path)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration file: {e}")

    def save(self):
        try:
            with self.path.open("w") as cf:
                self._config.write(cf)
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration file: {e}")

    def _get(self, section: str, option: str, fallback=None):
        return self._config.get(section, option, fallback=fallback)

    def _set(self, section: str, option: str, value: str):
        self._config.set(section, option, value)
        self.save()

    def _get_boolean(self, section: str, option: str, fallback: bool = False) -> bool:
        return self._config.getboolean(section, option, fallback=fallback)

    def _get_float(self, section: str, option: str, fallback: float = 0.0) -> float:
        return self._config.getfloat(section, option, fallback=fallback)

    def _get_int(self, section: str, option: str, fallback: int = 0) -> int:
        return self._config.getint(section, option, fallback=fallback)

    @property
    def openai_key(self) -> str:
        return self._get("OpenAI", "key")

    @openai_key.setter
    def openai_key(self, key: str):
        self._set("OpenAI", "key", key)

    @property
    def openai_model(self) -> str:
        return self._get("OpenAI", "model")

    @openai_model.setter
    def openai_model(self, model_id: str):
        self._set("OpenAI", "model", model_id)

    @property
    def openai_reverse_proxy_url(self) -> str:
        return self._get("OpenAI", "reverse_proxy_url", fallback=None)

    @openai_reverse_proxy_url.setter
    def openai_reverse_proxy_url(self, url: str):
        self._set("OpenAI", "reverse_proxy_url", url)

    @property
    def openai_use_embeddings(self) -> bool:
        return self._get_boolean("OpenAI", "use_embeddings", fallback=False)

    @openai_use_embeddings.setter
    def openai_use_embeddings(self, use_embeddings: bool):
        self._set("OpenAI", "use_embeddings", "true" if use_embeddings else "false")

    @property
    def openai_similarity_threshold(self) -> float:
        return self._get_float("OpenAI", "similarity_threshold", fallback=0.83)

    @openai_similarity_threshold.setter
    def openai_similarity_threshold(self, similarity_threshold: float):
        self._set("OpenAI", "similarity_threshold", str(similarity_threshold))

    @property
    def openai_max_similar_messages(self) -> int:
        return self._get_int("OpenAI", "max_similar_messages", fallback=5)

    @openai_max_similar_messages.setter
    def openai_max_similar_messages(self, max_similar_messages: int):
        self._set("OpenAI", "max_similar_messages", str(max_similar_messages))

    @property
    def llm_context_messages_count(self) -> int:
        return self._get_int("LLM", "context_messages_count")

    @llm_context_messages_count.setter
    def llm_context_messages_count(self, count: int):
        self._set("LLM", "context_messages_count", str(count))

    @property
    def huggingface_key(self) -> str:
        return self._get("HuggingFace", "key")
    
    @huggingface_key.setter
    def huggingface_key(self, key: str):
        self._set("HuggingFace", "key", key)

    @property
    def discord_bot_api_key(self) -> str:
        return self._get("Discord", "bot_api_key")

    @discord_bot_api_key.setter
    def discord_bot_api_key(self, bot_api_key: str):
        self._set("Discord", "bot_api_key", bot_api_key)

    @property
    def discord_active_channels(self) -> List[int]:
        comma_sep_channels = self._get("Discord", "active_channels", fallback=None)
        if not comma_sep_channels:
            return []
        return [int(v.strip()) for v in comma_sep_channels.split(",")]

    @discord_active_channels.setter
    def discord_active_channels(self, active_channels: List[int]): 
        self._set("Discord", "active_channels", ",".join(map(str, active_channels)))

    def can_interact_with_channel_id(self, channel_id: int) -> bool:
        comma_sep_channels = self._get("Discord", "active_channels", fallback=None)
        # If active_channels is set to "all", it will reply to all
        if comma_sep_channels == "all":
            return True
        elif not comma_sep_channels:
            return False
        return channel_id in map(int, comma_sep_channels.split(","))

    @property
    def llama_model_name(self) -> str:
        return self._get("LLaMA", "model_name")

    @llama_model_name.setter
    def llama_model_name(self, model_name: str):
        self._set("LLaMA", "model_name", model_name)

    @property
    def llama_search_path(self) -> str:
        return self._get("LLaMA", "search_path")

    @llama_search_path.setter
    def llama_search_path(self, search_path: str):
        self._set("LLaMA", "search_path", search_path)

    @property
    def bot_identity(self) -> str:
        return self._get("Bot", "identity")

    @bot_identity.setter
    def bot_identity(self, identity: str):
        self._set("Bot", "identity", identity)

    @property
    def bot_name(self) -> str:
        return self._get("Bot", "name")

    @bot_name.setter
    def bot_name(self, name: str):
        self._set("Bot", "name", name)

    @property
    def bot_llm(self) -> str:
        return self._get("Bot", "llm")

    @bot_llm.setter
    def bot_llm(self, llm: str):
        self._set("Bot", "llm", llm)

    @property
    def bot_reminder(self) -> str:
        return self._get("Bot", "reminder", fallback="")

    @bot_reminder.setter
    def bot_reminder(self, reminder: str):
        self._set("Bot", "reminder", reminder)

    @property
    def bot_initial_prompt(self) -> str:
        return self._get(
            "Bot", 
            "initial_prompt", 
            fallback="Write {bot_name}'s next reply in a fictional chat between {bot_name} and {user_name}. Write 1 reply only in internet RP style, italicize actions, and avoid quotation marks. Be proactive, creative, and drive the plot and conversation forward. Write at least 1 paragraph, up to 4. Always stay in character and avoid repetition. {bot_identity} {user_identity}"
        )

    @bot_initial_prompt.setter
    def bot_initial_prompt(self, initial_prompt: str):
        self._set("Bot", "initial_prompt", initial_prompt)

    @property
    def llm_temperature(self) -> float:
        return self._get_float("LLM", "temperature")

    @llm_temperature.setter
    def llm_temperature(self, temperature: float):
        self._set("LLM", "temperature", str(temperature))

    @property
    def llm_presence_penalty(self) -> float:
        return self._get_float("LLM", "presence_penalty")

    @llm_presence_penalty.setter
    def llm_presence_penalty(self, presence_penalty: float):
        self._set("LLM", "presence_penalty", str(presence_penalty))

    @property
    def llm_max_tokens(self) -> int:
        return self._get_int("LLM", "max_tokens")

    @llm_max_tokens.setter
    def llm_max_tokens(self, max_tokens: int):
        self._set("LLM", "max_tokens", str(max_tokens))

    @property
    def llm_frequency_penalty(self) -> float:
        return self._get_float("LLM", "frequency_penalty")

    @llm_frequency_penalty.setter
    def llm_frequency_penalty(self, frequency_penalty: float):
        self._set("LLM", "frequency_penalty", str(frequency_penalty))

    @property
    def discord_bot_name(self) -> str:
        return self._get("Discord", "bot_name")

    @discord_bot_name.setter
    def discord_bot_name(self, bot_name: str):
        self._set("Discord", "bot_name", bot_name)

    @property
    def discord_auto_reconnect(self) -> bool:
        return self._get_boolean("Discord", "auto_reconnect", fallback=True)

    @discord_auto_reconnect.setter
    def discord_auto_reconnect(self, auto_reconnect: bool):
        self._set("Discord", "auto_reconnect", str(auto_reconnect))

    def can_interact_with_channel_id(self, channel_id: int) -> bool:
        channels = self._get("Discord", "active_channels", fallback=None)
        if channels == "all":
            return True
        elif not channels:
            return False
        return channel_id in map(int, channels.split(","))