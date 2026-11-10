"""
Provider-agnostic LLM client for Video Studio. Plain REST calls, no vendor SDKs.

Providers:
  groq       Groq's OpenAI-compatible API
  openai     OpenAI (ChatGPT models)
  anthropic  Anthropic (Claude models)
  custom     Any OpenAI-compatible endpoint (OpenRouter, Ollama, Together, LM Studio, ...)

Keys and models come from the caller (e.g. typed in the web UI) or fall back to environment variables.
"""

import json
import os
import re
from dataclasses import dataclass

import requests

TIMEOUT = 90

PROVIDERS = {
    "groq": dict(
        label="Groq", kind="openai", base="https://api.groq.com/openai/v1", env_base="GROQ_BASE_URL",
        env_key="GROQ_API_KEY", env_model="GROQ_MODEL", default_model="qwen/qwen3.8-27b",
    ),
    "openai": dict(
        label="OpenAI (ChatGPT)", kind="openai", base="https://api.openai.com/v1", env_base="OPENAI_BASE_URL",
        env_key="OPENAI_API_KEY", env_model="OPENAI_MODEL", default_model="",
    ),
    "anthropic": dict(
        label="Anthropic (Claude)", kind="anthropic", base="https://api.anthropic.com/v1",
        env_base="ANTHROPIC_BASE_URL", env_key="ANTHROPIC_API_KEY", env_model="ANTHROPIC_MODEL",
        default_model="claude-haiku-5-5",
    ),
    "custom": dict(
        label="Custom (OpenAI-compatible)", kind="openai", base="", env_base="OPENAI_COMPAT_BASE_URL",
        env_key="OPENAI_COMPAT_API_KEY", env_model="OPENAI_COMPAT_MODEL", default_model="", key_optional=True,
    ),
}

# Model ids containing these words are not text-chat models; hide them from the picker.
NON_CHAT = ("embed", "whisper", "tts", "dall-e", "moderation", "transcribe", "realtime",
            "image", "audio", "guard", "davinci", "babbage", "sora", "rerank")


class LLMError(Exception):
    """A problem talking to a provider, with a message that is safe to show to the user."""


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str


# ---------- configuration ----------

def default_provider() -> str:
    p = (os.getenv("LLM_PROVIDER") or "groq").lower()
    return p if p in PROVIDERS else "groq"


def server_key(provider: str) -> str:
    return os.getenv(PROVIDERS[provider]["env_key"], "") or ""


def default_model(provider: str) -> str:
    spec = PROVIDERS[provider]
    return os.getenv(spec["env_model"]) or spec["default_model"]


def env_base_url(provider: str) -> str:
    spec = PROVIDERS[provider]
    return os.getenv(spec["env_base"]) or spec["base"]


def build_config(provider: str, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, allow_custom_base: bool = True,
                 need_model: bool = True) -> LLMConfig:
    """Combine user-supplied values with environment defaults. Raises ValueError if unusable."""
    provider = (provider or default_provider()).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'.")
    spec = PROVIDERS[provider]

    key = (api_key or "").strip() or server_key(provider)
    if not key and not spec.get("key_optional"):
        raise ValueError(f"No API key for {spec['label']}. Paste one in the key box or set {spec['env_key']} in .env.")

    url = ""
    if provider == "custom":
        url = ((base_url or "").strip() if allow_custom_base else "") or env_base_url(provider)
        if not url:
            raise ValueError("Enter a base URL for the custom provider (for example https://openrouter.ai/api/v1).")
        if not re.match(r"^https?://", url):
            raise ValueError("Base URL must start with http:// or https://")
    else:
        url = env_base_url(provider)

    mdl = (model or "").strip() or default_model(provider)
    if need_model and not mdl:
        raise ValueError("Choose a model first.")
    return LLMConfig(provider=provider, model=mdl, api_key=key, base_url=url.rstrip("/"))


def config_from_env() -> LLMConfig | None:
    """The default provider configured purely from .env (used by the command line)."""
    try:
        return build_config(default_provider())
    except ValueError:
        return None


# ---------- HTTP helpers ----------

def _headers(cfg: LLMConfig) -> dict:
    if PROVIDERS[cfg.provider]["kind"] == "anthropic":
        return {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    h = {"content-type": "application/json"}
    if cfg.api_key:
        h["authorization"] = f"Bearer {cfg.api_key}"
    return h


def _error_message(resp: requests.Response) -> str:
    try:
        err = resp.json().get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err:
            return str(err)
    except ValueError:
        pass
    return (resp.text or "").strip()[:300] or "no details"


def _request(method: str, cfg: LLMConfig, path: str, **kwargs) -> requests.Response:
    try:
        return requests.request(method, cfg.base_url + path, headers=_headers(cfg), timeout=TIMEOUT, **kwargs)
    except requests.RequestException as e:
        raise LLMError(f"Could not reach {PROVIDERS[cfg.provider]['label']}: {type(e).__name__}") from None


def _check(resp: requests.Response, cfg: LLMConfig) -> None:
    if resp.status_code == 401 or resp.status_code == 403:
        raise LLMError(f"{PROVIDERS[cfg.provider]['label']} rejected the API key ({resp.status_code}).")
    if resp.status_code >= 400:
        raise LLMError(f"{PROVIDERS[cfg.provider]['label']} error {resp.status_code}: {_error_message(resp)}")


# ---------- models ----------

def list_models(cfg: LLMConfig) -> list[str]:
    path = "/models?limit=1000" if PROVIDERS[cfg.provider]["kind"] == "anthropic" else "/models"
    resp = _request("GET", cfg, path)
    _check(resp, cfg)
    try:
        items = resp.json().get("data", [])
        ids = [m["id"] for m in items if isinstance(m, dict) and m.get("id")]
    except ValueError:
        raise LLMError("The provider returned an unreadable model list.") from None
    ids = [i for i in ids if not any(w in i.lower() for w in NON_CHAT)]
    return sorted(set(ids))


# ---------- chat ----------

def parse_json(raw: str) -> dict:
    """Extract a JSON object from a model reply (tolerates <think> blocks and code fences)."""
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise LLMError("The model did not return JSON. Try another model. Reply started with: " + raw[:150])
    try:
        return json.loads(match.group(0))
    except ValueError:
        raise LLMError("The model returned malformed JSON. Try again or pick another model.") from None


def _post_with_fallback(cfg: LLMConfig, path: str, body: dict) -> requests.Response:
    """POST, and if the model rejects temperature / JSON mode, retry without them."""
    for _ in range(3):
        resp = _request("POST", cfg, path, json=body)
        if resp.status_code != 400:
            break
        text = resp.text.lower()
        changed = False
        if "temperature" in body and "temperature" in text:
            body.pop("temperature")
            changed = True
        if "response_format" in body and ("response_format" in text or "json" in text):
            body.pop("response_format")
            changed = True
        if not changed:
            break
    _check(resp, cfg)
    return resp


def chat_json(cfg: LLMConfig, system: str, user: str) -> dict:
    """Ask the model for a JSON reply and return it parsed."""
    if PROVIDERS[cfg.provider]["kind"] == "anthropic":
        body = {"model": cfg.model, "max_tokens": 4096, "temperature": 0.7, "system": system,
                "messages": [{"role": "user", "content": user}]}
        resp = _post_with_fallback(cfg, "/messages", body)
        blocks = resp.json().get("content", [])
        raw = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    else:
        body = {"model": cfg.model, "temperature": 0.7, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        resp = _post_with_fallback(cfg, "/chat/completions", body)
        try:
            raw = resp.json()["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, ValueError):
            raise LLMError("Unexpected response from the provider.") from None
    return parse_json(raw)