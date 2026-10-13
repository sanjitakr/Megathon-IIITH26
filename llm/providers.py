"""One adapter for free LLM options (spec 11). The primary is Google Gemini (free tier);
it is switchable by environment variable, falls back through a chain of other free
providers, and finally to deterministic templates (handled by the caller).

Environment (.env is read if present):
  LLM_PROVIDER      gemini | groq | openrouter | huggingface | ollama | none   (default gemini)
  LLM_FALLBACKS     comma list tried in order after the primary, e.g. "groq,ollama"
  GEMINI_API_KEY, GEMINI_MODEL (default gemini-flash-latest), GEMINI_FALLBACK_MODELS (tried on 404)
  GROQ_API_KEY, GROQ_MODEL (default llama-3.1-8b-instant)
  OPENROUTER_API_KEY, OPENROUTER_MODEL (default meta-llama/llama-3.1-8b-instruct:free)
  HF_TOKEN, HF_MODEL (default Qwen/Qwen2.5-7B-Instruct)
  OLLAMA_HOST (default http://localhost:11434), OLLAMA_MODEL (default qwen2.5:7b)
  GEMINI_THINKING_LEVEL (default low), LLM_TIMEOUT_S (default 60), LLM_RETRIES (429 retries, default 3), LLM_MAX_WAIT_S, LLM_MIN_INTERVAL_S
Free-tier limits change often: check each provider's current limits.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import requests

from sim.parameters import ROOT


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()


class ProviderError(RuntimeError):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


def _retry_delay(r: requests.Response) -> float | None:
    """Seconds to wait on a 429 (from Retry-After or Gemini's retryDelay), else None."""
    if r.status_code in (500, 503):
        return 5.0  # transient overload
    if r.status_code != 429:
        return None
    if r.headers.get("retry-after"):
        try:
            return float(r.headers["retry-after"])
        except ValueError:
            pass
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', r.text)
    return float(m.group(1)) if m else 10.0


def _timeout() -> float:
    return float(os.getenv("LLM_TIMEOUT_S", "60"))


@dataclass
class Provider:
    name: str

    def available(self) -> bool:
        return True

    def generate(self, system: str, prompt: str, json_mode: bool = False, temperature: float = 0.2) -> str:
        raise NotImplementedError


class GeminiProvider(Provider):
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    # Google retires model names; "-latest" aliases track the current free-tier model.
    DEFAULT_MODEL = "gemini-flash-latest"
    DEFAULT_FALLBACK_MODELS = "gemini-flash-latest,gemini-flash-lite-latest"

    def __init__(self):
        super().__init__("gemini")
        self.key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = os.getenv("GEMINI_MODEL") or self.DEFAULT_MODEL
        fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", self.DEFAULT_FALLBACK_MODELS)
        self.models = list(dict.fromkeys([self.model] + [m.strip() for m in fallbacks.split(",") if m.strip()]))

    def available(self) -> bool:
        return bool(self.key)

    def _call(self, model: str, body: dict) -> requests.Response:
        return requests.post(self.URL.format(model=model), json=body, headers={"x-goog-api-key": self.key}, timeout=_timeout())

    def generate(self, system, prompt, json_mode=False, temperature=0.2):
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            # generous limit: newer models spend part of it on internal reasoning
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096},
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        errors = []
        level = os.getenv("GEMINI_THINKING_LEVEL", "low").strip()
        for model in list(self.models):
            b = body
            if level and level != "default" and "lite" not in model:
                # newer Gemini models think by default (~13 s per answer); "low" keeps answers fast
                b = {**body, "generationConfig": {**body["generationConfig"], "thinkingConfig": {"thinkingLevel": level}}}
            try:
                r = self._call(model, b)
                if r.status_code == 400 and "thinking" in r.text.lower():
                    r = self._call(model, body)  # model does not accept the setting
            except requests.Timeout:
                errors.append(f"{model}: timed out")
                if model != self.models[-1]:
                    continue
                raise ProviderError(f"gemini ({model}) timed out", retry_after=5.0)
            if r.status_code in (400, 404) and "model" in r.text.lower():
                # retired or unknown model: try the next one and stop using this one
                errors.append(f"{model}: HTTP {r.status_code}")
                if len(self.models) > 1:
                    self.models.remove(model)
                continue
            if r.status_code in (429, 500, 503) and model != self.models[-1]:
                # overloaded or rate-limited: the next model has its own quota
                errors.append(f"{model}: HTTP {r.status_code}")
                continue
            if r.status_code != 200:
                raise ProviderError(f"gemini ({model}) HTTP {r.status_code}: {r.text[:300]}", retry_after=_retry_delay(r))
            self.model = model
            data = r.json()
            try:
                text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            except (KeyError, IndexError) as exc:
                raise ProviderError(f"gemini ({model}): no text in response {str(data)[:200]}") from exc
            if not text.strip():
                raise ProviderError(f"gemini ({model}): empty response")
            return text
        raise ProviderError("gemini: no usable model (" + "; ".join(errors) + "). Set GEMINI_MODEL to a model your key lists.")


class OpenAICompatibleProvider(Provider):
    """Groq, OpenRouter and the Hugging Face router all speak the OpenAI chat API."""

    CONFIG = {
        "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", "GROQ_MODEL", "llama-3.1-8b-instant"),
        "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),
        "huggingface": ("https://router.huggingface.co/v1/chat/completions", "HF_TOKEN", "HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    }

    def __init__(self, name: str):
        super().__init__(name)
        url, key_env, model_env, default_model = self.CONFIG[name]
        self.url = url
        self.key = os.getenv(key_env)
        self.model = os.getenv(model_env, default_model)

    def available(self) -> bool:
        return bool(self.key)

    def generate(self, system, prompt, json_mode=False, temperature=0.2):
        body = {"model": self.model, "temperature": temperature, "max_tokens": 1024,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(self.url, json=body, headers={"Authorization": f"Bearer {self.key}"}, timeout=_timeout())
        if r.status_code != 200:
            raise ProviderError(f"{self.name} HTTP {r.status_code}: {r.text[:200]}", retry_after=_retry_delay(r))
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"{self.name}: unexpected response") from exc


class OllamaProvider(Provider):
    def __init__(self):
        super().__init__("ollama")
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    def available(self) -> bool:
        try:
            return requests.get(f"{self.host}/api/tags", timeout=1.5).status_code == 200
        except requests.RequestException:
            return False

    def generate(self, system, prompt, json_mode=False, temperature=0.2):
        body = {"model": self.model, "stream": False, "options": {"temperature": temperature},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if json_mode:
            body["format"] = "json"
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=max(_timeout(), 120))
        if r.status_code != 200:
            raise ProviderError(f"ollama HTTP {r.status_code}: {r.text[:200]}")
        return r.json()["message"]["content"]


_last_call: dict[str, float] = {}


def _throttle(name: str) -> None:
    """Keep a minimum gap between calls (LLM_MIN_INTERVAL_S) to stay inside free-tier per-minute limits."""
    gap = float(os.getenv("LLM_MIN_INTERVAL_S", "4"))
    wait = _last_call.get(name, 0.0) + gap - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call[name] = time.monotonic()


def make_provider(name: str) -> Provider | None:
    name = name.strip().lower()
    if name in ("", "none", "template"):
        return None
    if name == "gemini":
        return GeminiProvider()
    if name in OpenAICompatibleProvider.CONFIG:
        return OpenAICompatibleProvider(name)
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(f"unknown LLM provider {name}")


class LLMChain:
    """Try the primary provider, then each fallback; raise ProviderError if all fail."""

    def __init__(self, names: list[str] | None = None):
        if names is None:
            primary = os.getenv("LLM_PROVIDER", "gemini")
            fallbacks = [n for n in os.getenv("LLM_FALLBACKS", "").split(",") if n.strip()]
            names = [primary] + fallbacks
        self.providers = [p for p in (make_provider(n) for n in names) if p is not None]
        self.last_errors: list[str] = []

    def available(self) -> list[str]:
        return [p.name for p in self.providers if p.available()]

    def generate(self, system: str, prompt: str, json_mode: bool = False, retries: int | None = None) -> tuple[str, str]:
        """Rate-limited (429) calls are retried after the provider's suggested delay
        (capped by LLM_MAX_WAIT_S) before moving on to the next provider."""
        retries = int(os.getenv("LLM_RETRIES", "3")) if retries is None else retries
        max_wait = float(os.getenv("LLM_MAX_WAIT_S", "65"))
        self.last_errors = []
        for p in self.providers:
            if not p.available():
                self.last_errors.append(f"{p.name}: not configured")
                continue
            for attempt in range(retries + 1):
                _throttle(p.name)
                try:
                    return p.generate(system, prompt, json_mode=json_mode), p.name
                except (ProviderError, requests.RequestException, ValueError) as exc:
                    self.last_errors.append(f"{p.name}: {exc}")
                    wait = getattr(exc, "retry_after", None)
                    if wait is not None and attempt < retries and wait <= max_wait:
                        time.sleep(wait + 1)
                        continue
                    break
        raise ProviderError("; ".join(self.last_errors) or "no LLM provider configured")


def status() -> dict:
    chain = LLMChain()
    gem = next((p for p in chain.providers if p.name == "gemini"), None)
    return {
        "configured_chain": [p.name for p in chain.providers],
        "gemini_models": gem.models if gem else None,
        "available": chain.available(),
        "fallback": "deterministic templates",
    }
