"""Central prompt translation service for CMK Flow.

Configuration and cached translations live in ComfyUI's user directory. API
keys are never exposed through the public status response or written to logs.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import folder_paths


_LOCK = threading.RLock()
_PROVIDER = "google"
_TARGET = "en"
_TEST_TEXT = "Guten Morgen"


def _storage_root() -> Path:
    override = os.environ.get("CMK_TRANSLATION_CONFIG_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(folder_paths.get_user_directory()).resolve() / "default" / "cmk_nodes"


def _config_path() -> Path:
    return _storage_root() / "translation.json"


def _cache_path() -> Path:
    return _storage_root() / "translation_cache.json"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_private_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    temporary.replace(path)


def load_config() -> dict:
    with _LOCK:
        raw = _read_json(_config_path())
    return {
        "enabled": bool(raw.get("enabled", False)),
        "provider": _PROVIDER,
        "source": "auto",
        "target": _TARGET,
        "api_key": str(raw.get("api_key", "")).strip(),
        "verified": bool(raw.get("verified", False)),
    }


def public_status() -> dict:
    config = load_config()
    key = config["api_key"]
    return {
        "enabled": config["enabled"],
        "configured": bool(key),
        "connected": bool(key) and config["enabled"] and config["verified"],
        "provider": _PROVIDER,
        "source": "AUTO",
        "target": "ENGLISH",
        "key_suffix": key[-4:] if key else "",
    }


def save_config(*, api_key: str | None = None, enabled: bool | None = None) -> dict:
    with _LOCK:
        current = load_config()
        if api_key is not None:
            current["api_key"] = str(api_key).strip()
            current["verified"] = False
        if enabled is not None:
            current["enabled"] = bool(enabled)
        _write_private_json(
            _config_path(),
            {
                "enabled": current["enabled"],
                "provider": _PROVIDER,
                "api_key": current["api_key"],
                "verified": current["verified"],
            },
        )
    return public_status()


def remove_config() -> dict:
    with _LOCK:
        try:
            _config_path().unlink()
        except FileNotFoundError:
            pass
    return public_status()


@dataclass(frozen=True)
class TranslationResult:
    text: str
    status: str
    provider: str = _PROVIDER
    error_code: str = ""
    error_message: str = ""

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    def log_line(self, label: str = "PROMPT") -> str:
        if self.status == "failed":
            return f"TRANSLATION {label}: FAILED · ORIGINAL USED · {self.error_message}"
        labels = {"translated": "GOOGLE", "cache_hit": "CACHE HIT"}
        return f"TRANSLATION {label}: {labels.get(self.status, self.status.upper())}"


class TranslationProviderError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.safe_message = message


def _classify_google_error(status: int, payload: dict | None) -> TranslationProviderError:
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    code = str(error.get("status", "") or status)
    message = str(error.get("message", "") or "Google Translation request failed")
    lowered = message.casefold()
    if "billing" in lowered:
        return TranslationProviderError("BILLING_REQUIRED", "Billing is required for Cloud Translation")
    if "has not been used" in lowered or "is disabled" in lowered:
        return TranslationProviderError("API_NOT_ENABLED", "Cloud Translation API is not enabled")
    if status in (401, 403) or code in {"PERMISSION_DENIED", "UNAUTHENTICATED"}:
        return TranslationProviderError("KEY_REJECTED", "API key rejected or not permitted for Cloud Translation")
    if status == 429 or code == "RESOURCE_EXHAUSTED":
        return TranslationProviderError("QUOTA_EXCEEDED", "Cloud Translation quota exceeded")
    return TranslationProviderError(code or "REQUEST_FAILED", message[:240])


def _google_translate(text: str, api_key: str, timeout: float = 15.0) -> str:
    query = urllib.parse.urlencode({"key": api_key})
    request = urllib.request.Request(
        f"https://translation.googleapis.com/language/translate/v2?{query}",
        data=json.dumps({"q": text, "target": _TARGET, "format": "text"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
        except Exception:
            payload = None
        raise _classify_google_error(error.code, payload) from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise TranslationProviderError("NETWORK_ERROR", "Google Translation is temporarily unavailable") from error
    try:
        translated = payload["data"]["translations"][0]["translatedText"]
    except (KeyError, IndexError, TypeError):
        raise TranslationProviderError("INVALID_RESPONSE", "Google Translation returned an invalid response") from None
    return html.unescape(str(translated))


def _cache_key(text: str) -> str:
    return hashlib.sha256(f"cmk-translation-v1\0{_PROVIDER}\0auto\0{_TARGET}\0{text}".encode("utf-8")).hexdigest()


def translate_prompt(text, *, force: bool = False) -> TranslationResult:
    original = str(text or "")
    if not original.strip():
        return TranslationResult(original, "empty")
    config = load_config()
    if not force and not config["enabled"]:
        return TranslationResult(original, "disabled")
    if not config["api_key"]:
        return TranslationResult(original, "not_configured")
    key = _cache_key(original)
    with _LOCK:
        cached = _read_json(_cache_path()).get(key)
    if isinstance(cached, dict) and isinstance(cached.get("text"), str):
        return TranslationResult(cached["text"], "cache_hit")
    try:
        translated = _google_translate(original, config["api_key"])
    except TranslationProviderError as error:
        return TranslationResult(
            original, "failed", error_code=error.code, error_message=error.safe_message
        )
    with _LOCK:
        cache = _read_json(_cache_path())
        cache[key] = {"text": translated, "provider": _PROVIDER, "target": _TARGET}
        _write_private_json(_cache_path(), cache)
    return TranslationResult(translated, "translated")


def test_connection() -> TranslationResult:
    config = load_config()
    if not config["api_key"]:
        return TranslationResult(_TEST_TEXT, "not_configured")
    try:
        translated = _google_translate(_TEST_TEXT, config["api_key"])
    except TranslationProviderError as error:
        return TranslationResult(
            _TEST_TEXT, "failed", error_code=error.code, error_message=error.safe_message
        )
    with _LOCK:
        current = load_config()
        if current["api_key"] == config["api_key"]:
            _write_private_json(
                _config_path(),
                {
                    "enabled": current["enabled"],
                    "provider": _PROVIDER,
                    "api_key": current["api_key"],
                    "verified": True,
                },
            )
    return TranslationResult(translated, "translated")
