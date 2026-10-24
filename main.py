#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TG AI Bot — Telegram-бот с подключаемыми AI-провайдерами (как в opencode).

Локально (long polling, сервер не нужен):
    pip install aiohttp
    BOT_TOKEN=123:ABC python main.py      # или просто python main.py — спросит токен

На Render (Web Service, всё через переменные окружения — config.json не нужен):
    BOT_TOKEN            токен от @BotFather (обязательно)
    ADMIN_IDS            ID админов через запятую
    PROVIDER_1..N        провайдеры, применяются при каждом старте:
                           PROVIDER_1=openrouter:sk-or-v1-…        (пресет:ключ)
                           PROVIDER_2=DeepSeek|openai|https://api.deepseek.com/v1|sk-…
                           (пресеты: openrouter openai anthropic google gemini zai
                            groq deepseek mistral xai moonshot together fireworks cerebras)
    CUSTOM_URL / CUSTOM_KEY / CUSTOM_NAME / CUSTOM_PROTOCOL / CUSTOM_MODEL
                         свой провайдер с произвольным URL — без пресетов,
                         ни от кого не зависит (id: ec)
    DEFAULT_MODEL        модель по умолчанию, напр. deepseek-chat или openai/gpt-4o-mini
    DEFAULT_PROVIDER     имя провайдера для дефолта (иначе первый PROVIDER_N)
    SYSTEM_PROMPT / MAX_HISTORY / ADMIN_ONLY — применяются при каждом старте
    Render сам задает PORT и RENDER_EXTERNAL_URL — бот поднимет HTTP-сервер
    (health-check на "/") и зарегистрирует webhook.

Дальше всё настраивается и прямо в Telegram:
    /add       — добавить провайдера (пресеты OpenRouter/OpenAI/Anthropic/...)
    /model     — выбрать модель для чата
    /providers — список провайдеров
    /help      — все команды

Провайдеры хранятся в config.json, история диалогов — в state.json
(пути можно переопределить переменными CONFIG_PATH / STATE_PATH — например,
на постоянный диск /data при платном тарифе Render; на Render провайдеров
можно задавать целиком через PROVIDER_1..N — тогда config.json не нужен).

Режим работы (переменная MODE): "polling", "webhook" или пусто (авто —
webhook, если задан RENDER_EXTERNAL_URL / WEBHOOK_URL, иначе polling).
Зависимость одна: aiohttp. Python 3.9+.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import signal
import sys
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    print("Не хватает зависимости. Установите её командой:\n    pip install aiohttp")
    sys.exit(1)

# ----------------------------------------------------------------------------
# Константы
# ----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("CONFIG_PATH") or os.path.join(BASE_DIR, "config.json")
STATE_PATH = os.environ.get("STATE_PATH") or os.path.join(BASE_DIR, "state.json")
PORT = int(os.environ.get("PORT", "8080") or 8080)
WEBHOOK_URL_ENV = (
    os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("WEBHOOK_URL") or ""
).strip().rstrip("/")

TG_API = "https://api.telegram.org/bot{token}/{method}"

PAGE_SIZE = 12          # моделей на страницу в клавиатуре
MODELS_TTL = 300        # секунд кэша списка моделей
EDIT_EVERY = 1.6        # интервал редактирования сообщения при стриминге
MAX_TG = 4000           # безопасный лимит длины сообщения Telegram
MAX_TOKENS_ANTHROPIC = 8192

PROTOCOLS = ("openai", "anthropic", "gemini")
PROTOCOL_TITLES = {
    "openai": "OpenAI-compatible",
    "anthropic": "Anthropic-compatible",
    "gemini": "Google Gemini",
}

PRESETS: Dict[str, Dict[str, str]] = {
    "custom-openai": {
        "name": "Свой сервер (OpenAI-compatible)",
        "protocol": "openai",
        "base_url": "",
        "docs": "",
        "hint": "Любой ваш URL: vLLM, LM Studio, прокси…",
    },
    "custom-anthropic": {
        "name": "Свой сервер (Anthropic-compatible)",
        "protocol": "anthropic",
        "base_url": "",
        "docs": "",
        "hint": "Любой клон Claude API",
    },
    "custom-gemini": {
        "name": "Свой сервер (Gemini-compatible)",
        "protocol": "gemini",
        "base_url": "",
        "docs": "",
        "hint": "Любой прокси Gemini API",
    },
    "openrouter": {
        "name": "OpenRouter",
        "protocol": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "docs": "https://openrouter.ai/keys",
        "hint": "Сотни моделей одним ключом",
    },
    "openai": {
        "name": "OpenAI",
        "protocol": "openai",
        "base_url": "https://api.openai.com/v1",
        "docs": "https://platform.openai.com/api-keys",
        "hint": "GPT-4o, GPT-4.1, o3",
    },
    "anthropic": {
        "name": "Anthropic",
        "protocol": "anthropic",
        "base_url": "https://api.anthropic.com",
        "docs": "https://console.anthropic.com/settings/keys",
        "hint": "Claude Opus / Sonnet / Haiku",
    },
    "google": {
        "name": "Google Gemini",
        "protocol": "gemini",
        "base_url": "https://generativelanguage.googleapis.com",
        "docs": "https://aistudio.google.com/apikey",
        "hint": "Gemini 2.5 Pro / Flash",
    },
    "zai": {
        "name": "Z.AI (Zhipu)",
        "protocol": "openai",
        "base_url": "https://api.z.ai/api/paas/v4",
        "docs": "https://z.ai/manage-apikey/apikey-list",
        "hint": "GLM-4.6, GLM-4.5",
    },
    "groq": {
        "name": "Groq",
        "protocol": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "docs": "https://console.groq.com/keys",
        "hint": "Llama, Qwen — быстрый инференс",
    },
    "deepseek": {
        "name": "DeepSeek",
        "protocol": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "docs": "https://platform.deepseek.com/api_keys",
        "hint": "DeepSeek-V3, R1",
    },
    "mistral": {
        "name": "Mistral AI",
        "protocol": "openai",
        "base_url": "https://api.mistral.ai/v1",
        "docs": "https://console.mistral.ai/api-keys",
        "hint": "Mistral Large / Medium",
    },
    "xai": {
        "name": "xAI (Grok)",
        "protocol": "openai",
        "base_url": "https://api.x.ai/v1",
        "docs": "https://console.x.ai",
        "hint": "Grok-4, Grok-3",
    },
    "moonshot": {
        "name": "Moonshot (Kimi)",
        "protocol": "openai",
        "base_url": "https://api.moonshot.ai/v1",
        "docs": "https://platform.moonshot.ai/console/api-keys",
        "hint": "Kimi K2",
    },
    "together": {
        "name": "Together AI",
        "protocol": "openai",
        "base_url": "https://api.together.xyz/v1",
        "docs": "https://api.together.ai/settings/api-keys",
        "hint": "Open-source модели",
    },
    "fireworks": {
        "name": "Fireworks AI",
        "protocol": "openai",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "docs": "https://fireworks.ai/account/api-keys",
        "hint": "Хостинг open-source",
    },
    "cerebras": {
        "name": "Cerebras",
        "protocol": "openai",
        "base_url": "https://api.cerebras.ai/v1",
        "docs": "https://cloud.cerebras.ai",
        "hint": "Сверхбыстрый инференс",
    },
}

HELP_TEXT = "\n".join([
    "👋 Я AI-бот с подключаемыми провайдерами (как в opencode).",
    "",
    "💬 Просто напишите сообщение — я отвечу через выбранную модель.",
    "",
    "Команды:",
    "/model — выбрать модель для этого чата",
    "/status — какая модель используется",
    "/clear — очистить историю диалога",
    "/providers — список подключённых провайдеров",
    "",
    "Для администраторов:",
    "/add — подключить провайдера (мастер с пресетами)",
    "/del — удалить провайдера",
    "/default — модель по умолчанию для новых чатов",
    "/key <id> <ключ> — сменить API-ключ провайдера",
    "/url <id> <url> — сменить baseURL провайдера",
    "/sys <текст> — системный промпт (/sys off — выключить)",
    "/lock — переключить режим «только администраторы»",
    "/id — показать ID (чата и ваш)",
    "/cancel — отменить текущий мастер",
])


class ProviderError(RuntimeError):
    """Ошибка на стороне провайдера (показываем пользователю как есть)."""


class TgError(RuntimeError):
    """Ошибка Telegram Bot API."""


# ----------------------------------------------------------------------------
# Работа с файлами (config.json / state.json)
# ----------------------------------------------------------------------------

def default_config() -> Dict[str, Any]:
    return {
        "bot_token": "",
        "admins": [],            # Telegram user id; пусто = первый /start станет владельцем
        "admins_only": False,    # True = общаться могут только администраторы
        "system_prompt": "",
        "max_history": 24,       # сообщений (пар) в истории на чат
        "default": {"provider": "", "model": ""},
        "providers": {},         # pid -> {name, protocol, base_url, api_key, enabled}
    }


def _atomic_write(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if fallback:
                # для config.json: объединяем с дефолтами (недостающие ключи добавятся)
                merged = json.loads(json.dumps(fallback))
                merged.update(data)
                return merged
            return data  # для state.json
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"⚠️ Не удалось прочитать {path}: {e} — начинаю с чистого файла")
    return json.loads(json.dumps(fallback))  # deep copy


def apply_env_overrides(cfg: Dict[str, Any]) -> None:
    """Настройки из окружения — источник истины на Render.

    Применяются при КАЖДОМ старте, если переменная задана (перекрывают
    значения, изменённые в Telegram через /sys, /lock и т.д.).
    Хотите управлять из Telegram — удалите переменную из Render.
    """
    applied: List[str] = []
    ids_raw = os.environ.get("ADMIN_IDS", "").strip()
    if ids_raw:
        ids: List[int] = []
        for x in re.split(r"[,;\s]+", ids_raw):
            if x.lstrip("-").isdigit():
                ids.append(int(x))
        if ids:
            cfg["admins"] = ids
            applied.append("ADMIN_IDS")
    sp = os.environ.get("SYSTEM_PROMPT", "").strip()
    if sp:
        cfg["system_prompt"] = sp
        applied.append("SYSTEM_PROMPT")
    mh = os.environ.get("MAX_HISTORY", "").strip()
    if mh.isdigit():
        cfg["max_history"] = max(2, int(mh))
        applied.append("MAX_HISTORY")
    ao = os.environ.get("ADMIN_ONLY", "").strip().lower()
    if ao in ("1", "true", "yes"):
        cfg["admins_only"] = True
        applied.append("ADMIN_ONLY=1")
    elif ao in ("0", "false", "no"):
        cfg["admins_only"] = False
        applied.append("ADMIN_ONLY=0")
    if applied:
        print("Из окружения применены: " + ", ".join(applied))


def parse_provider_spec(raw: str) -> Optional[Dict[str, str]]:
    """Разбор описания провайдера из переменной PROVIDER_N.

    Поддерживаются две формы:
      1) «пресет:ключ»       → openrouter:sk-or-… (URL и протокол из пресета)
      2) «Имя|протокол|URL|ключ» → Свой|openai|https://api.example.com/v1|key
    """
    raw = raw.strip()
    if not raw:
        return None
    # форма 1: пресет:ключ
    if "|" not in raw and ":" in raw:
        preset_id, _, key = raw.partition(":")
        pid = preset_id.strip().lower()
        if pid == "gemini":
            pid = "google"
        preset = PRESETS.get(pid)
        if preset and preset.get("base_url"):
            return {
                "name": preset["name"],
                "protocol": preset["protocol"],
                "base_url": preset["base_url"],
                "api_key": key.strip(),
            }
    # форма 2: Имя|протокол|URL|ключ
    parts = raw.split("|")
    if len(parts) >= 3:
        name, proto, url = parts[0].strip(), parts[1].strip().lower(), parts[2].strip()
        key = parts[3].strip() if len(parts) > 3 else ""
        if proto == "google":
            proto = "gemini"
        if proto in PROTOCOLS and url.startswith(("http://", "https://")):
            return {"name": name, "protocol": proto, "base_url": url.rstrip("/"), "api_key": key}
    return None


def parse_env_providers() -> List[Dict[str, Any]]:
    """Собрать все непустые PROVIDER_N (N=1..64), порядок — по номеру."""
    items: List[Tuple[int, str]] = []
    for i in range(1, 65):
        raw = os.environ.get(f"PROVIDER_{i}", "").strip()
        if raw:
            items.append((i, raw))
    items.sort()
    out: List[Dict[str, Any]] = []
    for num, raw in items:
        spec = parse_provider_spec(raw)
        if spec:
            spec["_idx"] = num  # id провайдера = номер переменной: PROVIDER_5 → e5
            out.append(spec)
        else:
            print(f"⚠️ PROVIDER_{num}: не понял формат «{raw[:60]}…». "
                  "Примеры: openrouter:sk-… или Имя|openai|https://…/v1|ключ")
    return out


def sync_env_providers(cfg: Dict[str, Any]) -> None:
    """Провайдеры из PROVIDER_1..N и CUSTOM_* — применяются при каждом старте.

    id = номеру переменной (e1 = PROVIDER_1, e5 = PROVIDER_5, пропуски — ок);
    свой провайдер с произвольным URL — CUSTOM_URL/CUSTOM_KEY/… (id ec);
    провайдеры, добавленные через /add (p1, p2…), не затрагиваются. Чтобы
    убрать env-провайдера — уберите его переменную из Render (удаление
    через /del переживёт только до рестарта).
    """
    specs = parse_env_providers()
    c_url = clean_secret(os.environ.get("CUSTOM_URL", "")).rstrip("/")
    if not specs and not c_url:
        return
    providers = cfg["providers"]
    for k in [k for k in providers if k.startswith("e")]:
        del providers[k]
    names: List[str] = []
    for spec in specs:
        pid = f"e{spec['_idx']}"
        name, base = spec["name"], spec["name"]
        n = 2
        while any(p["name"] == name for k, p in providers.items() if k != pid):
            name = f"{base} {n}"
            n += 1
        providers[pid] = {
            "name": name,
            "protocol": spec["protocol"],
            "base_url": spec["base_url"],
            "api_key": spec["api_key"],
            "enabled": True,
        }
        names.append(f"{pid}={name}")

    # --- свой провайдер с произвольным URL (CUSTOM_*), id "ec" — ни от кого не зависит
    if c_url:
        if not c_url.startswith(("http://", "https://")):
            print("⚠️ CUSTOM_URL должен начинаться с http:// или https:// — провайдер пропущен")
        else:
            proto = clean_secret(os.environ.get("CUSTOM_PROTOCOL", "")).lower()
            if proto == "google":
                proto = "gemini"
            if proto not in PROTOCOLS:
                if proto:
                    print(f"⚠️ CUSTOM_PROTOCOL={proto}: знаю только openai/anthropic/gemini — беру openai")
                proto = "openai"
            cname = clean_secret(os.environ.get("CUSTOM_NAME", "")) or "Мой провайдер"
            base_name, n = cname, 2
            while any(p["name"] == cname for p in providers.values()):
                cname = f"{base_name} {n}"
                n += 1
            providers["ec"] = {
                "name": cname,
                "protocol": proto,
                "base_url": c_url,
                "api_key": clean_secret(os.environ.get("CUSTOM_KEY", "")),
                "enabled": True,
            }
            names.append(f"ec={cname}")
            cm = clean_secret(os.environ.get("CUSTOM_MODEL", ""))
            if cm and not os.environ.get("DEFAULT_MODEL", "").strip():
                cfg["default"] = {"provider": "ec", "model": cm}
                names.append(f"по умолчанию: {cm}")

    # модель по умолчанию из окружения (DEFAULT_MODEL / DEFAULT_PROVIDER)
    dm = os.environ.get("DEFAULT_MODEL", "").strip()
    if dm:
        dp = os.environ.get("DEFAULT_PROVIDER", "").strip()
        match = None
        for pid, p in providers.items():
            if pid == dp or p["name"].lower() == dp.lower():
                match = pid
                break
        if not match:
            match = "e1" if "e1" in providers else next(iter(providers), "")
        if match:
            cfg["default"] = {"provider": match, "model": dm}
            names.append(f"по умолчанию: {dm}")

    _atomic_write(CONFIG_PATH, cfg)
    print("Провайдеры из окружения: " + ", ".join(names))


def clean_secret(s: str) -> str:
    """Убрать пробелы по краям и случайные обрамляющие кавычки.

    Частая ошибка при вставке значений в дашборд Render — значение вида
    "123:ABC" вместе с кавычками; Telegram тогда видит мусор в токене.
    """
    s = (s or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def tg_hint(err: str) -> str:
    """Человекочитаемая подсказка по ошибке Telegram Bot API (для логов при старте)."""
    low = err.lower()
    if "401" in err or "unauthorized" in low:
        return ("Токен неверного формата — должен быть вида 123456789:ABC-DEF… "
                "(без кавычек и пробелов). Проверьте BOT_TOKEN.")
    if "404" in err or "not found" in low:
        return ("Telegram не находит бота с этим токеном. Проверьте бота у @BotFather (/mybots); "
                "если токен отозван — выпустите новый (/token) и обновите BOT_TOKEN в Render.")
    if "409" in err or "conflict" in low:
        return ("Конфликт: webhook или getUpdates заняты другим экземпляром бота. "
                "Остановите второй экземпляр и перезапустите сервис.")
    if "429" in err or "too many requests" in low:
        return "Слишком много запросов к Telegram. Подождите пару минут и сделайте Manual Deploy → Restart."
    if "400" in err or "bad request" in low:
        return ("Telegram отклонил параметры запроса (обычно URL вебхука). Проверьте, что "
                "RENDER_EXTERNAL_URL / WEBHOOK_URL — валидный https-адрес без опечаток.")
    return "Проверьте BOT_TOKEN и настройки сервиса, затем Manual Deploy → Restart."


def first_run_setup(cfg: Dict[str, Any]) -> None:
    """Токен: приоритет у окружения BOT_TOKEN, иначе интерактивно."""
    env = clean_secret(os.environ.get("BOT_TOKEN", ""))
    if env:
        if not re.fullmatch(r"\d+:[\w-]+", env):
            print("⚠️ BOT_TOKEN не похож на токен Telegram (вида 123456789:ABC-DEF…). "
                  "Проверьте переменную — нет ли лишних символов, кавычек или пробелов.")
        if cfg.get("bot_token") != env:
            first = not cfg.get("bot_token")
            cfg["bot_token"] = env
            _atomic_write(CONFIG_PATH, cfg)
            print("Токен обновлён из переменной окружения BOT_TOKEN" if not first
                  else "Токен взят из переменной окружения BOT_TOKEN")
        return
    if cfg["bot_token"]:
        return
    if not (sys.stdin and sys.stdin.isatty()):
        sys.exit("❌ BOT_TOKEN не задан. Укажите переменную окружения BOT_TOKEN "
                 "(токен от @BotFather) или вписать его в config.json.")
    print("=" * 60)
    print("Первый запуск. Создайте бота у @BotFather и вставьте его токен.")
    print("(токен будет сохранён в config.json рядом со скриптом)")
    print("=" * 60)
    try:
        token = clean_secret(input("Токен (123456789:ABC-DEF…): "))
    except (EOFError, KeyboardInterrupt):
        print("\nОтмена. Можно вписать токен в config.json вручную.")
        sys.exit(1)
    if not re.fullmatch(r"\d+:[\w-]+", token):
        print("Токен выглядит странно, но сохраню как есть — если что, поправьте config.json")
    cfg["bot_token"] = token
    _atomic_write(CONFIG_PATH, cfg)


# ----------------------------------------------------------------------------
# Клиенты провайдеров (три протокола, как в opencode)
# ----------------------------------------------------------------------------

async def _json_or_error(resp: "aiohttp.ClientResponse") -> Any:
    if resp.status >= 400:
        text = (await resp.text())[:300].replace("\n", " ")
        raise ProviderError(f"HTTP {resp.status}: {text}")
    try:
        return await resp.json(content_type=None)
    except Exception:
        raise ProviderError(f"HTTP {resp.status}: ответ не JSON")


async def _sse_data_lines(resp: "aiohttp.ClientResponse") -> AsyncGenerator[str, None]:
    """Разбор SSE-потока: отдаёт содержимое строк 'data: ...'."""
    buf = b""
    async for chunk in resp.content.iter_any():
        buf += chunk
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            line = raw.decode("utf-8", "replace").strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    yield payload


async def fetch_models(session: "aiohttp.ClientSession", provider: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Список моделей провайдера: [(model_id, display_name), ...]"""
    proto = provider["protocol"]
    base = provider["base_url"].rstrip("/")
    key = provider.get("api_key", "")

    if proto == "gemini":
        url = f"{base}/v1beta/models?pageSize=200"
        if key:
            url += f"&key={key}"
        async with session.get(url) as r:
            data = await _json_or_error(r)
        out: List[Tuple[str, str]] = []
        for m in data.get("models", []):
            methods = m.get("supportedGenerationMethods") or []
            if methods and "generateContent" not in methods:
                continue
            name = m.get("name", "")
            mid = name[len("models/"):] if name.startswith("models/") else name
            if mid:
                out.append((mid, m.get("displayName") or mid))
        return out

    if proto == "anthropic":
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        async with session.get(f"{base}/v1/models?limit=1000", headers=headers) as r:
            data = await _json_or_error(r)
        out = []
        for m in data.get("data", []):
            mid = m.get("id") or ""
            if mid:
                out.append((mid, m.get("display_name") or m.get("name") or mid))
        return out

    # openai-compatible
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with session.get(f"{base}/models", headers=headers) as r:
        data = await _json_or_error(r)
    items = data.get("data") or data.get("models") or []
    out = []
    for m in items:
        mid = m.get("id") or m.get("name")
        if mid:
            out.append((str(mid), str(mid)))
    return out


async def stream_chat(
    session: "aiohttp.ClientSession",
    provider: Dict[str, Any],
    model: str,
    messages: List[Dict[str, str]],
) -> AsyncGenerator[str, None]:
    """Стриминг ответа модели: отдаёт кусочки текста (deltas)."""
    proto = provider["protocol"]
    base = provider["base_url"].rstrip("/")
    key = provider.get("api_key", "")
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=300)

    if proto == "openai":
        url = f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if "openrouter.ai" in base:
            headers["HTTP-Referer"] = "https://github.com/tg-ai-bot"
            headers["X-Title"] = "TG AI Bot"
        payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": True}
    elif proto == "anthropic":
        url = f"{base}/v1/messages"
        headers = {"Content-Type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"}
        payload = {
            "model": model,
            "max_tokens": MAX_TOKENS_ANTHROPIC,
            "messages": [{"role": m["role"], "content": m["content"]} for m in rest],
            "stream": True,
        }
        if system:
            payload["system"] = system
    elif proto == "gemini":
        url = f"{base}/v1beta/models/{model}:streamGenerateContent?alt=sse"
        if key:
            url += f"&key={key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "role": "model" if m["role"] == "assistant" else "user",
                    "parts": [{"text": m["content"]}],
                }
                for m in rest
            ]
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
    else:
        raise ProviderError(f"Неизвестный протокол: {proto}")

    async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
        if resp.status >= 400:
            text = (await resp.text())[:400].replace("\n", " ")
            raise ProviderError(f"HTTP {resp.status}: {text}")

        async for line in _sse_data_lines(resp):
            if line == "[DONE]":
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if proto == "openai":
                choices = obj.get("choices") or [{}]
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    yield delta
            elif proto == "anthropic":
                t = obj.get("type")
                if t == "content_block_delta":
                    d = obj.get("delta") or {}
                    piece = d.get("text") or d.get("partial_json")
                    if piece:
                        yield piece
                elif t == "error":
                    raise ProviderError(json.dumps(obj.get("error"), ensure_ascii=False)[:300])
            else:  # gemini
                for cand in obj.get("candidates") or []:
                    for part in ((cand.get("content") or {}).get("parts") or []):
                        piece = part.get("text")
                        if piece:
                            yield piece


# ----------------------------------------------------------------------------
# Бот
# ----------------------------------------------------------------------------

class App:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.state: Dict[str, Any] = load_json(STATE_PATH, {})
        self.session: Optional[aiohttp.ClientSession] = None
        self.me: Dict[str, Any] = {}
        # dialogи мастеров: chat_id -> {"action": "add", "step": ..., "data": {...}}
        self.dialogs: Dict[int, Dict[str, Any]] = {}
        # кэш моделей: pid -> (timestamp, [(model_id, name)])
        self.models_cache: Dict[str, Tuple[float, List[Tuple[str, str]]]] = {}
        self._save_lock = asyncio.Lock()

    # ---------- persistence ----------

    def save_config(self) -> None:
        _atomic_write(CONFIG_PATH, self.cfg)

    def save_state(self) -> None:
        _atomic_write(STATE_PATH, self.state)

    # ---------- telegram api ----------

    async def api(self, method: str, http_timeout: float = 60.0, **params: Any) -> Any:
        assert self.session is not None
        url = TG_API.format(token=self.cfg["bot_token"], method=method)
        try:
            async with self.session.post(
                url, json=params, timeout=aiohttp.ClientTimeout(total=http_timeout)
            ) as r:
                try:
                    data = await r.json(content_type=None)
                except Exception:
                    raise TgError(f"HTTP {r.status}: не-JSON ответ")
        except asyncio.TimeoutError:
            raise TgError("таймаут запроса к Telegram")
        except aiohttp.ClientError as e:
            raise TgError(f"сеть: {e}")
        if not data.get("ok"):
            raise TgError(str(data.get("description", "unknown error")))
        return data.get("result")

    async def api_with_retry(self, method: str, attempts: int = 3, delay: float = 4.0,
                             fatal_markers: Tuple[str, ...] = (), **params: Any) -> Any:
        """api() с повторами: transient-сбои сети/Telegram переживаются,
        ошибки из fatal_markers (невалидный токен, плохие параметры) — сразу наверх."""
        err: Optional[TgError] = None
        for i in range(max(1, attempts)):
            try:
                return await self.api(method, **params)
            except TgError as e:
                err = e
                s = str(e).lower()
                if any(m.lower() in s for m in fatal_markers):
                    raise
                if i < attempts - 1:
                    print(f"⚠️ Telegram {method}: {e} — повтор {i + 2}/{attempts} через {delay:.0f}с")
                    await asyncio.sleep(delay)
        raise err  # type: ignore[misc]

    async def send(self, chat_id: int, text: str, reply_markup: Optional[Dict] = None) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text[:MAX_TG],
            "disable_web_page_preview": True,
        }
        if reply_markup:
            kwargs["reply_markup"] = reply_markup
        return await self.api("sendMessage", **kwargs)

    async def edit(self, chat_id: int, message_id: int, text: str, reply_markup: Optional[Dict] = None) -> None:
        kwargs: Dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text[:MAX_TG],
            "disable_web_page_preview": True,
        }
        if reply_markup is not None:
            kwargs["reply_markup"] = reply_markup
        try:
            await self.api("editMessageText", **kwargs)
        except TgError as e:
            if "message is not modified" in str(e).lower():
                return
            raise

    # ---------- доступ ----------

    def is_admin(self, user_id: int) -> bool:
        return user_id in self.cfg.get("admins", [])

    def can_chat(self, user_id: int) -> bool:
        if not self.cfg.get("admins_only"):
            return True
        return self.is_admin(user_id)

    # ---------- провайдеры ----------

    def next_pid(self) -> str:
        i = 1
        while f"p{i}" in self.cfg["providers"]:
            i += 1
        return f"p{i}"

    @staticmethod
    def mask_key(key: str) -> str:
        if not key:
            return "без ключа"
        if len(key) <= 8:
            return "•••"
        return f"{key[:4]}…{key[-4:]}"

    async def get_models(self, pid: str, force: bool = False) -> List[Tuple[str, str]]:
        """Список моделей с кэшем."""
        cached = self.models_cache.get(pid)
        if cached and not force and time.monotonic() - cached[0] < MODELS_TTL:
            return cached[1]
        provider = self.cfg["providers"].get(pid)
        if not provider:
            return []
        if self.session is None:
            raise ProviderError("Бот ещё не готов, попробуйте через минуту")
        try:
            models = await fetch_models(self.session, provider)
        except ProviderError:
            raise
        except asyncio.TimeoutError:
            raise ProviderError("таймаут запроса к провайдеру")
        except aiohttp.ClientError as e:
            raise ProviderError(f"ошибка сети: {e}")
        except RuntimeError as e:
            # напр. "Session is closed" у aiohttp
            raise ProviderError(f"ошибка запроса: {e}")
        self.models_cache[pid] = (time.monotonic(), models)
        return models

    def model_display(self, pid: str, model_id: str) -> str:
        cached = self.models_cache.get(pid)
        if cached:
            for mid, name in cached[1]:
                if mid == model_id:
                    return name
        return model_id

    def resolve(self, chat_id: int) -> Optional[Tuple[str, Dict[str, Any], str]]:
        """Модель для чата: явный выбор или глобальная настройка по умолчанию."""
        st = self.state.get(str(chat_id), {})
        pid = st.get("provider") or self.cfg["default"].get("provider", "")
        mid = st.get("model") or self.cfg["default"].get("model", "")
        provider = self.cfg["providers"].get(pid)
        if not provider or not provider.get("enabled", True) or not mid:
            # выбор чата невалиден — пробуем глобальный дефолт
            pid = self.cfg["default"].get("provider", "")
            mid = self.cfg["default"].get("model", "")
            provider = self.cfg["providers"].get(pid)
            if not provider or not mid:
                return None
        return pid, provider, mid

    # ---------- клавиатуры ----------

    @staticmethod
    def kb(rows: List[List[Tuple[str, str]]]) -> Dict[str, Any]:
        return {
            "inline_keyboard": [
                [{"text": t, "callback_data": d} for (t, d) in row]
                for row in rows
            ]
        }

    def kb_providers(self, pick_prefix: str, back_data: Optional[str] = None) -> Dict[str, Any]:
        rows: List[List[Tuple[str, str]]] = []
        for pid, p in self.cfg["providers"].items():
            if not p.get("enabled", True):
                continue
            cached = self.models_cache.get(pid)
            n = f" · {len(cached[1])} мод." if cached else ""
            rows.append([(f"{p['name']} ({pid}){n}", f"{pick_prefix}:{pid}")])
        if not rows:
            rows.append([("нет провайдеров — /add", "noop")])
        if back_data:
            rows.append([("⬅️ Назад", back_data)])
        return self.kb(rows)

    async def kb_models(self, pid: str, page: int, pick_prefix: str, back_data: str):
        try:
            models = await self.get_models(pid)
        except ProviderError as e:
            return None, str(e)
        if not models:
            return None, "Провайдер не вернул моделей. Проверьте ключ (/key) и baseURL (/url)."

        pages = max(1, (len(models) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = min(max(0, page), pages - 1)
        chunk = models[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
        rows: List[List[Tuple[str, str]]] = []
        for i, (mid, name) in enumerate(chunk):
            idx = page * PAGE_SIZE + i
            label = name if len(name) <= 60 else name[:57] + "…"
            rows.append([(label, f"{pick_prefix}:{pid}:{idx}")])
        nav: List[Tuple[str, str]] = []
        if page > 0:
            nav.append(("⬅️", f"{pick_prefix}pg:{pid}:{page - 1}"))
        nav.append((f"{page + 1}/{pages}", "noop"))
        if page < pages - 1:
            nav.append(("➡️", f"{pick_prefix}pg:{pid}:{page + 1}"))
        rows.append(nav)
        rows.append([("⬅️ К провайдерам", back_data)])
        return self.kb(rows), None

    def idx_to_model(self, pid: str, idx: int) -> Optional[Tuple[str, str]]:
        cached = self.models_cache.get(pid)
        if not cached or idx < 0 or idx >= len(cached[1]):
            return None
        return cached[1][idx]

    # ---------- история ----------

    def chat_state(self, chat_id: int) -> Dict[str, Any]:
        return self.state.setdefault(str(chat_id), {"provider": "", "model": "", "history": []})

    # ---------- генерация ----------

    async def typing_loop(self, chat_id: int, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.api("sendChatAction", http_timeout=10, chat_id=chat_id, action="typing")
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                pass

    async def generate_and_send(self, msg: Dict[str, Any], user_text: str) -> None:
        chat_id = msg["chat"]["id"]
        resolved = self.resolve(chat_id)
        if not resolved:
            await self.send(
                chat_id,
                "⚠️ Модель не выбрана.\n\nИспользуйте /model — выбрать модель\n"
                "Или /add (если вы админ) — подключить провайдера",
            )
            return
        pid, provider, model_id = resolved
        display = self.model_display(pid, model_id)

        st = self.chat_state(chat_id)
        history: List[Dict[str, str]] = st.get("history", [])
        max_hist = int(self.cfg.get("max_history", 24) or 24)
        sys_prompt = self.cfg.get("system_prompt", "")

        messages: List[Dict[str, str]] = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages += history
        messages.append({"role": "user", "content": user_text})

        placeholder = await self.send(chat_id, f"⏳ {provider['name']} · {display}\nДумаю…")
        msg_id = placeholder["message_id"]

        stop = asyncio.Event()
        typing_task = asyncio.create_task(self.typing_loop(chat_id, stop))
        full = ""
        buf = ""
        last_edit = 0.0
        try:
            async for delta in stream_chat(self.session, provider, model_id, messages):  # type: ignore[arg-type]
                buf += delta
                now = time.monotonic()
                if now - last_edit >= EDIT_EVERY:
                    full += buf
                    buf = ""
                    last_edit = now
                    shown = (full + buf)[-MAX_TG:]
                    try:
                        await self.edit(chat_id, msg_id, shown + " ▌")
                    except TgError:
                        pass
            full += buf

            if not full:
                await self.edit(chat_id, msg_id, "⚠️ Пустой ответ от модели.")
                return

            # финальный текст (разбивка длинных ответов)
            if len(full) <= MAX_TG:
                await self.edit(chat_id, msg_id, full)
            else:
                chunks = []
                rest = full
                while rest:
                    chunks.append(rest[:MAX_TG])
                    rest = rest[MAX_TG:]
                await self.edit(chat_id, msg_id, chunks[0])
                for c in chunks[1:]:
                    await self.send(chat_id, c)

            # сохраняем историю
            history = history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": full},
            ]
            st["history"] = history[-max_hist * 2:]
            self.save_state()
        except ProviderError as e:
            err = f"⚠️ Ошибка провайдера ({provider['name']}):\n{e}"
            try:
                await self.edit(chat_id, msg_id, err[:MAX_TG])
            except TgError:
                await self.send(chat_id, err[:MAX_TG])
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            err = f"⚠️ Непредвиденная ошибка:\n{e}"
            try:
                await self.edit(chat_id, msg_id, err[:MAX_TG])
            except TgError:
                await self.send(chat_id, err[:MAX_TG])
        finally:
            stop.set()
            typing_task.cancel()
            try:
                await typing_task
            except (asyncio.CancelledError, Exception):
                pass

    # ---------- обработка сообщений ----------

    async def on_message(self, msg: Dict[str, Any]) -> None:
        chat_id = msg["chat"]["id"]
        uid = (msg.get("from") or {}).get("id", 0)
        text = msg.get("text", "").strip()

        # первый пользователь в личке становится владельцем
        if not self.cfg.get("admins") and msg["chat"]["type"] == "private" and uid:
            self.cfg["admins"] = [uid]
            self.save_config()
            await self.send(
                chat_id,
                "✅ Вы назначены администратором бота.\n\nНачните с /add — подключите провайдера, "
                "затем /model — выберите модель.\n\n" + HELP_TEXT,
            )
            return

        if not text:
            await self.send(chat_id, "Пока я понимаю только текстовые сообщения 🙏")
            return

        # мастер добавления провайдера перехватывает текст
        dlg = self.dialogs.get(chat_id)
        if dlg:
            low = text.lower()
            if low.startswith("/cancel"):
                self.dialogs.pop(chat_id, None)
                await self.send(chat_id, "Мастер отменён.")
                return
            if not text.startswith("/") or low.startswith("/skip"):
                await self.wizard_text(msg, dlg, text)
                return

        m = re.match(r"^/([a-zA-Z_]+)(?:@\S+)?\s*(.*)$", text, re.S)
        if m:
            cmd, args = m.group(1).lower(), m.group(2).strip()
            handled = await self.dispatch_command(msg, cmd, args)
            if handled:
                return
            await self.send(chat_id, f"Неизвестная команда /{cmd}. Смотрите /help")
            return

        if not self.can_chat(uid):
            await self.send(chat_id, "⛔ Общение с ботом ограничено администраторами.")
            return
        await self.generate_and_send(msg, text)

    async def dispatch_command(self, msg: Dict[str, Any], cmd: str, args: str) -> bool:
        chat_id = msg["chat"]["id"]
        uid = (msg.get("from") or {}).get("id", 0)

        if cmd in ("start", "help"):
            await self.send(chat_id, HELP_TEXT)
            return True

        if cmd == "id":
            await self.send(chat_id, f"Ваш ID: {uid}\nID чата: {chat_id}")
            return True

        if cmd == "cancel":
            if self.dialogs.pop(chat_id, None):
                await self.send(chat_id, "Мастер отменён.")
            else:
                await self.send(chat_id, "Нечего отменять.")
            return True

        if cmd == "model":
            extra = []
            st = self.chat_state(chat_id)
            if st.get("provider") and st.get("model"):
                extra = [("⟲ Сбросить на модель по умолчанию", "seldef")]
            kb = self.kb_providers("pv")
            if extra:
                kb["inline_keyboard"] = [extra] + kb["inline_keyboard"]
            await self.send(chat_id, "🧩 Выберите провайдера:", kb)
            return True

        if cmd == "status":
            st = self.chat_state(chat_id)
            r = self.resolve(chat_id)
            if r:
                pid, provider, mid = r
                src = "выбрана в этом чате" if (st.get("model") or st.get("provider")) else "по умолчанию"
                await self.send(
                    chat_id,
                    f"📌 Модель {src}:\n{provider['name']} → {self.model_display(pid, mid)}",
                )
            else:
                await self.send(chat_id, "⚠️ Модель не выбрана. Используйте /model")
            return True

        if cmd == "clear":
            st = self.chat_state(chat_id)
            st["history"] = []
            self.save_state()
            await self.send(chat_id, "🧹 История диалога очищена.")
            return True

        if cmd == "providers":
            lines = ["🧩 Подключённые провайдеры:"]
            if not self.cfg["providers"]:
                lines.append("— пока none. Админ может добавить через /add")
            for pid, p in self.cfg["providers"].items():
                cached = self.models_cache.get(pid)
                n = str(len(cached[1])) + " мод." if cached else "модели не загружены"
                flag = "" if p.get("enabled", True) else " (выкл)"
                lines.append(
                    f"• {pid} — {p['name']} [{PROTOCOL_TITLES.get(p['protocol'], p['protocol'])}]{flag}\n"
                    f"  {p['base_url']} · ключ: {self.mask_key(p.get('api_key', ''))} · {n}"
                )
            await self.send(chat_id, "\n".join(lines))
            return True

        # ----- дальше только для админов -----
        if cmd == "add":
            if not self.is_admin(uid):
                await self.send(chat_id, "⛔ Команда только для администраторов.")
                return True
            await self.wizard_start(chat_id)
            return True

        if cmd == "del":
            if not self.is_admin(uid):
                await self.send(chat_id, "⛔ Команда только для администраторов.")
                return True
            rows = [
                [(f"{p['name']} ({pid})", f"del:{pid}")]
                for pid, p in self.cfg["providers"].items()
            ] or [[("нет провайдеров", "noop")]]
            await self.send(chat_id, "🗑 Какого провайдера удалить?", self.kb(rows))
            return True

        if cmd == "default":
            if not self.is_admin(uid):
                await self.send(chat_id, "⛔ Команда только для администраторов.")
                return True
            await self.send(chat_id, "⭐ Провайдер по умолчанию:", self.kb_providers("defpv", "noop"))
            return True

        if cmd == "key":
            if not self.is_admin(uid):
                await self.send(chat_id, "⛔ Команда только для администраторов.")
                return True
            parts = args.split(maxsplit=1)
            if len(parts) < 2 or parts[0] not in self.cfg["providers"]:
                await self.send(chat_id, "Формат: /key <id> <новый ключ>\nID смотрите в /providers")
                return True
            self.cfg["providers"][parts[0]]["api_key"] = parts[1].strip()
            self.models_cache.pop(parts[0], None)
            self.save_config()
            await self.send(chat_id, f"✅ Ключ {parts[0]} обновлён. Проверить: /model")
            return True

        if cmd == "url":
            if not self.is_admin(uid):
                await self.send(chat_id, "⛔ Команда только для администраторов.")
                return True
            parts = args.split(maxsplit=1)
            if len(parts) < 2 or parts[0] not in self.cfg["providers"]:
                await self.send(chat_id, "Формат: /url <id> <baseURL>\nНапример: /url p1 https://openrouter.ai/api/v1")
                return True
            url = parts[1].strip().rstrip("/")
            if not url.startswith(("http://", "https://")):
                await self.send(chat_id, "URL должен начинаться с http:// или https://")
                return True
            self.cfg["providers"][parts[0]]["base_url"] = url
            self.models_cache.pop(parts[0], None)
            self.save_config()
            await self.send(chat_id, f"✅ baseURL {parts[0]} → {url}")
            return True

        if cmd == "sys":
            if not self.is_admin(uid):
                await self.send(chat_id, "⛔ Команда только для администраторов.")
                return True
            if not args or args.lower() in ("off", "выкл"):
                self.cfg["system_prompt"] = ""
                self.save_config()
                await self.send(chat_id, "Системный промпт выключен.")
            else:
                self.cfg["system_prompt"] = args
                self.save_config()
                await self.send(chat_id, f"✅ Системный промпт сохранён:\n{args[:500]}")
            return True

        if cmd == "lock":
            if not self.is_admin(uid):
                await self.send(chat_id, "⛔ Команда только для администраторов.")
                return True
            self.cfg["admins_only"] = not self.cfg.get("admins_only", False)
            self.save_config()
            state = "включён — писать могут только администраторы" if self.cfg["admins_only"] else "выключен — писать могут все"
            await self.send(chat_id, f"🔒 Режим «только администраторы» {state}.")
            return True

        return False

    # ---------- мастер добавления провайдера ----------

    async def wizard_start(self, chat_id: int) -> None:
        self.dialogs[chat_id] = {"action": "add", "step": "preset", "data": {}}
        rows: List[List[Tuple[str, str]]] = []
        preset_items = list(PRESETS.items())
        for i in range(0, len(preset_items), 2):
            row = []
            for pid, p in preset_items[i:i + 2]:
                row.append((p["name"], f"pre:{pid}"))
            rows.append(row)
        await self.send(
            chat_id,
            "🧩 Подключение провайдера.\n"
            "«Свой сервер» — любой ваш URL и ключ, ни от кого не зависит.\n"
            "Остальные — пресеты: выбрал → вставил только API-ключ:",
            self.kb(rows),
        )

    async def wizard_text(self, msg: Dict[str, Any], dlg: Dict[str, Any], text: str) -> None:
        chat_id = msg["chat"]["id"]
        data = dlg["data"]
        step = dlg["step"]

        if step == "url":
            url = text.strip().rstrip("/")
            if not url.startswith(("http://", "https://")):
                await self.send(chat_id, "URL должен начинаться с http:// или https://. Попробуйте ещё раз (или /cancel):")
                return
            data["base_url"] = url
            dlg["step"] = "key"
            await self.send(
                chat_id,
                "🔑 Теперь пришлите API-ключ (или /skip, если сервер без ключа):",
            )
            return

        if step == "key":
            key = "" if text.strip().lower().lstrip("/") == "skip" else text.strip()
            data["api_key"] = key
            pid = self.next_pid()
            name = data["name"]
            base = name
            n = 2
            while any(p["name"] == name for p in self.cfg["providers"].values()):
                name = f"{base} {n}"
                n += 1
            self.cfg["providers"][pid] = {
                "name": name,
                "protocol": data["protocol"],
                "base_url": data["base_url"],
                "api_key": key,
                "enabled": True,
            }
            self.save_config()
            self.dialogs.pop(chat_id, None)

            status = f"✅ Провайдер «{name}» сохранён (id: {pid})."
            await self.send(chat_id, f"{status}\nЗагружаю список моделей…")
            try:
                models = await self.get_models(pid, force=True)
                if models:
                    await self.send(chat_id, f"📦 Загружено моделей: {len(models)}\n\nТеперь: /model — выбрать модель, или /default — сделать её дефолтной.")
                else:
                    await self.send(chat_id, "⚠️ Список моделей пуст. Проверьте ключ (/key) и адрес (/url).")
            except ProviderError as e:
                await self.send(
                    chat_id,
                    f"⚠️ Не удалось получить модели:\n{e}\n\n"
                    f"Провайдер сохранён. Поправьте ключ: /key {pid} <ключ>\nили адрес: /url {pid} <baseURL>",
                )
            return

        self.dialogs.pop(chat_id, None)
        await self.send(chat_id, "Мастер сброшен. Начните заново: /add")

    # ---------- callback-кнопки ----------

    async def on_callback(self, cq: Dict[str, Any]) -> None:
        chat_id = (cq.get("message") or {}).get("chat", {}).get("id")
        if not chat_id:
            return
        uid = (cq.get("from") or {}).get("id", 0)
        data = cq.get("data", "")

        try:
            await self.api("answerCallbackQuery", http_timeout=10, callback_query_id=cq["id"])
        except TgError:
            pass

        if data == "noop":
            return

        msg = cq.get("message") or {}

        # --- мастер /add ---
        if data.startswith("pre:"):
            if not self.is_admin(uid):
                await self.send(chat_id, "⛔ Только для администраторов.")
                return
            preset_id = data[4:]
            preset = PRESETS.get(preset_id)
            if not preset:
                return
            dlg = self.dialogs.setdefault(chat_id, {"action": "add", "data": {}})
            dlg["action"] = "add"
            dlg["data"] = {
                "name": preset["name"],
                "protocol": preset["protocol"],
                "base_url": preset["base_url"],
                "preset_id": preset_id,
            }
            if preset["base_url"]:
                dlg["step"] = "key"
                docs = f"\nКлюч: {preset['docs']}" if preset.get("docs") else ""
                await self.edit(
                    chat_id,
                    msg.get("message_id", 0),
                    f"✅ {preset['name']} — {preset['hint']}{docs}\n\n🔑 Пришлите API-ключ (или /skip):",
                )
            else:
                dlg["step"] = "url"
                await self.edit(
                    chat_id,
                    msg.get("message_id", 0),
                    f"✅ {preset['name']}\nПришлите baseURL, например:\nhttps://api.example.com/v1",
                )
            return

        # --- /model ---
        if data == "seldef":
            st = self.chat_state(chat_id)
            st["provider"] = ""
            st["model"] = ""
            self.save_state()
            r = self.resolve(chat_id)
            if r:
                await self.edit(chat_id, msg.get("message_id", 0),
                                f"✅ Будет использоваться модель по умолчанию:\n{r[1]['name']} → {self.model_display(r[0], r[2])}")
            else:
                await self.edit(chat_id, msg.get("message_id", 0), "Модель по умолчанию не задана. Админ может задать: /default")
            return

        if data.startswith("pv:"):
            pid = data[3:]
            provider = self.cfg["providers"].get(pid)
            if not provider:
                await self.edit(chat_id, msg.get("message_id", 0), "⚠️ Провайдер удалён. Откройте /model заново.")
                return
            kb, err = await self.kb_models(pid, 0, "sel", "bk")
            if err:
                await self.edit(chat_id, msg.get("message_id", 0), f"⚠️ {err}")
                return
            await self.edit(chat_id, msg.get("message_id", 0),
                            f"🤖 {provider['name']} — выберите модель:", kb)
            return

        if data.startswith("sel:"):
            _, pid, idx_s = data.split(":", 2)
            if pid not in self.cfg["providers"]:
                await self.edit(chat_id, msg.get("message_id", 0), "⚠️ Провайдер удалён. Откройте /model заново.")
                return
            item = self.idx_to_model(pid, int(idx_s))
            if not item:
                await self.edit(chat_id, msg.get("message_id", 0), "⚠️ Список устарел, откройте /model заново.")
                return
            mid, _name = item
            st = self.chat_state(chat_id)
            st["provider"] = pid
            st["model"] = mid
            self.save_state()
            await self.edit(
                chat_id,
                msg.get("message_id", 0),
                f"✅ Модель выбрана:\n{self.cfg['providers'][pid]['name']} → {self.model_display(pid, mid)}\n\nТеперь просто пишите сообщения.",
            )
            return

        if data == "bk":
            await self.edit(chat_id, msg.get("message_id", 0), "🧩 Выберите провайдера:", self.kb_providers("pv"))
            return

        # --- /default ---
        if data.startswith("defpv:"):
            pid = data[6:]
            kb, err = await self.kb_models(pid, 0, "defsel", "defbk")
            if err:
                await self.edit(chat_id, msg.get("message_id", 0), f"⚠️ {err}")
                return
            await self.edit(chat_id, msg.get("message_id", 0),
                            f"⭐ {self.cfg['providers'][pid]['name']} — модель по умолчанию:", kb)
            return

        if data == "defbk":
            await self.edit(chat_id, msg.get("message_id", 0), "⭐ Провайдер по умолчанию:", self.kb_providers("defpv", "noop"))
            return

        if data.startswith("defsel:"):
            _, pid, idx_s = data.split(":", 2)
            item = self.idx_to_model(pid, int(idx_s))
            if not item:
                await self.edit(chat_id, msg.get("message_id", 0), "⚠️ Список устарел, откройте /default заново.")
                return
            mid, _name = item
            self.cfg["default"] = {"provider": pid, "model": mid}
            self.save_config()
            await self.edit(
                chat_id,
                msg.get("message_id", 0),
                f"⭐ Модель по умолчанию:\n{self.cfg['providers'][pid]['name']} → {self.model_display(pid, mid)}",
            )
            return

        # --- /del ---
        if data.startswith("del:"):
            pid = data[4:]
            p = self.cfg["providers"].pop(pid, None)
            self.models_cache.pop(pid, None)
            if p:
                if self.cfg["default"].get("provider") == pid:
                    self.cfg["default"] = {"provider": "", "model": ""}
                for st in self.state.values():
                    if st.get("provider") == pid:
                        st["provider"] = ""
                        st["model"] = ""
                self.save_config()
                self.save_state()
                await self.edit(chat_id, msg.get("message_id", 0), f"🗑 Провайдер «{p['name']}» удалён.")
            return

        # --- пагинация (selpg/defpg) ---
        m = re.match(r"^(sel|defsel)pg:([^:]+):(\d+)$", data)
        if m:
            prefix, pid, page = m.group(1), m.group(2), int(m.group(3))
            pick = "sel" if prefix == "sel" else "defsel"
            back = "bk" if prefix == "sel" else "defbk"
            kb, err = await self.kb_models(pid, page, pick, back)
            if err:
                await self.edit(chat_id, msg.get("message_id", 0), f"⚠️ {err}")
                return
            try:
                await self.api(
                    "editMessageReplyMarkup",
                    chat_id=chat_id,
                    message_id=msg.get("message_id", 0),
                    reply_markup=kb,
                )
            except TgError:
                pass
            return

    # ---------- обработка апдейтов ----------

    async def handle_update(self, up: Dict[str, Any]) -> None:
        try:
            if "callback_query" in up:
                await self.on_callback(up["callback_query"])
            elif "message" in up:
                await self.on_message(up["message"])
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Ошибка обработки апдейта: {e}")

    # ---------- webhook-режим (Render и другие хостинги) ----------

    async def run_webhook(self, base_url: str, port: int) -> None:
        """Webhook-режим: HTTP-сервер принимает апдейты от Telegram.

        Нужен на хостингах типа Render (Web Service обязан слушать PORT и
        отвечать на health-check). Входящие сообщения бота — это HTTP-трафик,
        который поддерживает сервис активным и будит его после сна.
        """
        if not base_url.startswith(("http://", "https://")):
            sys.exit("❌ WEBHOOK_URL / RENDER_EXTERNAL_URL должен начинаться с http(s)://")
        timeout = aiohttp.ClientTimeout(total=None)
        hook_path = "hook-" + hashlib.sha256(self.cfg["bot_token"].encode()).hexdigest()[:16]
        secret = secrets.token_hex(24)
        me: Dict[str, Any] = {}

        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.session = session
            try:
                me = await self.api_with_retry(
                    "getMe", attempts=3, delay=3.0,
                    fatal_markers=("401", "unauthorized", "404", "not found"),
                )
            except TgError as e:
                sys.exit(f"❌ Telegram отклонил токен: {e}\n💡 {tg_hint(str(e))}")

            async def telegram_update(request: web.Request) -> web.Response:
                if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != secret:
                    return web.Response(status=403, text="forbidden")
                try:
                    up = await request.json()
                except Exception:
                    return web.Response(status=400, text="bad json")
                asyncio.create_task(self.handle_update(up))
                return web.Response(text="ok")

            async def health(request: web.Request) -> web.Response:
                return web.Response(
                    text=f"🤖 Бот работает: @{me.get('username', '?')}",
                    content_type="text/plain", charset="utf-8",
                )

            webapp = web.Application()
            webapp.router.add_get("/", health)
            webapp.router.add_get("/healthz", health)
            webapp.router.add_post("/" + hook_path, telegram_update)

            runner = web.AppRunner(webapp, access_log=None)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", port)
            await site.start()

            try:
                await self.api_with_retry(
                    "setWebhook", attempts=4, delay=6.0,
                    fatal_markers=("400", "bad request"),
                    http_timeout=30,
                    url=f"{base_url}/{hook_path}",
                    secret_token=secret,
                    allowed_updates=["message", "callback_query"],
                    max_connections=40,
                    drop_pending_updates=False,
                )
            except TgError as e:
                await runner.cleanup()
                sys.exit(f"❌ Не удалось зарегистрировать webhook: {e}\n💡 {tg_hint(str(e))}\n"
                         "   (если повторяется — Manual Deploy → Restart в дашборде Render)")

            print("=" * 60)
            print(f"✅ Бот @{me.get('username', '?')} запущен (webhook): https://t.me/{me.get('username', '?')}")
            print(f"   Webhook: {base_url}/{hook_path}")
            print(f"   Слушаю 0.0.0.0:{port} · Конфиг: {CONFIG_PATH}")
            if not self.cfg["providers"]:
                print("   Провайдеров пока нет — отправьте боту /add")
            print("=" * 60)

            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, stop.set)
                except NotImplementedError:
                    pass
            await stop.wait()

            # webhook НЕ удаляем: Telegram подержит апдейты, пока сервис рестартует
            print("⏹ Останавливаюсь…")
            await runner.cleanup()

    # ---------- главный цикл ----------

    async def run(self) -> None:
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.session = session

            try:
                me = await self.api("getMe")
            except TgError as e:
                sys.exit(f"❌ Telegram отклонил токен: {e}\n💡 {tg_hint(str(e))}")
            self.me = me
            username = me.get("username", "?")
            print("=" * 60)
            print(f"✅ Бот @{username} запущен: https://t.me/{username}")
            print(f"   Конфиг: {CONFIG_PATH}")
            if not self.cfg["providers"]:
                print("   Провайдеров пока нет — отправьте боту /add")
            print("   Остановить: Ctrl+C")
            print("=" * 60)

            try:
                await self.api("deleteWebhook", http_timeout=15)
            except TgError:
                pass

            offset = 0
            while True:
                try:
                    updates = await self.api(
                        "getUpdates",
                        http_timeout=45,
                        offset=offset,
                        timeout=25,
                        allowed_updates=["message", "callback_query"],
                    )
                    for up in updates or []:
                        offset = up["update_id"] + 1
                        asyncio.create_task(self.handle_update(up))
                except TgError as e:
                    desc = str(e)
                    if "409" in desc or "webhook" in desc.lower():
                        print("⚠️ Конфликт (webhook или второй экземпляр). Пробую снять webhook…")
                        try:
                            await self.api("deleteWebhook", http_timeout=15, drop_pending_updates=True)
                        except TgError:
                            pass
                        await asyncio.sleep(2)
                    elif "401" in desc or "Unauthorized" in desc:
                        print("❌ Неверный токен! Исправьте bot_token в config.json и перезапустите.")
                        await asyncio.sleep(10)
                    else:
                        print(f"⚠️ Telegram: {desc}")
                        await asyncio.sleep(3)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    print(f"⚠️ Сеть: {e} — переподключаюсь…")
                    await asyncio.sleep(3)


def main() -> None:
    # пути для данных могут быть переопределены (например, диск /data на Render)
    for p in (CONFIG_PATH, STATE_PATH):
        d = os.path.dirname(p)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)

    cfg = load_json(CONFIG_PATH, default_config())
    apply_env_overrides(cfg)
    sync_env_providers(cfg)
    first_run_setup(cfg)

    mode = os.environ.get("MODE", "").strip().lower()
    use_webhook = (mode == "webhook") or (mode != "polling" and bool(WEBHOOK_URL_ENV))

    app = App(cfg)
    try:
        if use_webhook:
            if not WEBHOOK_URL_ENV:
                sys.exit("❌ Режим webhook включён, но не задан RENDER_EXTERNAL_URL или WEBHOOK_URL")
            print(f"🌐 Режим: webhook ({WEBHOOK_URL_ENV})")
            asyncio.run(app.run_webhook(WEBHOOK_URL_ENV, PORT))
        else:
            print("🔌 Режим: long polling")
            asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\n👋 Бот остановлен.")


if __name__ == "__main__":
    main()
