#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TG AI Bot — Telegram-бот с подключаемыми AI-провайдерами (как в opencode).

Умеет:
    • стриминг ответов с анимацией «⌛ Думаю · 7с» и живыми статусами
      (Ищу в интернете / Читаю страницу), ответы оформляются markdown → HTML;
    • ВСЕ КОМАНДЫ — кнопками reply-клавиатуры под полем ввода,
      разбитые по разделам (чат / провайдеры / настройки);
    • ШРИФТ ОТВЕТОВ ИИ — переключается кнопкой 🔤 Шрифт или /font:
      обычный (markdown), моноширинный, всё жирным, всё курсивом, простой;
    • ПРИНИМАЕТ ФОТО (vision) и файлы (код, txt, csv, json…) — модель видит их
      содержимое; история хранит маркеры вместо тяжёлых медиа;
    • веб-инструменты web_search (DuckDuckGo) и read_url с автоматическим
      tool-calling во всех трёх протоколах (до 4 раундов); провайдеры без
      поддержки tools отключаются автоматически; /web — вкл/выкл.

Локально (long polling, сервер не нужен):
    pip install aiohttp
    BOT_TOKEN=123:ABC python main.py      # или просто python main.py — спросит токен

На Render (Web Service, всё через переменные окружения — config.json не нужен):
    BOT_TOKEN            токен от @BotFather (обязательно)
    ADMIN_IDS            ID админов через запятую
    PROVIDER_1..N        провайдеры, применяются при каждом старте:
                           PROVIDER_1=openrouter:sk-or-v1-…        (пресет:ключ)
                           PROVIDER_2=DeepSeek|openai|https://api.deepseek.com/v1|sk-…
                           PROVIDER_3=My|auto|https://api.example.com/v1|sk-…  (тип API определится сам)
                           (пресеты: openrouter openai anthropic google gemini zai
                            groq deepseek mistral xai moonshot together fireworks cerebras)
    CUSTOM_URL / CUSTOM_KEY / CUSTOM_NAME / CUSTOM_PROTOCOL / CUSTOM_MODEL
                         свой провайдер с произвольным URL — без пресетов,
                         ни от кого не зависит (id: ec); CUSTOM_KEY пустой = сервер
                         без ключа (Ollama/LM Studio/vLLM); CUSTOM_PROTOCOL=auto —
                         определить по URL
    DEFAULT_MODEL        модель по умолчанию, напр. deepseek-chat или openai/gpt-4o-mini
    DEFAULT_PROVIDER     имя провайдера для дефолта (иначе первый PROVIDER_N)
    SYSTEM_PROMPT / MAX_HISTORY / ADMIN_ONLY — применяются при каждом старте
    WEB_TOOLS            0 = выключить веб-поиск по умолчанию (вкл: 1)
    FONT                 шрифт ответов по умолчанию: rich | mono | bold | italic | plain
                         (каждый чат может выбрать свой — кнопкой 🔤 Шрифт)
    Render сам задает PORT и RENDER_EXTERNAL_URL — бот поднимет HTTP-сервер
    (health-check на "/") и зарегистрирует webhook.

Дальше всё настраивается и прямо в Telegram:
    /add       — добавить провайдера: «🔗 Своя ссылка (авто)» — просто URL,
                 бот сам определит тип API и сам загрузит список доступных
                 моделей по вашему ключу (ключ можно пропустить — для
                 Ollama/LM Studio/vLLM); либо пресеты OpenRouter/OpenAI/…
    /model     — выбрать модель для чата (кнопка 🔄 перезагружает список
                 моделей с сервера; кнопка «🤖 Сменить модель» появляется
                 под каждым ответом — можно менять модель каждый раз)
    /providers — список провайдеров
    /web       — веб-инструменты: поиск и чтение страниц
    /font      — меню шрифтов ИИ: обычный, моно, жирный, курсив, простой
    /get_id    — custom_emoji_id стикеров набора (свои эмодзи-статусы)
    /help      — все команды (продублированы кнопками под полем ввода)

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
import base64
import hashlib
import html
import json
import os
import re
import secrets
import signal
import sys
import time
from datetime import date
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from urllib.parse import unquote

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

# --- медиа и инструменты ---
MAX_MEDIA_BYTES = 10 * 1024 * 1024   # максимальный размер фото/файла
MAX_FILE_TEXT = 15000                # максимальный размер текста из файла
MAX_TOOL_ROUNDS = 4                  # раундов «модель → инструменты → модель»
TG_FILE_BASE = "https://api.telegram.org/file/bot{token}/{path}"  # скачивание файлов
UA = {"User-Agent": "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120 Mobile Safari/537.36"}
TOOL_LABELS = {"web_search": "🌐 Ищу в интернете", "read_url": "📄 Читаю страницу"}

# --- шрифты, которыми пишет ИИ (меню /font) ---
FONTS = {
    "rich": "🖋 Обычный (markdown)",
    "mono": "💻 Моноширинный",
    "bold": "🅱️ Всё жирным",
    "italic": "✏️ Всё курсивом",
    "plain": "📄 Простой текст",
}
FONT_ORDER = ["rich", "mono", "bold", "italic", "plain"]

# --- reply-клавиатура: все команды кнопками, по разделам ---
MENU_BTN = {
    "🤖 Модель": "/model",
    "📊 Статус": "/status",
    "🔤 Шрифт": "/font",
    "🌐 Поиск": "/web",
    "➕ Провайдер": "/add",
    "🧩 Провайдеры": "/providers",
    "⭐ Дефолт": "/default",
    "🗑 Удалить": "/del",
    "💬 Промпт": "/sys",
    "🔒 Только админы": "/lock",
    "🧹 Очистить": "/clear",
    "📋 Помощь": "/help",
    "🆔 Мой ID": "/id",
    "✖️ Отмена": "/cancel",
}
# Раздел «Чат и оформление» — видно всем:
KB_USER_ROWS = [
    ["🤖 Модель", "📊 Статус"],
    ["🔤 Шрифт", "🧹 Очистить"],
    ["📋 Помощь", "🆔 Мой ID"],
]
# Для админов добавляются разделы «Веб», «Провайдеры», «Настройки»:
KB_ADMIN_ROWS = [
    ["🤖 Модель", "📊 Статус"],
    ["🔤 Шрифт", "🌐 Поиск"],
    ["➕ Провайдер", "🧩 Провайдеры"],
    ["⭐ Дефолт", "🗑 Удалить"],
    ["💬 Промпт", "🔒 Только админы"],
    ["🧹 Очистить", "📋 Помощь"],
    ["🆔 Мой ID", "✖️ Отмена"],
]

TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".log", ".json", ".xml", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".env", ".py", ".pyw", ".js", ".mjs", ".ts", ".tsx",
    ".jsx", ".java", ".kt", ".kts", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs",
    ".rb", ".php", ".lua", ".swift", ".m", ".sql", ".sh", ".bash", ".zsh", ".ps1",
    ".html", ".htm", ".css", ".scss", ".vue", ".svelte", ".dart", ".r", ".ipynb",
    ".dockerfile", ".gitignore", ".diff", ".patch", ".tex", ".srt", ".vtt",
}
IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# инструменты (формат OpenAI; для других протоколов конвертируются)
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Поиск в интернете (DuckDuckGo). Возвращает заголовки, ссылки и сниппеты.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Поисковый запрос"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": "Открыть веб-страницу и получить её текст.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]

PROTOCOLS = ("openai", "anthropic", "gemini")
PROTOCOL_TITLES = {
    "openai": "OpenAI-compatible",
    "anthropic": "Anthropic-compatible",
    "gemini": "Google Gemini",
}

PRESETS: Dict[str, Dict[str, str]] = {
    "custom-auto": {
        "name": "🔗 Своя ссылка (авто)",
        "protocol": "",  # определится автоматически по URL
        "base_url": "",
        "docs": "",
        "hint": "Просто URL — бот сам определит тип API",
    },
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
    "💬 Просто напишите сообщение — отвечу через выбранную модель.",
    "📷 Можно присылать фото (модель их увидит) и файлы: код, txt, csv, json…",
    "🌐 Умею искать в интернете и открывать ссылки, если нужны свежие данные.",
    "⌨️ Все команды продублированы кнопками под полем ввода, по разделам.",
    "",
    "— Чат и модель —",
    "/model — выбрать модель для этого чата",
    "/status — какая модель используется",
    "/clear — очистить историю диалога",
    "/id — показать ID (чата и ваш)",
    "",
    "— Оформление —",
    "/font — шрифт ответов ИИ: 🖋 обычный / 💻 моно / 🅱 жирный / ✏️ курсив / 📄 простой",
    "",
    "— Провайдеры (админ) —",
    "/providers — список подключённых провайдеров",
    "/add — подключить провайдера: своя ссылка (авто) или пресет",
    "/del — удалить провайдера",
    "/default — модель по умолчанию для новых чатов",
    "/key <id> <ключ> — сменить API-ключ провайдера",
    "/url <id> <url> — сменить baseURL провайдера",
    "/web — вкл/выкл веб-инструменты (поиск и чтение страниц)",
    "",
    "— Настройки (админ) —",
    "/sys <текст> — системный промпт (/sys off — выключить)",
    "/lock — переключить режим «только администраторы»",
    "",
    "— Прочее —",
    "/get_id <набор> — custom_emoji_id стикеров набора",
    "/cancel — отменить текущий мастер",
])


class ProviderError(RuntimeError):
    """Ошибка на стороне провайдера (показываем пользователю как есть)."""


def detect_protocol(url: str) -> str:
    """Определить протокол API по ссылке — как в opencode, где обычно
    достаточно одного URL. Anthropic и Google узнаются по домену,
    всё остальное (OpenAI, OpenRouter, vLLM, Ollama, прокси) —
    OpenAI-compatible, это де-факто стандарт."""
    u = (url or "").lower()
    if "anthropic" in u:
        return "anthropic"
    if "generativelanguage" in u or "gemini" in u or "aiplatform.googleapis" in u:
        return "gemini"
    return "openai"


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
        "web_tools": True,       # веб-инструменты: поиск и чтение страниц
        "font": "rich",          # вид ответов: rich | mono | plain
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
    wt = os.environ.get("WEB_TOOLS", "").strip().lower()
    if wt in ("1", "true", "yes"):
        cfg["web_tools"] = True
        applied.append("WEB_TOOLS=1")
    elif wt in ("0", "false", "no"):
        cfg["web_tools"] = False
        applied.append("WEB_TOOLS=0")
    ft = os.environ.get("FONT", "").strip().lower()
    if ft in FONTS:
        cfg["font"] = ft
        applied.append(f"FONT={ft}")
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
    # форма 2: Имя|протокол|URL|ключ (протокол "auto" — определить по URL)
    parts = raw.split("|")
    if len(parts) >= 3:
        name, proto, url = parts[0].strip(), parts[1].strip().lower(), parts[2].strip()
        key = parts[3].strip() if len(parts) > 3 else ""
        if proto == "google":
            proto = "gemini"
        if proto == "auto":
            proto = detect_protocol(url)
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
            if proto == "auto":
                proto = detect_protocol(c_url)
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
        headers = {"anthropic-version": "2023-06-01"}
        if key:
            headers["x-api-key"] = key
        async with session.get(f"{base}/v1/models?limit=1000", headers=headers) as r:
            data = await _json_or_error(r)
        out = []
        for m in data.get("data", []):
            mid = m.get("id") or ""
            if mid:
                out.append((mid, m.get("display_name") or m.get("name") or mid))
        return out

    # openai-compatible; если пользователь дал ссылку без /v1 — пробуем и /v1/models
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    candidates = [f"{base}/models"]
    if not re.search(r"/v\d+$", base):
        candidates.append(f"{base}/v1/models")
    data: Dict[str, Any] = {}
    for u in candidates:
        async with session.get(u, headers=headers) as r:
            if r.status == 404 and u != candidates[-1]:
                continue
            data = await _json_or_error(r)
            break
    items = data.get("data") or data.get("models") or []
    out = []
    for m in items:
        mid = m.get("id") or m.get("name")
        if mid:
            out.append((str(mid), str(mid)))
    return out


# --- конвертация контента и инструментов под протоколы ---

def _tools_for(proto: str) -> List[Dict[str, Any]]:
    if proto == "anthropic":
        return [{"name": t["function"]["name"],
                 "description": t["function"]["description"],
                 "input_schema": t["function"]["parameters"]} for t in TOOLS_SPEC]
    if proto == "gemini":
        return [{"functionDeclarations": [
            {"name": t["function"]["name"],
             "description": t["function"]["description"],
             "parameters": t["function"]["parameters"]} for t in TOOLS_SPEC]}]
    return TOOLS_SPEC


def _safe_json(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s or "{}")
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _openai_content(c: Any) -> Any:
    if isinstance(c, str):
        return c
    out: List[Dict[str, Any]] = []
    for p in c:
        if p.get("type") == "text":
            out.append({"type": "text", "text": p.get("text", "")})
        elif p.get("type") == "image":
            out.append({"type": "image_url",
                        "image_url": {"url": f"data:{p.get('media_type', 'image/jpeg')};base64,{p.get('data', '')}"}})
    return out


def _anth_content(c: Any) -> List[Dict[str, Any]]:
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    out: List[Dict[str, Any]] = []
    for p in c:
        if p.get("type") == "text":
            out.append({"type": "text", "text": p.get("text", "")})
        elif p.get("type") == "image":
            out.append({"type": "image", "source": {"type": "base64",
                                                    "media_type": p.get("media_type", "image/jpeg"),
                                                    "data": p.get("data", "")}})
    return out


def _gem_parts(c: Any) -> List[Dict[str, Any]]:
    if isinstance(c, str):
        return [{"text": c}]
    out: List[Dict[str, Any]] = []
    for p in c:
        if p.get("type") == "text":
            out.append({"text": p.get("text", "")})
        elif p.get("type") == "image":
            out.append({"inline_data": {"mime_type": p.get("media_type", "image/jpeg"),
                                        "data": p.get("data", "")}})
    return out


async def stream_chat(
    session: "aiohttp.ClientSession",
    provider: Dict[str, Any],
    model: str,
    messages: List[Dict[str, Any]],
    tools: bool = False,
) -> AsyncGenerator[Tuple[str, Any], None]:
    """Стриминг ответа модели. Отдаёт события-кортежи:
       ("text", "кусочек") и, если модель вызвала инструменты,
       ("tools", [{"id", "name", "args"}]) в конце раунда."""
    proto = provider["protocol"]
    base = provider["base_url"].rstrip("/")
    key = provider.get("api_key", "")
    system = "\n\n".join(m["content"] for m in messages
                         if m["role"] == "system" and isinstance(m.get("content"), str))
    rest = [m for m in messages if m["role"] != "system"]
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=300)

    if proto == "openai":
        # без /v1 в конце — пробуем и /v1/chat/completions (кастомные ссылки)
        urls = [f"{base}/chat/completions"]
        if not re.search(r"/v\d+$", base):
            urls.append(f"{base}/v1/chat/completions")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if "openrouter.ai" in base:
            headers["HTTP-Referer"] = "https://github.com/tg-ai-bot"
            headers["X-Title"] = "TG AI Bot"
        msgs: List[Dict[str, Any]] = []
        for m in rest:
            if m["role"] == "assistant" and m.get("tool_calls"):
                msgs.append({
                    "role": "assistant",
                    "content": m.get("content") or None,
                    "tool_calls": [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                                   for c in m["tool_calls"]],
                })
            elif m["role"] == "tool":
                msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                msgs.append({"role": m["role"], "content": _openai_content(m["content"])})
        payload: Dict[str, Any] = {"model": model, "messages": msgs, "stream": True}
        if tools:
            payload["tools"] = _tools_for(proto)
    elif proto == "anthropic":
        url = f"{base}/v1/messages"
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if key:
            headers["x-api-key"] = key
        a_msgs: List[Dict[str, Any]] = []
        pend: List[Dict[str, Any]] = []

        def flush_tools() -> None:
            if pend:
                a_msgs.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": t["tool_call_id"], "content": t["content"]}
                    for t in pend]})
                pend.clear()

        for m in rest:
            if m["role"] == "assistant" and m.get("tool_calls"):
                flush_tools()
                blocks: List[Dict[str, Any]] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for c in m["tool_calls"]:
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"],
                                   "input": _safe_json(c["args"])})
                a_msgs.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                pend.append(m)
            else:
                flush_tools()
                a_msgs.append({"role": "user", "content": _anth_content(m["content"])})
        flush_tools()
        payload = {"model": model, "max_tokens": MAX_TOKENS_ANTHROPIC,
                   "messages": a_msgs, "stream": True}
        if tools:
            payload["tools"] = _tools_for(proto)
    elif proto == "gemini":
        url = f"{base}/v1beta/models/{model}:streamGenerateContent?alt=sse"
        if key:
            url += f"&key={key}"
        headers = {"Content-Type": "application/json"}
        g_msgs: List[Dict[str, Any]] = []
        pend_g: List[Dict[str, Any]] = []

        def flush_tools_g() -> None:
            if pend_g:
                g_msgs.append({"role": "function", "parts": [
                    {"functionResponse": {"name": t["name"], "response": {"result": t["content"]}}}
                    for t in pend_g]})
                pend_g.clear()

        for m in rest:
            if m["role"] == "assistant" and m.get("tool_calls"):
                flush_tools_g()
                parts: List[Dict[str, Any]] = []
                if m.get("content"):
                    parts.append({"text": m["content"]})
                for c in m["tool_calls"]:
                    parts.append({"functionCall": {"name": c["name"], "args": _safe_json(c["args"])}})
                g_msgs.append({"role": "model", "parts": parts})
            elif m["role"] == "tool":
                pend_g.append(m)
            else:
                flush_tools_g()
                g_msgs.append({"role": "user", "parts": _gem_parts(m["content"])})
        flush_tools_g()
        payload = {"contents": g_msgs}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = _tools_for(proto)
    else:
        raise ProviderError(f"Неизвестный протокол: {proto}")

    resp: Optional[aiohttp.ClientResponse] = None
    oa_calls: Dict[int, Dict[str, str]] = {}
    anth_calls: List[Dict[str, str]] = []
    anth_cur: Optional[Dict[str, str]] = None
    gem_calls: List[Dict[str, str]] = []
    try:
        if proto == "openai":
            for i, u in enumerate(urls):
                r = await session.post(u, json=payload, headers=headers, timeout=timeout)
                if r.status == 404 and i < len(urls) - 1:
                    r.close()
                    continue
                resp = r
                break
        else:
            resp = await session.post(url, json=payload, headers=headers, timeout=timeout)

        async with resp:
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
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        yield ("text", delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index")
                        idx = len(oa_calls) if idx is None else int(idx)
                        c = oa_calls.setdefault(idx, {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            c["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            c["name"] += fn["name"]
                        if fn.get("arguments"):
                            c["args"] += fn["arguments"]
                elif proto == "anthropic":
                    t = obj.get("type")
                    if t == "content_block_start":
                        b = obj.get("content_block") or {}
                        if b.get("type") == "tool_use":
                            anth_cur = {"id": b.get("id", ""), "name": b.get("name", ""), "args": ""}
                    elif t == "content_block_delta":
                        d = obj.get("delta") or {}
                        if d.get("type") == "input_json_delta" and anth_cur is not None:
                            anth_cur["args"] += d.get("partial_json") or ""
                        elif d.get("type") in (None, "text_delta") and d.get("text"):
                            yield ("text", d["text"])
                    elif t == "content_block_stop":
                        if anth_cur is not None:
                            anth_calls.append(anth_cur)
                            anth_cur = None
                    elif t == "error":
                        raise ProviderError(json.dumps(obj.get("error"), ensure_ascii=False)[:300])
                else:  # gemini
                    for cand in obj.get("candidates") or []:
                        for part in ((cand.get("content") or {}).get("parts") or []):
                            if part.get("text"):
                                yield ("text", part["text"])
                            if part.get("functionCall"):
                                fc = part["functionCall"]
                                gem_calls.append({"id": f"call{len(gem_calls)}",
                                                  "name": fc.get("name", ""),
                                                  "args": json.dumps(fc.get("args") or {}, ensure_ascii=False)})

        if proto == "openai" and oa_calls:
            yield ("tools", [{"id": c["id"] or f"call{i}", "name": c["name"], "args": c["args"] or "{}"}
                             for i, c in sorted(oa_calls.items())])
        elif proto == "anthropic" and anth_calls:
            yield ("tools", anth_calls)
        elif proto == "gemini" and gem_calls:
            yield ("tools", gem_calls)
    finally:
        if resp is not None and not resp.closed:
            resp.close()


# --- веб-инструменты (web_search / read_url) ---

def _clean_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


async def web_search(session: "aiohttp.ClientSession", query: str, limit: int = 6) -> str:
    try:
        async with session.get("https://html.duckduckgo.com/html/",
                               params={"q": query, "kl": "ru-ru"},
                               headers=UA,
                               timeout=aiohttp.ClientTimeout(total=15)) as r:
            page = await r.text(errors="ignore")
    except Exception as e:  # noqa: BLE001
        return f"Ошибка поиска: {e}"
    links = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.S)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)
    out = []
    for i, (href, title) in enumerate(links[:limit]):
        m = re.search(r"uddg=([^&]+)", href)
        url = unquote(m.group(1)) if m else href
        snip = _clean_html(snippets[i]) if i < len(snippets) else ""
        out.append(f"{i + 1}. {_clean_html(title)}\n{url}\n{snip}")
    return "\n\n".join(out) or "Ничего не найдено."


async def read_url(session: "aiohttp.ClientSession", url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "Неверная ссылка."
    try:
        async with session.get(url, headers=UA,
                               timeout=aiohttp.ClientTimeout(total=15)) as r:
            page = await r.text(errors="ignore")
    except Exception as e:  # noqa: BLE001
        return f"Не удалось открыть: {e}"
    page = re.sub(r"(?is)<(script|style|noscript|svg|header|footer|nav).*?</\1>", " ", page)
    text = _clean_html(re.sub(r"(?s)<[^>]+>", " ", page))
    text = re.sub(r"\s+", " ", text)
    return text[:4000] or "Страница пустая."


async def run_tool(session: Optional["aiohttp.ClientSession"], name: str, args_json: str) -> str:
    args = _safe_json(args_json)
    if name == "web_search":
        return await web_search(session, args.get("query", ""))
    if name == "read_url":
        return await read_url(session, args.get("url", ""))
    return "Неизвестный инструмент."


def _is_tools_error(err: str) -> bool:
    """Похоже ли на ошибку «провайдер не умеет tools/functions»."""
    low = (err or "").lower()
    mentions = any(m in low for m in ("tools", "tool_use", "function", "functioncall"))
    broken = any(m in low for m in ("not support", "unsupported", "unknown", "unrecognized",
                                    "invalid", "unexpected", "400"))
    return mentions and broken


# --- оформление ответов ---

def md_to_html(text: str) -> str:
    """Markdown → Telegram HTML: **жирный**, *курсив*, `код`, ```блоки```, # заголовки."""
    text = html.escape(text, quote=False)
    text = re.sub(r"```(\w*)\n?(.*?)```",
                  lambda m: f'<pre><code class="language-{m.group(1)}">{m.group(2)}</code></pre>'
                  if m.group(1) else f"<pre><code>{m.group(2)}</code></pre>",
                  text, flags=re.DOTALL)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"^#{1,4}\s+(.*)$", r"<b>\1</b>", text, flags=re.MULTILINE)
    return text


def fmt_strict(text: str) -> str:
    """Строгий моноширинный вид: без markdown, весь ответ <code>…</code>."""
    text = re.sub(r"```\w*\n?", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
    return f"<code>{html.escape(text.strip(), quote=False)}</code>"


# --- анимация «Думаю · Nс» (как черновик из референса, но через editMessageText) ---

class ThinkingTimer:
    """Обновляет сообщение-статус раз в секунду: «⌛ Провайдер · Думаю\n7с».
    Метки меняются во время работы инструментов (Ищу в интернете / Читаю страницу)."""

    def __init__(self, app: "App", chat_id: int, message_id: int, provider_name: str) -> None:
        self.app = app
        self.chat_id = chat_id
        self.message_id = message_id
        self.provider = provider_name
        self.label = "Думаю"
        self.start = time.monotonic()
        self.active = False
        self.task: Optional[asyncio.Task] = None

    async def _loop(self) -> None:
        while self.active:
            sec = int(time.monotonic() - self.start)
            try:
                await self.app.edit(self.chat_id, self.message_id,
                                    f"⌛ {self.provider} · {self.label}\n{sec}с")
            except Exception:  # noqa: BLE001
                pass
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                raise

    def run(self) -> None:
        if not self.active:
            self.active = True
            self.start = time.monotonic()
            self.task = asyncio.create_task(self._loop())

    def set_label(self, label: str) -> None:
        self.label = label
        self.start = time.monotonic()

    async def stop(self) -> None:
        if self.active:
            self.active = False
            if self.task is not None:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
                self.task = None


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
        # провайдеры, у которых веб-инструменты выключены (не поддерживают tools)
        self._no_tools: set = set()
        # чаты, которым уже отправлена reply-клавиатура (не спамим повторами)
        self._kb_chats: set = set()
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

    async def send(self, chat_id: int, text: str, reply_markup: Optional[Dict] = None,
                   parse_mode: Optional[str] = None, attach_kb: bool = False) -> Dict[str, Any]:
        """sendMessage. attach_kb=True — приложить reply-клавиатуру-меню
        (один раз на чат; в группах не прикладываем, там chat_id < 0)."""
        kwargs: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text[:MAX_TG],
            "disable_web_page_preview": True,
        }
        if reply_markup:
            kwargs["reply_markup"] = reply_markup
        elif attach_kb and chat_id > 0 and chat_id not in self._kb_chats:
            self._kb_chats.add(chat_id)
            kwargs["reply_markup"] = self.reply_kb(chat_id)  # в личке chat_id == uid
        if parse_mode:
            kwargs["parse_mode"] = parse_mode
        return await self.api("sendMessage", **kwargs)

    async def edit(self, chat_id: int, message_id: int, text: str, reply_markup: Optional[Dict] = None,
                   parse_mode: Optional[str] = None) -> None:
        kwargs: Dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text[:MAX_TG],
            "disable_web_page_preview": True,
        }
        if reply_markup is not None:
            kwargs["reply_markup"] = reply_markup
        if parse_mode:
            kwargs["parse_mode"] = parse_mode
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

    # ---------- фото и файлы ----------

    async def download_tg_file(self, file_id: str, max_bytes: int = MAX_MEDIA_BYTES) -> Tuple[bytes, str]:
        """Скачать файл с серверов Telegram: (bytes, mime по расширению)."""
        f = await self.api("getFile", http_timeout=30, file_id=file_id)
        path = (f or {}).get("file_path") or ""
        ext = ("." + path.rsplit(".", 1)[-1].lower()) if "." in path else ""
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                    "webp": "image/webp", "gif": "image/gif", "pdf": "application/pdf",
                    "mp3": "audio/mpeg", "ogg": "audio/ogg", "mp4": "video/mp4"}
        mime = mime_map.get(ext.lstrip("."), "application/octet-stream")
        url = TG_FILE_BASE.format(token=self.cfg["bot_token"], path=path)
        assert self.session is not None
        async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
            if r.status >= 400:
                raise ProviderError(f"не удалось скачать файл: HTTP {r.status}")
            data = await r.read()
        if len(data) > max_bytes:
            raise ProviderError(f"файл слишком большой ({len(data) // 1024 // 1024} МБ, лимит "
                                f"{max_bytes // 1024 // 1024} МБ)")
        return data, mime

    async def extract_media(self, msg: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str, List[str]]:
        """Фото и документы из сообщения → (картинки для vision, текст файлов, примечания).
        Картинки: [{"type": "image", "media_type", "data"(base64)}]."""
        images: List[Dict[str, str]] = []
        files_text = ""
        notes: List[str] = []

        async def add_image(file_id: str, mime: str) -> None:
            try:
                data, m = await self.download_tg_file(file_id)
                mime = mime if mime.startswith("image/") else m
                images.append({"type": "image", "media_type": mime or m,
                               "data": base64.b64encode(data).decode()})
            except ProviderError as e:
                notes.append(str(e))
            except Exception as e:  # noqa: BLE001
                notes.append(f"не удалось скачать: {e}")

        photo = msg.get("photo")
        if photo:
            best = max(photo, key=lambda p: p.get("width", 0) * p.get("height", 0))
            await add_image(best.get("file_id", ""), "image/jpeg")

        doc = msg.get("document")
        if doc:
            name = doc.get("file_name") or "file"
            mime = (doc.get("mime_type") or "").lower()
            ext = os.path.splitext(name)[1].lower()
            if mime.startswith("image/") or ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
                await add_image(doc.get("file_id", ""), mime)
            elif mime.startswith("text/") or mime in (
                    "application/json", "application/xml", "application/javascript",
                    "application/x-yaml", "application/toml", "application/x-sh") or ext in TEXT_EXTS:
                try:
                    data, _ = await self.download_tg_file(doc.get("file_id", ""))
                except ProviderError as e:
                    notes.append(str(e))
                else:
                    txt = data.decode("utf-8", "replace")
                    if txt.count("\ufffd") > len(txt) * 0.3:
                        notes.append(f"файл «{name}» двоичный — пришлите текстом")
                    else:
                        cut = " …(обрезано)" if len(txt) > MAX_FILE_TEXT else ""
                        files_text += f"📎 Файл «{name}»:\n\n{txt[:MAX_FILE_TEXT]}{cut}\n\n"
            else:
                notes.append(f"файл «{name}» этого типа пока не поддерживается — "
                             "пришлите текстом или скриншотом")
        return images, files_text, notes

    # ---------- клавиатуры ----------

    @staticmethod
    def kb(rows: List[List[Tuple[str, str]]]) -> Dict[str, Any]:
        return {
            "inline_keyboard": [
                [{"text": t, "callback_data": d} for (t, d) in row]
                for row in rows
            ]
        }

    def reply_kb(self, uid: int) -> Dict[str, Any]:
        """Постоянная reply-клавиатура: все команды кнопками, по разделам.
        Не-админы видят только общие разделы (чат / оформление)."""
        rows = KB_ADMIN_ROWS if self.is_admin(uid) else KB_USER_ROWS
        return {
            "keyboard": [[{"text": t} for t in row] for row in rows],
            "resize_keyboard": True,
            "is_persistent": True,
            "input_field_placeholder": "Сообщение, фото или файл для ИИ…",
        }

    def kb_font(self, cur: str) -> Dict[str, Any]:
        """Меню выбора шрифта, которым пишет ИИ (✅ — текущий)."""
        rows: List[List[Tuple[str, str]]] = []
        for i in range(0, len(FONT_ORDER), 2):
            row: List[Tuple[str, str]] = []
            for f in FONT_ORDER[i:i + 2]:
                title = ("✅ " + FONTS[f]) if f == cur else FONTS[f]
                row.append((title, f"font:{f}"))
            rows.append(row)
        return self.kb(rows)

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
        rows.append([("🔄 Обновить список моделей", f"rm:{pick_prefix}:{pid}")])
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

    async def send_final(self, chat_id: int, msg_id: int, text: str) -> None:
        """Финальный ответ: оформление по шрифту чата (+ кнопка смены модели).
        Длинные ответы разбиваются (без форматирования)."""
        quick = self.kb([[("🤖 Сменить модель", "qmodel")]])
        st = self.chat_state(chat_id)
        font = st.get("font") or self.cfg.get("font", "rich")
        html_text: Optional[str] = None
        if font == "mono":
            html_text = fmt_strict(text)
        elif font == "bold":
            html_text = "<b>" + md_to_html(text) + "</b>"
        elif font == "italic":
            html_text = "<i>" + md_to_html(text) + "</i>"
        elif font == "rich":
            html_text = md_to_html(text)
        if html_text is not None and len(html_text) <= MAX_TG:
            try:
                await self.edit(chat_id, msg_id, html_text, quick, parse_mode="HTML")
                return
            except TgError:
                pass  # не разобралась разметка → обычный текст ниже
        if len(text) <= MAX_TG:
            await self.edit(chat_id, msg_id, text, quick)
        else:
            chunks = []
            rest = text
            while rest:
                chunks.append(rest[:MAX_TG])
                rest = rest[MAX_TG:]
            await self.edit(chat_id, msg_id, chunks[0], quick)
            for c in chunks[1:]:
                await self.send(chat_id, c)

    async def generate_and_send(self, msg: Dict[str, Any], user_text: str,
                                images: Optional[List[Dict[str, str]]] = None,
                                files_text: str = "", marker: Optional[str] = None) -> None:
        chat_id = msg["chat"]["id"]
        resolved = self.resolve(chat_id)
        if not resolved:
            await self.send(
                chat_id,
                "⚠️ Модель не выбрана.\n\nИспользуйте /model — выбрать модель\n"
                "Или /add (если вы админ) — подключить провайдера",
                attach_kb=True,
            )
            return
        pid, provider, model_id = resolved
        display = self.model_display(pid, model_id)
        images = images or []

        st = self.chat_state(chat_id)
        history: List[Dict[str, Any]] = st.get("history", [])
        max_hist = int(self.cfg.get("max_history", 24) or 24)

        tools_on = bool(self.cfg.get("web_tools", True)) and pid not in self._no_tools

        def build_messages(with_tools: bool) -> List[Dict[str, Any]]:
            sp = self.cfg.get("system_prompt", "")
            if with_tools:
                sp = (sp + "\n\n" if sp else "") + (
                    "Если нужны свежие данные (новости, курсы, погода, события, факты, "
                    "в которых не уверен) — используй web_search, а при необходимости "
                    "read_url, чтобы открыть страницу. В конце коротко укажи источники ссылками.")
            date_note = f"Сегодня {date.today().strftime('%d.%m.%Y')}."
            sp = (sp + "\n\n" + date_note) if sp else date_note
            content_text = ((files_text + "\n\n") if files_text else "") + user_text
            user_content: Any = (
                [{"type": "text", "text": content_text}] + images if images else content_text)
            return [{"role": "system", "content": sp}] + history + \
                [{"role": "user", "content": user_content}]

        messages = build_messages(tools_on)

        placeholder = await self.send(chat_id, f"⌛ {provider['name']} · {display}\nДумаю\n0с")
        msg_id = placeholder["message_id"]
        timer = ThinkingTimer(self, chat_id, msg_id, provider["name"])

        stop = asyncio.Event()
        typing_task = asyncio.create_task(self.typing_loop(chat_id, stop))
        full = ""
        try:
            timer.run()
            for attempt in (0, 1):
                conv = list(messages)
                full = ""
                last_edit = 0.0
                rounds = MAX_TOOL_ROUNDS if tools_on else 1
                try:
                    for _round in range(rounds):
                        acc = ""
                        calls: List[Dict[str, str]] = []
                        async for kind, piece in stream_chat(self.session, provider, model_id, conv, tools=tools_on):  # type: ignore[arg-type]
                            if kind == "tools":
                                calls = piece
                                continue
                            acc += piece
                            if timer.active:
                                await timer.stop()
                            now = time.monotonic()
                            if now - last_edit >= EDIT_EVERY:
                                last_edit = now
                                try:
                                    await self.edit(chat_id, msg_id, acc[-MAX_TG:] + " ▌")
                                except TgError:
                                    pass
                        if not calls:
                            full = acc
                            break
                        # модель вызвала инструменты — выполняем и идём на следующий раунд
                        conv.append({"role": "assistant", "content": acc, "tool_calls": calls})
                        for c in calls:
                            timer.set_label(TOOL_LABELS.get(c["name"], "Работаю"))
                            timer.run()
                            result = await run_tool(self.session, c["name"], c["args"])
                            conv.append({"role": "tool", "tool_call_id": c["id"],
                                         "name": c["name"], "content": result})
                        timer.set_label("Думаю")
                        full = acc
                    break
                except ProviderError as e:
                    if attempt == 0 and tools_on and _is_tools_error(str(e)):
                        # провайдер не умеет инструменты — запоминаем и повторяем без них
                        self._no_tools.add(pid)
                        tools_on = False
                        try:
                            await self.edit(
                                chat_id, msg_id,
                                f"⚠️ {provider['name']} не поддерживает веб-инструменты.\n"
                                "Продолжаю без поиска… (выключить совсем: /web off)")
                        except TgError:
                            pass
                        await asyncio.sleep(1)
                        continue
                    raise
            await timer.stop()

            if not full:
                await self.edit(chat_id, msg_id, "⚠️ Пустой ответ от модели.")
                return

            # в историю — маркер вместо тяжёлых медиа
            hist_user = marker or user_text
            history = history + [
                {"role": "user", "content": hist_user},
                {"role": "assistant", "content": full},
            ]
            st["history"] = history[-max_hist * 2:]
            self.save_state()

            await self.send_final(chat_id, msg_id, full)
        except ProviderError as e:
            err = f"⚠️ Ошибка провайдера ({provider['name']}):\n{e}"
            if images and any(w in str(e).lower() for w in ("image", "vision", "медиа", "мультимод", "input type")):
                err += "\n\n💡 Похоже, эта модель не принимает изображения — выберите другую: /model"
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
            await timer.stop()
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
        text = (msg.get("text") or msg.get("caption") or "").strip()
        # кнопки reply-клавиатуры присылают свой текст — превращаем в команду
        if text in MENU_BTN:
            text = MENU_BTN[text]
        has_media = bool(msg.get("photo") or msg.get("document"))

        # первый пользователь в личке становится владельцем
        if not self.cfg.get("admins") and msg["chat"]["type"] == "private" and uid:
            self.cfg["admins"] = [uid]
            self.save_config()
            await self.send(
                chat_id,
                "✅ Вы назначены администратором бота.\n\nНачните с /add — подключите провайдера, "
                "затем /model — выберите модель.\n\n" + HELP_TEXT,
                attach_kb=True,
            )
            return

        if not text and not has_media:
            await self.send(chat_id, "Пока я понимаю текст, фото и файлы 🙏", attach_kb=True)
            return

        # фото и файлы → сразу в генерацию
        if has_media:
            if self.dialogs.get(chat_id):
                await self.send(chat_id, "Сначала завершите мастер или отмените его (/cancel), потом пришлите файл.")
                return
            if not self.can_chat(uid):
                await self.send(chat_id, "⛔ Общение с ботом ограничено администраторами.")
                return
            images, files_text, notes = await self.extract_media(msg)
            if notes:
                await self.send(chat_id, "⚠️ " + "\n⚠️ ".join(notes))
            if not images and not files_text and not text:
                return
            if not text:
                text = "Опиши подробно, что на изображении." if images else "Проанализируй содержимое файла."
            marker = text if (msg.get("text") or msg.get("caption")) else ("[Фото]" if images else "[Файл]")
            await self.generate_and_send(msg, text, images=images, files_text=files_text, marker=marker)
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
            await self.send(chat_id, f"Неизвестная команда /{cmd}. Смотрите /help", attach_kb=True)
            return

        if not self.can_chat(uid):
            await self.send(chat_id, "⛔ Общение с ботом ограничено администраторами.")
            return
        await self.generate_and_send(msg, text)

    async def dispatch_command(self, msg: Dict[str, Any], cmd: str, args: str) -> bool:
        chat_id = msg["chat"]["id"]
        uid = (msg.get("from") or {}).get("id", 0)

        if cmd in ("start", "help"):
            await self.send(chat_id, HELP_TEXT, attach_kb=True)
            return True

        if cmd == "id":
            await self.send(chat_id, f"Ваш ID: {uid}\nID чата: {chat_id}", attach_kb=True)
            return True

        if cmd == "cancel":
            if self.dialogs.pop(chat_id, None):
                await self.send(chat_id, "Мастер отменён.", attach_kb=True)
            else:
                await self.send(chat_id, "Нечего отменять.", attach_kb=True)
            return True

        if cmd == "model":
            extra = []
            st = self.chat_state(chat_id)
            if st.get("provider") and st.get("model"):
                extra = [("⟲ Сбросить на модель по умолчанию", "seldef")]
            kb = self.kb_providers("pv")
            if extra:
                kb["inline_keyboard"] = [extra] + kb["inline_keyboard"]
            await self.send(chat_id, "🧩 Выберите провайдера:", kb, attach_kb=True)
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
                    attach_kb=True,
                )
            else:
                await self.send(chat_id, "⚠️ Модель не выбрана. Используйте /model", attach_kb=True)
            return True

        if cmd == "clear":
            st = self.chat_state(chat_id)
            st["history"] = []
            self.save_state()
            await self.send(chat_id, "🧹 История диалога очищена.", attach_kb=True)
            return True

        if cmd == "web":
            if not self.is_admin(uid):
                await self.send(chat_id, "⛔ Команда только для администраторов.")
                return True
            on = not self.cfg.get("web_tools", True)
            self.cfg["web_tools"] = on
            self.save_config()
            if on:
                await self.send(chat_id,
                                "🌐 Веб-инструменты ВКЛЮЧЕНЫ: модель может искать в интернете "
                                "(web_search) и открывать страницы (read_url). "
                                "Провайдеры без поддержки tools отключаются для них автоматически.",
                                attach_kb=True)
            else:
                await self.send(chat_id,
                                "🌐 Веб-инструменты ВЫКЛЮЧЕНЫ — модель отвечает только из своих знаний.",
                                attach_kb=True)
            return True

        if cmd == "font":
            st = self.chat_state(chat_id)
            cur = st.get("font") or self.cfg.get("font", "rich")
            a = args.strip().lower()
            if a in FONTS:
                st["font"] = a
                self.save_state()
                cur = a
            await self.send(
                chat_id,
                f"🔤 Шрифт, которым пишет ИИ (сейчас: {FONTS[cur]}).\n"
                "Выберите из меню — настройка действует для этого чата.\n"
                "Глобальный вид по умолчанию — переменная FONT в окружении.",
                self.kb_font(cur),
                attach_kb=True,
            )
            return True

        if cmd == "get_id":
            if not args:
                await self.send(chat_id, "Использование: /get_id <имя_набора_стикеров>", attach_kb=True)
                return True
            try:
                ss = await self.api("getStickerSet", http_timeout=20, name=args.strip())
            except TgError as e:
                await self.send(chat_id, f"Ошибка: {e}", attach_kb=True)
                return True
            lines = [f"{s.get('emoji', '')} — {s['custom_emoji_id']}"
                     for s in ss.get("stickers", []) if s.get("custom_emoji_id")]
            await self.send(chat_id, "\n".join(lines) or "В наборе нет стикеров с custom_emoji_id.",
                            attach_kb=True)
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
            await self.send(chat_id, "\n".join(lines), attach_kb=True)
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
            await self.send(chat_id, "🗑 Какого провайдера удалить?", self.kb(rows), attach_kb=True)
            return True

        if cmd == "default":
            if not self.is_admin(uid):
                await self.send(chat_id, "⛔ Команда только для администраторов.")
                return True
            await self.send(chat_id, "⭐ Провайдер по умолчанию:", self.kb_providers("defpv", "noop"),
                            attach_kb=True)
            return True

        if cmd == "key":
            if not self.is_admin(uid):
                await self.send(chat_id, "⛔ Команда только для администраторов.")
                return True
            parts = args.split(maxsplit=1)
            if len(parts) < 2 or parts[0] not in self.cfg["providers"]:
                await self.send(chat_id, "Формат: /key <id> <новый ключ>\nID смотрите в /providers",
                                attach_kb=True)
                return True
            self.cfg["providers"][parts[0]]["api_key"] = parts[1].strip()
            self.models_cache.pop(parts[0], None)
            self.save_config()
            await self.send(chat_id, f"✅ Ключ {parts[0]} обновлён. Проверить: /model", attach_kb=True)
            return True

        if cmd == "url":
            if not self.is_admin(uid):
                await self.send(chat_id, "⛔ Команда только для администраторов.")
                return True
            parts = args.split(maxsplit=1)
            if len(parts) < 2 or parts[0] not in self.cfg["providers"]:
                await self.send(chat_id,
                                "Формат: /url <id> <baseURL>\nНапример: /url p1 https://openrouter.ai/api/v1",
                                attach_kb=True)
                return True
            url = parts[1].strip().rstrip("/")
            if not url.startswith(("http://", "https://")):
                await self.send(chat_id, "URL должен начинаться с http:// или https://", attach_kb=True)
                return True
            self.cfg["providers"][parts[0]]["base_url"] = url
            self.models_cache.pop(parts[0], None)
            self.save_config()
            await self.send(chat_id, f"✅ baseURL {parts[0]} → {url}", attach_kb=True)
            return True

        if cmd == "sys":
            if not self.is_admin(uid):
                await self.send(chat_id, "⛔ Команда только для администраторов.")
                return True
            if not args or args.lower() in ("off", "выкл"):
                self.cfg["system_prompt"] = ""
                self.save_config()
                await self.send(chat_id, "Системный промпт выключен.", attach_kb=True)
            else:
                self.cfg["system_prompt"] = args
                self.save_config()
                await self.send(chat_id, f"✅ Системный промпт сохранён:\n{args[:500]}", attach_kb=True)
            return True

        if cmd == "lock":
            if not self.is_admin(uid):
                await self.send(chat_id, "⛔ Команда только для администраторов.")
                return True
            self.cfg["admins_only"] = not self.cfg.get("admins_only", False)
            self.save_config()
            state = "включён — писать могут только администраторы" if self.cfg["admins_only"] else "выключен — писать могут все"
            await self.send(chat_id, f"🔒 Режим «только администраторы» {state}.", attach_kb=True)
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
            "«🔗 Своя ссылка (авто)» — просто URL, бот сам определит тип API\n"
            "и сам загрузит список моделей. Ключ не обязателен — подойдёт\n"
            "самохостинг без ключа (Ollama / LM Studio / vLLM).\n"
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
            note = ""
            if data.get("auto"):
                data["protocol"] = detect_protocol(url)
                m = re.match(r"https?://([^/\s]+)", url)
                data["name"] = m.group(1) if m else "Мой провайдер"
                note = (f"🔌 Тип API определён: {PROTOCOL_TITLES.get(data['protocol'], data['protocol'])}\n")
            dlg["step"] = "key"
            await self.send(
                chat_id,
                f"{note}🔑 Теперь пришлите API-ключ (или /skip, если сервер без ключа):",
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
            await self.send(chat_id, f"{status}\n🔍 Проверяю ключ и загружаю список моделей…")
            try:
                models = await self.get_models(pid, force=True)
                if models:
                    await self.send(
                        chat_id,
                        f"📦 Доступно моделей: {len(models)} — ключ работает.\n\n"
                        f"Теперь: /model — выбрать модель (кнопкой 🔄 список моделей\n"
                        f"можно перезагрузить в любой момент) или /default — сделать модель\n"
                        f"дефолтной для новых чатов.",
                    )
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
                "auto": preset_id == "custom-auto",
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
                if preset_id == "custom-auto":
                    txt = ("Пришлите просто ссылку на API (baseURL), например:\n"
                           "https://api.example.com/v1\n\n"
                           "Я сам определю тип API (OpenAI / Anthropic / Gemini),\n"
                           "проверю ключ и загружу список доступных моделей.\n"
                           "Ключ можно пропустить (/skip) — для Ollama, LM Studio и т.п.")
                else:
                    txt = "Пришлите baseURL, например:\nhttps://api.example.com/v1"
                await self.edit(
                    chat_id,
                    msg.get("message_id", 0),
                    f"✅ {preset['name']}\n{txt}",
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

        # --- быстрая смена модели (кнопка под ответом бота) ---
        if data == "qmodel":
            st = self.chat_state(chat_id)
            kb = self.kb_providers("pv")
            if st.get("provider") and st.get("model"):
                kb["inline_keyboard"] = [[("⟲ Сбросить на модель по умолчанию", "seldef")]] + kb["inline_keyboard"]
            await self.edit(chat_id, msg.get("message_id", 0), "🧩 Выберите провайдера:", kb)
            return

        # --- выбор шрифта, которым пишет ИИ (меню /font) ---
        if data.startswith("font:"):
            f = data[5:]
            if f in FONTS:
                st = self.chat_state(chat_id)
                st["font"] = f
                self.save_state()
                await self.edit(
                    chat_id,
                    msg.get("message_id", 0),
                    f"🔤 Шрифт ответов ИИ: {FONTS[f]}\nСледующий ответ придёт в этом стиле.",
                    self.kb_font(f),
                )
            return

        # --- обновление списка моделей с сервера (🔄) ---
        m = re.match(r"^rm:(sel|defsel):([^:]+)$", data)
        if m:
            pick, pid = m.group(1), m.group(2)
            back = "bk" if pick == "sel" else "defbk"
            provider = self.cfg["providers"].get(pid)
            if not provider:
                await self.edit(chat_id, msg.get("message_id", 0), "⚠️ Провайдер удалён. Откройте /model заново.")
                return
            try:
                await self.get_models(pid, force=True)
            except ProviderError as e:
                await self.edit(chat_id, msg.get("message_id", 0),
                                f"⚠️ Не удалось обновить список моделей:\n{e}")
                return
            kb, err = await self.kb_models(pid, 0, pick, back)
            if err:
                await self.edit(chat_id, msg.get("message_id", 0), f"⚠️ {err}")
                return
            title = "🤖" if pick == "sel" else "⭐"
            n = len(self.models_cache.get(pid, (0, []))[1])
            await self.edit(
                chat_id,
                msg.get("message_id", 0),
                f"{title} {provider['name']} — список моделей обновлён ({n} шт.):",
                kb,
            )
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
