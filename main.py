#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TG AI Bot — Telegram-бот с подключаемыми AI-провайдерами (как в opencode).

Что умеет:
    • Анимации в стиле «черновика»: живой статус «💭 Думаю… 7с» (кастомное эмодзи),
      «Ищу в интернете · запрос», «Читаю страницу · сайт», затем текст ответа
      печатается на лету (sendMessageDraft). Если Telegram не поддерживает
      черновики — автоматически включается запасной вариант через редактирование.
    • 15 СТИЛЕЙ ШРИФТА ответов (меню 🔤 Шрифт или /font): Claude-стиль, markdown,
      моно, блок кода, жирный, курсив, жирный курсив, подчёркнутый, цитата,
      спойлер, рукопись, готика, двойной, гротеск, широкий.
      ВАЖНО: Telegram не умеет подгружать настоящие шрифты — это либо родное
      оформление (работает и на русском), либо Unicode-алфавиты (меняют латиницу;
      для кириллицы бот добавляет родное жирное/курсив/подчёркивание).
    • Кнопки под ответом: 🔄 Ещё раз · 🔤 Шрифт · 🤖 Сменить модель.
    • Фото (vision) и файлы (код, txt, csv, json…), веб-поиск и чтение страниц
      (tool-calling во всех трёх протоколах), все команды — кнопками под полем ввода.

Локально:
    pip install aiohttp
    BOT_TOKEN=123:ABC python main.py

На Render (Web Service) — переменные окружения:
    BOT_TOKEN, ADMIN_IDS, PROVIDER_1..N (пресет:ключ или Имя|протокол|URL|ключ),
    CUSTOM_URL / CUSTOM_KEY / CUSTOM_NAME / CUSTOM_PROTOCOL / CUSTOM_MODEL,
    DEFAULT_MODEL, DEFAULT_PROVIDER, SYSTEM_PROMPT, MAX_HISTORY, ADMIN_ONLY,
    WEB_TOOLS (0/1), FONT (ключ шрифта, по умолчанию claude),
    CUSTOM_EMOJI_ID (id кастомного эмодзи для анимации; пусто = обычное 💭).
    Render сам задаёт PORT и RENDER_EXTERNAL_URL — бот поднимет webhook.

Зависимость одна: aiohttp. Python 3.9+.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import json
import os
import random
import re
import secrets
import signal
import sys
import time
from datetime import date
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple
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
TG_FILE_BASE = "https://api.telegram.org/file/bot{token}/{path}"

PAGE_SIZE = 12          # моделей на страницу в клавиатуре
MODELS_TTL = 300        # секунд кэша списка моделей
MAX_TG = 4000           # безопасный лимит длины сообщения Telegram
MAX_TOKENS_ANTHROPIC = 8192

# --- анимация ---
PLACEHOLDER_EMOJI = "💭"   # 2 UTF-16 единицы — под него считается length у entity
CUSTOM_EMOJI_ID = os.environ.get("CUSTOM_EMOJI_ID", "5395546579631388253").strip()

# --- медиа и инструменты ---
MAX_MEDIA_BYTES = 10 * 1024 * 1024
MAX_FILE_TEXT = 15000
MAX_TOOL_ROUNDS = 4
UA = {"User-Agent": "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120 Mobile Safari/537.36"}

# ----------------------------------------------------------------------------
# Шрифты (15 стилей)
# ----------------------------------------------------------------------------

FONTS: Dict[str, str] = {
    "claude": "🟠 Claude — чистый стиль",
    "rich": "🖋 Обычный (markdown)",
    "mono": "💻 Моноширинный",
    "pre": "🧱 Блок кода",
    "bold": "🅱️ Жирный",
    "italic": "✏️ Курсив",
    "bolditalic": "🔥 Жирный курсив",
    "underline": "➖ Подчёркнутый",
    "quote": "💬 Цитата",
    "spoiler": "🫥 Спойлер",
    "script": "🪶 Рукопись 𝓢𝓬𝓻𝓲𝓹𝓽",
    "fraktur": "🏰 Готика 𝕱𝖗𝖆𝖐𝖙𝖚𝖗",
    "double": "🔳 Двойной 𝔻𝕠𝕦𝕓𝕝𝕖",
    "sans": "🔠 Гротеск 𝗦𝗮𝗻𝘀",
    "wide": "🌐 Широкий ｗｉｄｅ",
}
FONT_ORDER = list(FONTS)
DEFAULT_FONT = "claude"
FONT_ALIASES = {
    "клод": "claude", "клауд": "claude", "чистый": "claude",
    "обычный": "rich", "маркдаун": "rich", "markdown": "rich",
    "моно": "mono", "моноширинный": "mono", "код": "pre", "блок": "pre",
    "жирный": "bold", "курсив": "italic", "жирныйкурсив": "bolditalic",
    "подчёркнутый": "underline", "подчеркнутый": "underline",
    "цитата": "quote", "спойлер": "spoiler",
    "рукопись": "script", "готика": "fraktur", "двойной": "double",
    "гротеск": "sans", "широкий": "wide",
    "простой": "rich", "plain": "rich",
}


def _mk_alpha(up: int, lo: int, dg: Optional[int] = None,
              exc: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for i in range(26):
        m[chr(65 + i)] = chr(up + i)
        m[chr(97 + i)] = chr(lo + i)
    if dg is not None:
        for i in range(10):
            m[chr(48 + i)] = chr(dg + i)
    if exc:
        m.update(exc)
    return m


# Unicode-алфавиты (Mathematical Alphanumeric / Fullwidth) — меняют только латиницу и цифры
UNI_MAPS: Dict[str, Dict[str, str]] = {
    "script": _mk_alpha(0x1D4D0, 0x1D4EA),
    "fraktur": _mk_alpha(0x1D56C, 0x1D586),
    "double": _mk_alpha(0x1D538, 0x1D552, 0x1D7D8,
                        {"C": "ℂ", "H": "ℍ", "N": "ℕ", "P": "ℙ", "Q": "ℚ", "R": "ℝ", "Z": "ℤ"}),
    "sans": _mk_alpha(0x1D5D4, 0x1D5EE, 0x1D7EC),
    "wide": _mk_alpha(0xFF21, 0xFF41, 0xFF10),
}
# чем оформить КИРИЛЛИЦУ в Unicode-стилях (в них нет русских букв)
UNI_CYR_TAG: Dict[str, Optional[str]] = {
    "script": "i", "fraktur": "b", "double": "u", "sans": "b", "wide": None,
}
_CYR = re.compile(r"[\u0400-\u04FF](?:[\u0400-\u04FF ,.\-]*[\u0400-\u04FF])?")

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
KB_USER_ROWS = [
    ["🤖 Модель", "📊 Статус"],
    ["🔤 Шрифт", "🧹 Очистить"],
    ["📋 Помощь", "🆔 Мой ID"],
]
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
    "custom-auto": {"name": "🔗 Своя ссылка (авто)", "protocol": "", "base_url": "", "docs": "",
                    "hint": "Просто URL — бот сам определит тип API"},
    "custom-openai": {"name": "Свой сервер (OpenAI-compatible)", "protocol": "openai", "base_url": "",
                      "docs": "", "hint": "Любой ваш URL: vLLM, LM Studio, прокси…"},
    "custom-anthropic": {"name": "Свой сервер (Anthropic-compatible)", "protocol": "anthropic",
                         "base_url": "", "docs": "", "hint": "Любой клон Claude API"},
    "custom-gemini": {"name": "Свой сервер (Gemini-compatible)", "protocol": "gemini", "base_url": "",
                      "docs": "", "hint": "Любой прокси Gemini API"},
    "openrouter": {"name": "OpenRouter", "protocol": "openai", "base_url": "https://openrouter.ai/api/v1",
                   "docs": "https://openrouter.ai/keys", "hint": "Сотни моделей одним ключом"},
    "openai": {"name": "OpenAI", "protocol": "openai", "base_url": "https://api.openai.com/v1",
               "docs": "https://platform.openai.com/api-keys", "hint": "GPT-4o, GPT-4.1, o3"},
    "anthropic": {"name": "Anthropic", "protocol": "anthropic", "base_url": "https://api.anthropic.com",
                  "docs": "https://console.anthropic.com/settings/keys", "hint": "Claude Opus / Sonnet / Haiku"},
    "google": {"name": "Google Gemini", "protocol": "gemini",
               "base_url": "https://generativelanguage.googleapis.com",
               "docs": "https://aistudio.google.com/apikey", "hint": "Gemini 2.5 Pro / Flash"},
    "zai": {"name": "Z.AI (Zhipu)", "protocol": "openai", "base_url": "https://api.z.ai/api/paas/v4",
            "docs": "https://z.ai/manage-apikey/apikey-list", "hint": "GLM-4.6, GLM-4.5"},
    "groq": {"name": "Groq", "protocol": "openai", "base_url": "https://api.groq.com/openai/v1",
             "docs": "https://console.groq.com/keys", "hint": "Llama, Qwen — быстрый инференс"},
    "deepseek": {"name": "DeepSeek", "protocol": "openai", "base_url": "https://api.deepseek.com/v1",
                 "docs": "https://platform.deepseek.com/api_keys", "hint": "DeepSeek-V3, R1"},
    "mistral": {"name": "Mistral AI", "protocol": "openai", "base_url": "https://api.mistral.ai/v1",
                "docs": "https://console.mistral.ai/api-keys", "hint": "Mistral Large / Medium"},
    "xai": {"name": "xAI (Grok)", "protocol": "openai", "base_url": "https://api.x.ai/v1",
            "docs": "https://console.x.ai", "hint": "Grok-4, Grok-3"},
    "moonshot": {"name": "Moonshot (Kimi)", "protocol": "openai", "base_url": "https://api.moonshot.ai/v1",
                 "docs": "https://platform.moonshot.ai/console/api-keys", "hint": "Kimi K2"},
    "together": {"name": "Together AI", "protocol": "openai", "base_url": "https://api.together.xyz/v1",
                 "docs": "https://api.together.ai/settings/api-keys", "hint": "Open-source модели"},
    "fireworks": {"name": "Fireworks AI", "protocol": "openai",
                  "base_url": "https://api.fireworks.ai/inference/v1",
                  "docs": "https://fireworks.ai/account/api-keys", "hint": "Хостинг open-source"},
    "cerebras": {"name": "Cerebras", "protocol": "openai", "base_url": "https://api.cerebras.ai/v1",
                 "docs": "https://cloud.cerebras.ai", "hint": "Сверхбыстрый инференс"},
}

# Справка в HTML (спецсимволы экранированы вручную)
HELP_HTML = "\n".join([
    "<b>🤖 AI-бот с подключаемыми провайдерами</b>",
    "",
    "💬 Просто напишите сообщение — отвечу через выбранную модель.",
    "📷 Присылайте фото (модель их увидит) и файлы: код, txt, csv, json…",
    "🌐 Сам ищу в интернете и открываю ссылки, если нужны свежие данные.",
    "⌨️ Все команды — кнопками под полем ввода.",
    "",
    "<b>Чат и модель</b>",
    "/model — выбрать модель для этого чата",
    "/status — что сейчас выбрано",
    "/clear — очистить историю диалога",
    "/id — показать ID",
    "",
    "<b>Оформление</b>",
    "/font — 15 стилей шрифта ответов (Claude, моно, готика, рукопись…)",
    "",
    "<b>Провайдеры (админ)</b>",
    "/providers — список провайдеров",
    "/add — подключить провайдера (своя ссылка или пресет)",
    "/del — удалить провайдера",
    "/default — модель по умолчанию",
    "/key &lt;id&gt; &lt;ключ&gt; — сменить API-ключ",
    "/url &lt;id&gt; &lt;url&gt; — сменить baseURL",
    "/web — вкл/выкл веб-инструменты",
    "",
    "<b>Настройки (админ)</b>",
    "/sys — системный промпт (/sys &lt;текст&gt; · /sys off)",
    "/lock — режим «только администраторы»",
    "",
    "<b>Прочее</b>",
    "/get_id &lt;набор&gt; — custom_emoji_id стикеров набора",
    "/cancel — отменить текущий мастер",
])


class ProviderError(RuntimeError):
    """Ошибка на стороне провайдера (показываем пользователю как есть)."""


class TgError(RuntimeError):
    """Ошибка Telegram Bot API."""


def detect_protocol(url: str) -> str:
    """Определить протокол API по ссылке: anthropic / gemini / иначе OpenAI-compatible."""
    u = (url or "").lower()
    if "anthropic" in u:
        return "anthropic"
    if "generativelanguage" in u or "gemini" in u or "aiplatform.googleapis" in u:
        return "gemini"
    return "openai"


# ----------------------------------------------------------------------------
# Работа с файлами (config.json / state.json)
# ----------------------------------------------------------------------------

def default_config() -> Dict[str, Any]:
    return {
        "bot_token": "",
        "admins": [],
        "admins_only": False,
        "system_prompt": "",
        "max_history": 24,
        "web_tools": True,
        "font": DEFAULT_FONT,
        "default": {"provider": "", "model": ""},
        "providers": {},
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
                merged = json.loads(json.dumps(fallback))
                merged.update(data)
                return merged
            return data
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"⚠️ Не удалось прочитать {path}: {e} — начинаю с чистого файла")
    return json.loads(json.dumps(fallback))


def clean_secret(s: str) -> str:
    """Убрать пробелы и случайные обрамляющие кавычки (частая ошибка в дашборде Render)."""
    s = (s or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def apply_env_overrides(cfg: Dict[str, Any]) -> None:
    applied: List[str] = []
    ids_raw = os.environ.get("ADMIN_IDS", "").strip()
    if ids_raw:
        ids = [int(x) for x in re.split(r"[,;\s]+", ids_raw) if x.lstrip("-").isdigit()]
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
    ft = FONT_ALIASES.get(ft, ft)
    if ft in FONTS:
        cfg["font"] = ft
        applied.append(f"FONT={ft}")
    if cfg.get("font") not in FONTS:
        cfg["font"] = DEFAULT_FONT
    if applied:
        print("Из окружения применены: " + ", ".join(applied))


def parse_provider_spec(raw: str) -> Optional[Dict[str, str]]:
    """«пресет:ключ» или «Имя|протокол|URL|ключ» (протокол auto — по URL)."""
    raw = raw.strip()
    if not raw:
        return None
    if "|" not in raw and ":" in raw:
        preset_id, _, key = raw.partition(":")
        pid = preset_id.strip().lower()
        if pid == "gemini":
            pid = "google"
        preset = PRESETS.get(pid)
        if preset and preset.get("base_url"):
            return {"name": preset["name"], "protocol": preset["protocol"],
                    "base_url": preset["base_url"], "api_key": key.strip()}
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
            spec["_idx"] = num
            out.append(spec)
        else:
            print(f"⚠️ PROVIDER_{num}: не понял формат «{raw[:60]}…». "
                  "Примеры: openrouter:sk-… или Имя|openai|https://…/v1|ключ")
    return out


def sync_env_providers(cfg: Dict[str, Any]) -> None:
    """Провайдеры из PROVIDER_1..N и CUSTOM_* — применяются при каждом старте."""
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
        providers[pid] = {"name": name, "protocol": spec["protocol"], "base_url": spec["base_url"],
                          "api_key": spec["api_key"], "enabled": True}
        names.append(f"{pid}={name}")

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
            providers["ec"] = {"name": cname, "protocol": proto, "base_url": c_url,
                               "api_key": clean_secret(os.environ.get("CUSTOM_KEY", "")),
                               "enabled": True}
            names.append(f"ec={cname}")
            cm = clean_secret(os.environ.get("CUSTOM_MODEL", ""))
            if cm and not os.environ.get("DEFAULT_MODEL", "").strip():
                cfg["default"] = {"provider": "ec", "model": cm}
                names.append(f"по умолчанию: {cm}")

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


def tg_hint(err: str) -> str:
    low = err.lower()
    if "401" in err or "unauthorized" in low:
        return ("Токен неверного формата — должен быть вида 123456789:ABC-DEF… "
                "(без кавычек и пробелов). Проверьте BOT_TOKEN.")
    if "404" in err or "not found" in low:
        return ("Telegram не находит бота с этим токеном. Проверьте бота у @BotFather (/mybots); "
                "если токен отозван — выпустите новый и обновите BOT_TOKEN.")
    if "409" in err or "conflict" in low:
        return ("Конфликт: webhook или getUpdates заняты другим экземпляром бота. "
                "Остановите второй экземпляр и перезапустите сервис.")
    if "429" in err or "too many requests" in low:
        return "Слишком много запросов к Telegram. Подождите пару минут и перезапустите сервис."
    if "400" in err or "bad request" in low:
        return ("Telegram отклонил параметры запроса (обычно URL вебхука). Проверьте, что "
                "RENDER_EXTERNAL_URL / WEBHOOK_URL — валидный https-адрес.")
    return "Проверьте BOT_TOKEN и настройки сервиса, затем перезапустите."


def first_run_setup(cfg: Dict[str, Any]) -> None:
    env = clean_secret(os.environ.get("BOT_TOKEN", ""))
    if env:
        if not re.fullmatch(r"\d+:[\w-]+", env):
            print("⚠️ BOT_TOKEN не похож на токен Telegram (вида 123456789:ABC-DEF…). "
                  "Проверьте переменную — нет ли лишних символов, кавычек или пробелов.")
        if cfg.get("bot_token") != env:
            cfg["bot_token"] = env
            _atomic_write(CONFIG_PATH, cfg)
            print("Токен взят из переменной окружения BOT_TOKEN")
        return
    if cfg["bot_token"]:
        return
    if not (sys.stdin and sys.stdin.isatty()):
        sys.exit("❌ BOT_TOKEN не задан. Укажите переменную окружения BOT_TOKEN "
                 "(токен от @BotFather) или впишите его в config.json.")
    print("=" * 60)
    print("Первый запуск. Создайте бота у @BotFather и вставьте его токен.")
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
# Клиенты провайдеров (три протокола)
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
    proto = provider["protocol"]
    base = provider["base_url"].rstrip("/")
    key = provider.get("api_key", "")

    if proto == "gemini":
        vb = "" if re.search(r"/v\d\w*$", base) else "/v1beta"
        url = f"{base}{vb}/models?pageSize=200"
        headers = {"x-goog-api-key": key} if key else {}
        async with session.get(url, headers=headers) as r:
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
        v1 = "" if re.search(r"/v\d$", base) else "/v1"
        async with session.get(f"{base}{v1}/models?limit=1000", headers=headers) as r:
            data = await _json_or_error(r)
        out = []
        for m in data.get("data", []):
            mid = m.get("id") or ""
            if mid:
                out.append((mid, m.get("display_name") or m.get("name") or mid))
        return out

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


def _tools_for(proto: str) -> List[Dict[str, Any]]:
    if proto == "anthropic":
        return [{"name": t["function"]["name"], "description": t["function"]["description"],
                 "input_schema": t["function"]["parameters"]} for t in TOOLS_SPEC]
    if proto == "gemini":
        return [{"functionDeclarations": [
            {"name": t["function"]["name"], "description": t["function"]["description"],
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
            out.append({"type": "image_url", "image_url": {
                "url": f"data:{p.get('media_type', 'image/jpeg')};base64,{p.get('data', '')}"}})
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


def flatten_tools(conv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Свернуть вызовы инструментов в обычный текст — для финального ответа без tools
    (Anthropic/Gemini ругаются на tool-блоки в истории, если tools не переданы)."""
    out: List[Dict[str, Any]] = []
    results: List[str] = []
    for m in conv:
        if m["role"] == "assistant" and m.get("tool_calls"):
            continue
        if m["role"] == "tool":
            results.append(f"[{m.get('name', 'tool')}]\n{m['content']}")
            continue
        out.append(m)
    if results:
        out.append({"role": "user", "content": "Результаты инструментов:\n\n" + "\n\n".join(results)
                    + "\n\nОтветь пользователю на основе этих данных, коротко укажи источники."})
    return out


async def stream_chat(
    session: "aiohttp.ClientSession",
    provider: Dict[str, Any],
    model: str,
    messages: List[Dict[str, Any]],
    tools: bool = False,
) -> AsyncGenerator[Tuple[str, Any], None]:
    """Стриминг ответа модели. События: ("text", "кусок") и ("tools", [{id,name,args}])."""
    proto = provider["protocol"]
    base = provider["base_url"].rstrip("/")
    key = provider.get("api_key", "")
    system = "\n\n".join(m["content"] for m in messages
                         if m["role"] == "system" and isinstance(m.get("content"), str))
    rest = [m for m in messages if m["role"] != "system"]
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=300)
    urls: List[str] = []
    url = ""

    if proto == "openai":
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
        if system:
            msgs.insert(0, {"role": "system", "content": system})
        payload: Dict[str, Any] = {"model": model, "messages": msgs, "stream": True}
        if tools:
            payload["tools"] = _tools_for(proto)
    elif proto == "anthropic":
        v1 = "" if re.search(r"/v\d$", base) else "/v1"
        url = f"{base}{v1}/messages"
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
                a_msgs.append({"role": m["role"], "content": _anth_content(m["content"])})
        flush_tools()
        payload = {"model": model, "max_tokens": MAX_TOKENS_ANTHROPIC, "messages": a_msgs, "stream": True}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = _tools_for(proto)
    elif proto == "gemini":
        vb = "" if re.search(r"/v\d\w*$", base) else "/v1beta"
        url = f"{base}{vb}/models/{model}:streamGenerateContent?alt=sse"
        headers = {"Content-Type": "application/json"}
        if key:
            headers["x-goog-api-key"] = key
        g_msgs: List[Dict[str, Any]] = []
        pend_g: List[Dict[str, Any]] = []

        def flush_tools_g() -> None:
            if pend_g:
                g_msgs.append({"role": "user", "parts": [
                    {"functionResponse": {"name": t.get("name", "tool"),
                                          "response": {"result": t["content"]}}}
                    for t in pend_g]})
                pend_g.clear()

        for m in rest:
            if m["role"] == "assistant" and m.get("tool_calls"):
                flush_tools_g()
                parts: List[Dict[str, Any]] = []
                if m.get("content"):
                    parts.append({"text": m["content"]})
                for c in m["tool_calls"]:
                    p: Dict[str, Any] = {"functionCall": {"name": c["name"], "args": _safe_json(c["args"])}}
                    if c.get("sig"):
                        p["thoughtSignature"] = c["sig"]
                    parts.append(p)
                g_msgs.append({"role": "model", "parts": parts})
            elif m["role"] == "tool":
                pend_g.append(m)
            else:
                flush_tools_g()
                g_msgs.append({"role": "model" if m["role"] == "assistant" else "user",
                               "parts": _gem_parts(m["content"])})
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

    def gem_call(part: Dict[str, Any], idx: int) -> Dict[str, str]:
        fc = part["functionCall"]
        c = {"id": f"call{idx}", "name": fc.get("name", ""),
             "args": json.dumps(fc.get("args") or {}, ensure_ascii=False)}
        if part.get("thoughtSignature"):
            c["sig"] = part["thoughtSignature"]
        return c

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

        async with resp:  # type: ignore[union-attr]
            if resp.status >= 400:  # type: ignore[union-attr]
                text = (await resp.text())[:400].replace("\n", " ")  # type: ignore[union-attr]
                raise ProviderError(f"HTTP {resp.status}: {text}")  # type: ignore[union-attr]

            ct = (resp.headers.get("Content-Type") or "").lower()  # type: ignore[union-attr]
            if "event-stream" not in ct:
                # некоторые серверы игнорируют stream:true и отвечают обычным JSON
                try:
                    obj = await resp.json(content_type=None)  # type: ignore[union-attr]
                except Exception:
                    raise ProviderError(f"HTTP {resp.status}: ответ не JSON и не поток")  # type: ignore[union-attr]
                body = ""
                calls_out: List[Dict[str, str]] = []
                if proto == "openai":
                    ch = (obj.get("choices") or [{}])[0]
                    mm = ch.get("message") or {}
                    body = mm.get("content") or ""
                    for i, tc in enumerate(mm.get("tool_calls") or []):
                        fn = tc.get("function") or {}
                        calls_out.append({"id": tc.get("id") or f"call{i}", "name": fn.get("name", ""),
                                          "args": fn.get("arguments") or "{}"})
                elif proto == "anthropic":
                    for b in obj.get("content") or []:
                        if b.get("type") == "text":
                            body += b.get("text", "")
                        elif b.get("type") == "tool_use":
                            calls_out.append({"id": b.get("id", ""), "name": b.get("name", ""),
                                              "args": json.dumps(b.get("input") or {}, ensure_ascii=False)})
                else:
                    for cand in obj.get("candidates") or []:
                        for part in ((cand.get("content") or {}).get("parts") or []):
                            if part.get("thought"):
                                continue
                            if part.get("text"):
                                body += part["text"]
                            if part.get("functionCall"):
                                calls_out.append(gem_call(part, len(calls_out)))
                if body:
                    yield ("text", body)
                if calls_out:
                    yield ("tools", calls_out)
                return

            async for line in _sse_data_lines(resp):  # type: ignore[arg-type]
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
                else:
                    for cand in obj.get("candidates") or []:
                        for part in ((cand.get("content") or {}).get("parts") or []):
                            if part.get("thought"):
                                continue
                            if part.get("text"):
                                yield ("text", part["text"])
                            if part.get("functionCall"):
                                gem_calls.append(gem_call(part, len(gem_calls)))

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


# --- веб-инструменты ---

def _clean_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


async def web_search(session: "aiohttp.ClientSession", query: str, limit: int = 6) -> str:
    if not query.strip():
        return "Пустой запрос."
    page = ""
    try:
        async with session.post("https://html.duckduckgo.com/html/", data={"q": query, "kl": "ru-ru"},
                                headers=UA, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status < 400:
                page = await r.text(errors="ignore")
    except Exception:  # noqa: BLE001
        pass
    links = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.S)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)
    if not links:
        try:
            async with session.get("https://lite.duckduckgo.com/lite/", params={"q": query},
                                   headers=UA, timeout=aiohttp.ClientTimeout(total=15)) as r:
                page = await r.text(errors="ignore")
        except Exception as e:  # noqa: BLE001
            return f"Ошибка поиска: {e}"
        links = re.findall(r'class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.S)
        snippets = re.findall(r"class=\"result-snippet\"[^>]*>(.*?)</td>", page, re.S)
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
        async with session.get(url, headers=UA, timeout=aiohttp.ClientTimeout(total=15)) as r:
            page = await r.text(errors="ignore")
    except Exception as e:  # noqa: BLE001
        return f"Не удалось открыть: {e}"
    page = re.sub(r"(?is)<(script|style|noscript|svg|header|footer|nav).*?</\1>", " ", page)
    text = _clean_html(re.sub(r"(?s)<[^>]+>", " ", page))
    text = re.sub(r"\s+", " ", text)
    return text[:4000] or "Страница пустая."


async def run_tool(session: Optional["aiohttp.ClientSession"], name: str, args_json: str) -> str:
    args = _safe_json(args_json)
    if session is None:
        return "Сессия недоступна."
    if name == "web_search":
        return await web_search(session, args.get("query", ""))
    if name == "read_url":
        return await read_url(session, args.get("url", ""))
    return "Неизвестный инструмент."


def tool_label(c: Dict[str, str]) -> str:
    """Подпись статуса при работе инструмента: «Ищу в интернете · запрос»."""
    a = _safe_json(c.get("args", ""))
    if c.get("name") == "web_search":
        q = (a.get("query") or "").strip()
        return "Ищу в интернете" + (f" · {q[:40]}" if q else "")
    if c.get("name") == "read_url":
        m = re.match(r"https?://([^/\s]+)", a.get("url", "") or "")
        return "Читаю страницу" + (f" · {m.group(1)}" if m else "")
    return "Работаю"


def _is_tools_error(err: str) -> bool:
    low = (err or "").lower()
    mentions = any(m in low for m in ("tools", "tool_use", "function", "functioncall"))
    broken = any(m in low for m in ("not support", "unsupported", "unknown", "unrecognized",
                                    "invalid", "unexpected", "400"))
    return mentions and broken


# ----------------------------------------------------------------------------
# Оформление ответов: markdown → Telegram HTML и 15 шрифтов
# ----------------------------------------------------------------------------

_SEP_ROW = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _table_to_text(block: str) -> str:
    rows: List[List[str]] = []
    for ln in block.strip("\n").split("\n"):
        if _SEP_ROW.match(ln):
            continue
        cells = [re.sub(r"\*\*|`", "", c).strip() for c in ln.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return block
    n = max(len(r) for r in rows)
    for r in rows:
        r.extend([""] * (n - len(r)))
    w = [max(len(r[i]) for r in rows) for i in range(n)]
    lines = [" │ ".join(r[i].ljust(w[i]) for i in range(n)).rstrip() for r in rows]
    if len(lines) > 1:
        lines.insert(1, "─┼─".join("─" * x for x in w))
    return "\n".join(lines)


def md_to_html(text: str, pretty: bool = False) -> str:
    """Markdown → Telegram HTML. Код и таблицы прячутся в заглушки, чтобы разметка
    внутри них не ломалась. Поддержка: **жирный**, *курсив*, _курсив_, ~~зачёркнутый~~,
    ||спойлер||, `код`, ```блоки```, # заголовки, > цитаты, списки, таблицы, [ссылки](url)."""
    stash: List[str] = []

    def keep(s: str) -> str:
        stash.append(s)
        return f"\x00{len(stash) - 1}\x00"

    text = text.replace("\x00", "")

    def fence(m: "re.Match[str]") -> str:
        lang, body = m.group(1), html.escape(m.group(2).rstrip("\n"), quote=False)
        if lang:
            return keep(f'<pre><code class="language-{lang}">{body}</code></pre>')
        return keep(f"<pre>{body}</pre>")

    text = re.sub(r"```([\w+#.\-]*)[ \t]*\n?(.*?)```", fence, text, flags=re.S)
    text = re.sub(r"`([^`\n]+)`",
                  lambda m: keep("<code>" + html.escape(m.group(1), quote=False) + "</code>"), text)
    text = re.sub(r"(?:^[ \t]*\|[^\n]*\|[ \t]*(?:\n|$)){2,}",
                  lambda m: keep("<pre>" + html.escape(_table_to_text(m.group(0)), quote=False)
                                 + "</pre>\n"), text, flags=re.M)

    text = html.escape(text, quote=False)

    def quote(m: "re.Match[str]") -> str:
        raw = m.group(0)
        inner = "\n".join(re.sub(r"^&gt;[ \t]?", "", ln) for ln in raw.rstrip("\n").split("\n"))
        return f"<blockquote>{inner}</blockquote>" + ("\n" if raw.endswith("\n") else "")

    text = re.sub(r"(?:^&gt;[^\n]*(?:\n|$))+", quote, text, flags=re.M)
    text = re.sub(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)",
                  lambda m: f'<a href="{m.group(2).replace(chr(34), "%22")}">{m.group(1)}</a>', text)
    text = re.sub(r"^([ \t]*)[-*+][ \t]+",
                  lambda m: m.group(1) + ("• " if len(m.group(1).replace("\t", "  ")) < 2 else "◦ "),
                  text, flags=re.M)
    text = re.sub(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", "──────────", text, flags=re.M)
    text = re.sub(r"^#{1,6}[ \t]+(.*?)[ \t]*#*$",
                  lambda m: "<b>" + m.group(1).replace("**", "") + "</b>", text, flags=re.M)
    text = re.sub(r"(?<![\w_])__(.+?)__(?!\w)", r"<b>\1</b>", text, flags=re.S)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.S)
    text = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_(?!\s)([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"~~([^~\n]+)~~", r"<s>\1</s>", text)
    text = re.sub(r"\|\|([^|\n]+)\|\|", r"<tg-spoiler>\1</tg-spoiler>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
    if pretty:
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


_TAG_RE = re.compile(r"(<[^>]+>)")


def map_runs(h: str, fn: Callable[[str, List[str]], str]) -> str:
    """Применить fn к текстовым кускам HTML вне <code>/<pre>; fn получает стек открытых тегов."""
    out: List[str] = []
    stack: List[str] = []
    for tok in _TAG_RE.split(h):
        if not tok:
            continue
        if tok[0] == "<":
            m = re.match(r"</?\s*([a-zA-Z\-]+)", tok)
            name = m.group(1).lower() if m else ""
            if tok.startswith("</"):
                if name in stack:
                    idx = len(stack) - 1 - stack[::-1].index(name)
                    del stack[idx:]
            else:
                stack.append(name)
            out.append(tok)
        elif "code" in stack or "pre" in stack:
            out.append(tok)
        else:
            out.append(fn(tok, stack))
    return "".join(out)


def _wrap(tag: str) -> Callable[[str, List[str]], str]:
    def f(t: str, st: List[str]) -> str:
        if tag in st or not t.strip():
            return t
        return f"<{tag}>{t}</{tag}>"
    return f


def _uni(mapping: Dict[str, str], cyr_tag: Optional[str]) -> Callable[[str, List[str]], str]:
    def f(t: str, st: List[str]) -> str:
        res: List[str] = []
        for p in re.split(r"(&#?\w+;)", t):  # html-сущности не трогаем
            if re.fullmatch(r"&#?\w+;", p):
                res.append(p)
                continue
            p = "".join(mapping.get(c, c) for c in p)
            if cyr_tag and cyr_tag not in st:
                p = _CYR.sub(lambda m: f"<{cyr_tag}>{m.group(0)}</{cyr_tag}>", p)
            res.append(p)
        return "".join(res)
    return f


def fmt_mono(text: str) -> str:
    text = re.sub(r"```\w*\n?", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.S)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)
    return f"<code>{html.escape(text.strip(), quote=False)}</code>"


def fmt_pre(text: str) -> str:
    text = re.sub(r"```\w*\n?", "", text)
    return f"<pre>{html.escape(text.strip(), quote=False)}</pre>"


def render(text: str, font: str) -> str:
    """Текст ответа → HTML в выбранном из 15 стилей."""
    if font == "mono":
        return fmt_mono(text)
    if font == "pre":
        return fmt_pre(text)
    base = md_to_html(text, pretty=(font == "claude"))
    if font in ("claude", "rich") or font not in FONTS:
        return base
    simple = {"bold": "b", "italic": "i", "underline": "u", "quote": "blockquote", "spoiler": "tg-spoiler"}
    if font in simple:
        return map_runs(base, _wrap(simple[font]))
    if font == "bolditalic":
        return map_runs(map_runs(base, _wrap("b")), _wrap("i"))
    if font in UNI_MAPS:
        return map_runs(base, _uni(UNI_MAPS[font], UNI_CYR_TAG.get(font)))
    return base


def strip_html(h: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", h))


def split_text(text: str, limit: int) -> List[str]:
    """Разбить длинный ответ по строкам, сохраняя парность ``` между кусками."""
    if len(text) <= limit:
        return [text]
    lines: List[str] = []
    for ln in text.split("\n"):
        while len(ln) > limit:
            lines.append(ln[:limit])
            ln = ln[limit:]
        lines.append(ln)
    chunks: List[str] = []
    cur = ""
    in_fence = False
    lang = ""
    for ln in lines:
        if cur and len(cur) + len(ln) + 1 > limit:
            if in_fence:
                cur += "\n```"
            chunks.append(cur)
            cur = f"```{lang}" if in_fence else ""
        cur = (cur + "\n" + ln) if cur else ln
        if ln.strip().startswith("```"):
            in_fence = not in_fence
            lang = ln.strip()[3:].strip() if in_fence else ""
    if cur.strip():
        chunks.append(cur)
    return chunks


FONT_SAMPLE = ("# Пример шрифта\n"
               "Привет, мир! Hello, World 2026.\n\n"
               "**Жирный**, *курсив*, `код` и [ссылка](https://example.com).\n"
               "- первый пункт\n- второй пункт\n\n"
               "> цитата для проверки")


# ----------------------------------------------------------------------------
# Анимация статуса: «💭 Думаю… 7с» через sendMessageDraft (с запасным редактированием)
# ----------------------------------------------------------------------------

class Status:
    """Живой статус генерации. Режим draft — sendMessageDraft (как в эталонном боте),
    режим edit — запасной: одно сообщение, которое редактируется."""

    def __init__(self, app: "App", chat_id: int, provider: str, model: str) -> None:
        self.app = app
        self.chat_id = chat_id
        self.provider = provider
        self.model = model
        self.label = "Думаю"
        self.t0 = time.monotonic()
        self.mid: Optional[int] = None
        self.mode = "draft" if (app.draft_ok and chat_id > 0) else "edit"
        self.draft_id = random.randint(1, 2 ** 31 - 1)
        self.active = False
        self.task: Optional[asyncio.Task] = None
        self.every = 1.0 if chat_id > 0 else 2.5
        self._last_stream = 0.0
        self._i = 0

    def _head(self) -> str:
        sec = int(time.monotonic() - self.t0)
        dots = "." * (self._i % 3 + 1)
        return f"{self.label}{dots} · {sec}с\n{self.provider} · {self.model}"

    async def _draft(self, text: str, emoji: bool) -> bool:
        params: Dict[str, Any] = {"chat_id": self.chat_id, "draft_id": self.draft_id, "text": text[:4096]}
        if emoji:
            params["text"] = f"{PLACEHOLDER_EMOJI} {text}"[:4096]
            if self.app.emoji_ok and CUSTOM_EMOJI_ID:
                params["entities"] = [{"type": "custom_emoji", "offset": 0, "length": 2,
                                       "custom_emoji_id": CUSTOM_EMOJI_ID}]
        for _ in range(2):
            try:
                await self.app.api("sendMessageDraft", http_timeout=10, **params)
                return True
            except TgError as e:
                s = str(e).lower()
                if "entities" in params and any(k in s for k in ("emoji", "entit", "premium", "parse")):
                    self.app.emoji_ok = False
                    params.pop("entities", None)
                    continue
                if any(k in s for k in ("сеть", "таймаут", "too many", "retry", "timeout")):
                    return False  # временный сбой — драфты не отключаем
                print(f"ℹ️ sendMessageDraft недоступен ({e}) — перехожу на редактирование сообщения")
                self.app.draft_ok = False
                self.mode = "edit"
                return False
        return False

    async def _edit(self, text: str) -> None:
        try:
            if self.mid is None:
                r = await self.app.send(self.chat_id, text)
                self.mid = r["message_id"]
            else:
                await self.app.edit(self.chat_id, self.mid, text)
        except TgError:
            pass

    async def push(self, text: str, emoji: bool = True) -> None:
        if self.mode == "draft":
            if await self._draft(text, emoji):
                return
            if self.mode == "draft":
                return
        await self._edit(("⌛ " if emoji else "") + text)

    async def _loop(self) -> None:
        try:
            while self.active:
                await self.push(self._head())
                self._i += 1
                await asyncio.sleep(self.every)
        except asyncio.CancelledError:
            pass

    def start(self) -> None:
        if not self.active:
            self.active = True
            self.task = asyncio.create_task(self._loop())

    async def set_label(self, label: str) -> None:
        self.label = label
        self.t0 = time.monotonic()
        self._i = 0
        self.start()

    async def stop(self) -> None:
        self.active = False
        t, self.task = self.task, None
        if t is not None:
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)

    async def stream(self, acc: str) -> None:
        """Показать накопленный текст ответа (печать на лету)."""
        if self.active:
            await self.stop()
        gap = 0.3 if self.mode == "draft" else 1.6
        if self.chat_id < 0:
            gap = 3.0
        now = time.monotonic()
        if now - self._last_stream < gap:
            return
        if self.mode == "draft" and not acc.endswith((" ", "\n", ".", ",", "!", "?", "…", ":", ";")):
            return
        self._last_stream = now
        view = acc[-3900:]
        await self.push(view if self.mode == "draft" else view + " ▌", emoji=False)


# ----------------------------------------------------------------------------
# Бот
# ----------------------------------------------------------------------------

class App:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.state: Dict[str, Any] = load_json(STATE_PATH, {})
        self.session: Optional[aiohttp.ClientSession] = None
        self.me: Dict[str, Any] = {}
        self.dialogs: Dict[int, Dict[str, Any]] = {}
        self.models_cache: Dict[str, Tuple[float, List[Tuple[str, str]]]] = {}
        self._no_tools: set = set()
        self._kb_chats: set = set()
        self._tasks: set = set()
        self._locks: Dict[int, asyncio.Lock] = {}
        self.draft_ok = True     # sendMessageDraft доступен (отключится сам при ошибке)
        self.emoji_ok = True     # кастомное эмодзи принимается

    def spawn(self, coro: Any) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    # ---------- persistence ----------

    def save_config(self) -> None:
        _atomic_write(CONFIG_PATH, self.cfg)

    def save_state(self) -> None:
        _atomic_write(STATE_PATH, self.state)

    # ---------- telegram api ----------

    async def api(self, method: str, http_timeout: float = 60.0, **params: Any) -> Any:
        assert self.session is not None
        retried = params.pop("_retried", False)
        url = TG_API.format(token=self.cfg["bot_token"], method=method)
        status = 0
        try:
            async with self.session.post(
                url, json=params, timeout=aiohttp.ClientTimeout(total=http_timeout)
            ) as r:
                status = r.status
                try:
                    data = await r.json(content_type=None)
                except Exception:
                    raise TgError(f"HTTP {status}: не-JSON ответ")
        except asyncio.TimeoutError:
            raise TgError("таймаут запроса к Telegram")
        except aiohttp.ClientError as e:
            raise TgError(f"сеть: {e}")
        if not isinstance(data, dict) or not data.get("ok"):
            d = data if isinstance(data, dict) else {}
            ra = (d.get("parameters") or {}).get("retry_after")
            if d.get("error_code") == 429 and isinstance(ra, (int, float)) and ra <= 8 and not retried:
                await asyncio.sleep(ra + 0.3)  # flood-control: коротко ждём и повторяем
                return await self.api(method, http_timeout, _retried=True, **params)
            raise TgError(str(d.get("description", f"HTTP {status}: unknown error")))
        return data.get("result")

    async def api_with_retry(self, method: str, attempts: int = 3, delay: float = 4.0,
                             fatal_markers: Tuple[str, ...] = (), **params: Any) -> Any:
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
        kwargs: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text if parse_mode else text[:MAX_TG],
            "link_preview_options": {"is_disabled": True},
        }
        mark_kb = False
        if reply_markup:
            kwargs["reply_markup"] = reply_markup
        elif attach_kb and chat_id > 0 and chat_id not in self._kb_chats:
            kwargs["reply_markup"] = self.reply_kb(chat_id)
            mark_kb = True
        if parse_mode:
            kwargs["parse_mode"] = parse_mode
        res = await self.api("sendMessage", **kwargs)
        if mark_kb:
            self._kb_chats.add(chat_id)
        return res

    async def edit(self, chat_id: int, message_id: int, text: str, reply_markup: Optional[Dict] = None,
                   parse_mode: Optional[str] = None) -> None:
        kwargs: Dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text if parse_mode else text[:MAX_TG],
            "link_preview_options": {"is_disabled": True},
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

    async def put(self, chat_id: int, mid: Optional[int], text: str,
                  markup: Optional[Dict] = None, parse_mode: Optional[str] = None) -> Optional[int]:
        """Отредактировать сообщение mid, а если не вышло — отправить новое. Ошибки разметки пробрасываются."""
        if mid:
            try:
                await self.edit(chat_id, mid, text, markup, parse_mode)
                return mid
            except TgError as e:
                if "parse" in str(e).lower() or "entit" in str(e).lower():
                    raise
        r = await self.send(chat_id, text, markup, parse_mode)
        return (r or {}).get("message_id")

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
        st = self.state.get(str(chat_id), {})
        pid = st.get("provider") or self.cfg["default"].get("provider", "")
        mid = st.get("model") or self.cfg["default"].get("model", "")
        provider = self.cfg["providers"].get(pid)
        if not provider or not provider.get("enabled", True) or not mid:
            pid = self.cfg["default"].get("provider", "")
            mid = self.cfg["default"].get("model", "")
            provider = self.cfg["providers"].get(pid)
            if not provider or not mid:
                return None
        return pid, provider, mid

    def font_of(self, chat_id: int) -> str:
        f = self.state.get(str(chat_id), {}).get("font") or self.cfg.get("font", DEFAULT_FONT)
        return f if f in FONTS else DEFAULT_FONT

    # ---------- фото и файлы ----------

    async def download_tg_file(self, file_id: str, max_bytes: int = MAX_MEDIA_BYTES) -> Tuple[bytes, str]:
        f = await self.api("getFile", http_timeout=30, file_id=file_id)
        path = (f or {}).get("file_path") or ""
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                    "webp": "image/webp", "gif": "image/gif", "pdf": "application/pdf",
                    "mp3": "audio/mpeg", "ogg": "audio/ogg", "mp4": "video/mp4"}
        mime = mime_map.get(ext, "application/octet-stream")
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
        images: List[Dict[str, str]] = []
        files_text = ""
        notes: List[str] = []

        async def add_image(file_id: str, mime: str) -> None:
            try:
                data, m = await self.download_tg_file(file_id)
                mime = mime if mime.startswith("image/") else m
                images.append({"type": "image", "media_type": mime or m,
                               "data": base64.b64encode(data).decode()})
            except (ProviderError, TgError) as e:
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
                except (ProviderError, TgError) as e:
                    notes.append(str(e))
                else:
                    txt = data.decode("utf-8", "replace")
                    if txt.count("\ufffd") > len(txt) * 0.3:
                        notes.append(f"файл «{name}» двоичный — пришлите текстом")
                    else:
                        cut = " …(обрезано)" if len(txt) > MAX_FILE_TEXT else ""
                        files_text += f"📎 Файл «{name}»:\n\n{txt[:MAX_FILE_TEXT]}{cut}\n\n"
            elif ext == ".pdf" or "pdf" in mime:
                notes.append(f"📄 PDF «{name}» пока не поддерживается — пришлите текстом "
                             "или скриншотом нужных страниц")
            else:
                notes.append(f"файл «{name}» этого типа пока не поддерживается — "
                             "пришлите текстом или скриншотом")
        return images, files_text, notes

    # ---------- клавиатуры ----------

    @staticmethod
    def kb(rows: List[List[Tuple[str, str]]]) -> Dict[str, Any]:
        return {"inline_keyboard": [[{"text": t, "callback_data": d} for (t, d) in row] for row in rows]}

    def reply_kb(self, uid: int) -> Dict[str, Any]:
        rows = KB_ADMIN_ROWS if self.is_admin(uid) else KB_USER_ROWS
        return {
            "keyboard": [[{"text": t} for t in row] for row in rows],
            "resize_keyboard": True,
            "is_persistent": True,
            "input_field_placeholder": "Сообщение, фото или файл для ИИ…",
        }

    def kb_quick(self) -> Dict[str, Any]:
        """Кнопки под ответом бота."""
        return self.kb([[("🔄 Ещё раз", "regen"), ("🔤 Шрифт", "qfont")],
                        [("🤖 Сменить модель", "qmodel")]])

    def kb_font(self, cur: str) -> Dict[str, Any]:
        rows: List[List[Tuple[str, str]]] = []
        for i in range(0, len(FONT_ORDER), 2):
            row = []
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

    # ---------- вывод ответа ----------

    async def deliver(self, chat_id: int, text: str, mid: Optional[int] = None,
                      markup: bool = True) -> None:
        """Отправить итоговый ответ в выбранном шрифте: длинные ответы режутся на части
        с сохранением разметки; при ошибке разметки уходит простой текст."""
        font = self.font_of(chat_id)
        limit = 1700 if font in UNI_MAPS else 3500   # Unicode-буквы весят по 2 символа
        chunks = split_text(text, limit)
        cur_mid = mid
        for i, ch in enumerate(chunks):
            last = i == len(chunks) - 1
            mk = self.kb_quick() if (last and markup) else None
            h = render(ch, font)
            try:
                if len(h) > MAX_TG:
                    raise TgError("parse: слишком длинный html")
                cur_mid = await self.put(chat_id, cur_mid, h, mk, "HTML")
            except TgError as e:
                if "parse" not in str(e).lower() and "entit" not in str(e).lower() and "html" not in str(e).lower():
                    # не разметка (например, чат недоступен) — одна попытка простым текстом
                    pass
                try:
                    cur_mid = await self.put(chat_id, cur_mid, strip_html(h)[:MAX_TG], mk, None)
                except TgError as e2:
                    print(f"⚠️ Не удалось отправить ответ: {e2}")
                    return
            cur_mid = None  # остальные части — новыми сообщениями

    async def show_error(self, chat_id: int, status: Optional[Status], text: str) -> None:
        if status is not None:
            await status.stop()
        mk = self.kb([[("🔄 Повторить", "regen"), ("🤖 Другая модель", "qmodel")]])
        try:
            await self.put(chat_id, status.mid if status else None, text[:MAX_TG], mk, None)
        except TgError as e:
            print(f"⚠️ Не удалось показать ошибку: {e}")

    # ---------- генерация ----------

    async def typing_loop(self, chat_id: int, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.api("sendChatAction", http_timeout=10, chat_id=chat_id, action="typing")
            except Exception:  # noqa: BLE001
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                pass

    async def generate_and_send(self, msg: Dict[str, Any], user_text: str,
                                images: Optional[List[Dict[str, str]]] = None,
                                files_text: str = "", marker: Optional[str] = None) -> None:
        chat_id = msg["chat"]["id"]
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        async with lock:  # сообщения одного чата обрабатываются по очереди — история не путается
            await self._generate(chat_id, user_text, images or [], files_text, marker)

    async def _generate(self, chat_id: int, user_text: str, images: List[Dict[str, str]],
                        files_text: str, marker: Optional[str]) -> None:
        resolved = self.resolve(chat_id)
        if not resolved:
            await self.send(
                chat_id,
                "⚠️ Модель не выбрана.\n\n/model — выбрать модель\n/add (для админа) — подключить провайдера",
                attach_kb=True,
            )
            return
        pid, provider, model_id = resolved
        display = self.model_display(pid, model_id)

        st = self.chat_state(chat_id)
        history: List[Dict[str, Any]] = st.get("history", [])
        max_hist = int(self.cfg.get("max_history", 24) or 24)
        st["last_user"] = "" if (images or files_text) else user_text

        tools_on = bool(self.cfg.get("web_tools", True)) and f"{pid}|{model_id}" not in self._no_tools

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
            user_content: Any = [{"type": "text", "text": content_text}] + images if images else content_text
            return [{"role": "system", "content": sp}] + history + [{"role": "user", "content": user_content}]

        status = Status(self, chat_id, provider["name"], display)
        status.start()
        stop = asyncio.Event()
        typing_task = asyncio.create_task(self.typing_loop(chat_id, stop))
        full = ""

        async def consume(conv: List[Dict[str, Any]], use_tools: bool) -> Tuple[str, List[Dict[str, str]]]:
            acc, calls = "", []
            async for kind, piece in stream_chat(self.session, provider, model_id, conv,  # type: ignore[arg-type]
                                                 tools=use_tools):
                if kind == "tools":
                    calls = piece
                    continue
                acc += piece
                await status.stream(acc)
            return acc, calls

        try:
            for attempt in (0, 1):
                conv = build_messages(tools_on)
                full = ""
                try:
                    if tools_on:
                        for _round in range(MAX_TOOL_ROUNDS):
                            acc, calls = await consume(conv, True)
                            full = acc
                            if not calls:
                                break
                            conv.append({"role": "assistant", "content": acc, "tool_calls": calls})
                            for c in calls:
                                await status.set_label(tool_label(c))
                                result = await run_tool(self.session, c["name"], c["args"])
                                conv.append({"role": "tool", "tool_call_id": c["id"],
                                             "name": c["name"], "content": result})
                            await status.set_label("Думаю")
                        else:
                            # все раунды ушли на инструменты — просим финальный ответ без них
                            acc, _ = await consume(flatten_tools(conv), False)
                            full = acc or full
                    else:
                        full, _ = await consume(conv, False)
                    break
                except ProviderError as e:
                    if attempt == 0 and tools_on and _is_tools_error(str(e)):
                        self._no_tools.add(f"{pid}|{model_id}")
                        tools_on = False
                        await status.set_label("Продолжаю без поиска")
                        await asyncio.sleep(1)
                        continue
                    raise
            await status.stop()

            if not full.strip():
                await self.show_error(chat_id, status, "⚠️ Пустой ответ от модели. Попробуйте ещё раз "
                                                       "или смените модель.")
                return

            history = history + [{"role": "user", "content": marker or user_text},
                                 {"role": "assistant", "content": full}]
            st["history"] = history[-max_hist * 2:]
            self.save_state()
            await self.deliver(chat_id, full, status.mid)
        except ProviderError as e:
            err = f"⚠️ Ошибка провайдера ({provider['name']}):\n{e}"
            if images and any(w in str(e).lower() for w in ("image", "vision", "медиа", "мультимод", "input type")):
                err += "\n\n💡 Похоже, эта модель не принимает изображения — выберите другую: /model"
            await self.show_error(chat_id, status, err)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            await self.show_error(chat_id, status, "⏱ Провайдер отвечал слишком долго и соединение оборвалось.\n"
                                                   "Попробуйте ещё раз или выберите другую модель.")
        except aiohttp.ClientError as e:
            await self.show_error(chat_id, status, f"🔌 Соединение с провайдером оборвалось: {e}\n"
                                                   "Обычно помогает повторить запрос через минуту.")
        except Exception as e:  # noqa: BLE001
            await self.show_error(chat_id, status, f"⚠️ Непредвиденная ошибка:\n{e}")
        finally:
            await status.stop()
            stop.set()
            typing_task.cancel()
            await asyncio.gather(typing_task, return_exceptions=True)

    # ---------- обработка сообщений ----------

    async def on_message(self, msg: Dict[str, Any]) -> None:
        chat_id = msg["chat"]["id"]
        uid = (msg.get("from") or {}).get("id", 0)
        text = (msg.get("text") or msg.get("caption") or "").strip()
        private = msg["chat"].get("type") == "private"
        if text in MENU_BTN:
            text = MENU_BTN[text]
        has_media = bool(msg.get("photo") or msg.get("document"))

        # в группах отвечаем только на команды, упоминания и ответы боту
        if not private:
            uname = (self.me.get("username") or "").lower()
            mentioned = bool(uname) and f"@{uname}" in text.lower()
            reply_to_me = ((msg.get("reply_to_message") or {}).get("from") or {}).get("id") == self.me.get("id")
            if not (text.startswith("/") or mentioned or reply_to_me):
                return
            if uname:
                text = re.sub(rf"@{re.escape(uname)}", "", text, flags=re.I).strip()

        # первый пользователь в личке становится владельцем
        if not self.cfg.get("admins") and private and uid:
            self.cfg["admins"] = [uid]
            self.save_config()
            self._kb_chats.discard(chat_id)
            await self.send(chat_id, "✅ Вы назначены администратором бота.\n\n"
                                     "Начните с /add — подключите провайдера, затем /model.\n\n" + HELP_HTML,
                            parse_mode="HTML", attach_kb=True)
            return

        if not text and not has_media:
            hints = [("voice", "🎧 Голосовые и аудио я пока не понимаю — напишите текстом или пришлите фото/файл."),
                     ("audio", "🎧 Голосовые и аудио я пока не понимаю — напишите текстом или пришлите фото/файл."),
                     ("video", "🎬 Видео я пока не понимаю — пришлите скриншот или опишите вопрос текстом."),
                     ("video_note", "🎬 Видео я пока не понимаю — пришлите скриншот или опишите вопрос текстом."),
                     ("sticker", "🙂 Стикеры не понимаю — напишите текстом, пришлите фото или файл.")]
            for key, hint in hints:
                if msg.get(key):
                    await self.send(chat_id, hint, attach_kb=True)
                    return
            await self.send(chat_id, "Пока я понимаю текст, фото и файлы 🙏", attach_kb=True)
            return

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
            if await self.dispatch_command(msg, cmd, args):
                return
            await self.send(chat_id, f"Неизвестная команда /{cmd}. Смотрите /help", attach_kb=True)
            return

        if not self.can_chat(uid):
            await self.send(chat_id, "⛔ Общение с ботом ограничено администраторами.")
            return
        if not text:
            return
        await self.generate_and_send(msg, text)

    async def send_font_menu(self, chat_id: int) -> None:
        cur = self.font_of(chat_id)
        await self.send(
            chat_id,
            f"🔤 Шрифт ответов ИИ — сейчас: {FONTS[cur]}\n\n"
            "Выберите из 15 стилей — после выбора пришлю пример.\n"
            "ℹ️ Unicode-стили (рукопись, готика, двойной, гротеск, широкий) меняют латиницу; "
            "русский текст в них оформляется жирным/курсивом/подчёркиванием.",
            self.kb_font(cur), attach_kb=True)

    async def dispatch_command(self, msg: Dict[str, Any], cmd: str, args: str) -> bool:
        chat_id = msg["chat"]["id"]
        uid = (msg.get("from") or {}).get("id", 0)

        async def deny() -> bool:
            await self.send(chat_id, "⛔ Команда только для администраторов.")
            return True

        if cmd in ("start", "help"):
            self._kb_chats.discard(chat_id)  # на /start всегда обновляем клавиатуру
            await self.send(chat_id, HELP_HTML, parse_mode="HTML", attach_kb=True)
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
            st = self.chat_state(chat_id)
            kb = self.kb_providers("pv")
            if st.get("provider") and st.get("model"):
                kb["inline_keyboard"] = [[{"text": "⟲ Сбросить на модель по умолчанию",
                                           "callback_data": "seldef"}]] + kb["inline_keyboard"]
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
                    f"📌 Модель {src}:\n{provider['name']} → {self.model_display(pid, mid)}\n\n"
                    f"🔤 Шрифт: {FONTS[self.font_of(chat_id)]}\n"
                    f"🌐 Веб-поиск: {'вкл' if self.cfg.get('web_tools', True) else 'выкл'}\n"
                    f"🧹 Память диалога: {len(st.get('history', [])) // 2} сообщ.",
                    attach_kb=True)
            else:
                await self.send(chat_id, "⚠️ Модель не выбрана. Используйте /model", attach_kb=True)
            return True

        if cmd == "clear":
            st = self.chat_state(chat_id)
            st["history"] = []
            st["last_user"] = ""
            self.save_state()
            await self.send(chat_id, "🧹 История диалога очищена.", attach_kb=True)
            return True

        if cmd == "web":
            if not self.is_admin(uid):
                return await deny()
            on = not self.cfg.get("web_tools", True)
            self.cfg["web_tools"] = on
            self.save_config()
            if on:
                self._no_tools.clear()
                await self.send(chat_id, "🌐 Веб-инструменты ВКЛЮЧЕНЫ: модель может искать в интернете и "
                                         "открывать страницы. Провайдеры без tools отключаются автоматически.",
                                attach_kb=True)
            else:
                await self.send(chat_id, "🌐 Веб-инструменты ВЫКЛЮЧЕНЫ — модель отвечает из своих знаний.",
                                attach_kb=True)
            return True

        if cmd == "font":
            a = args.strip().lower().replace(" ", "")
            a = FONT_ALIASES.get(a, a)
            if a in FONTS:
                self.chat_state(chat_id)["font"] = a
                self.save_state()
                await self.send_font_sample(chat_id, a)
            await self.send_font_menu(chat_id)
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
            await self.send(chat_id, "\n".join(lines)[:MAX_TG] or "В наборе нет стикеров с custom_emoji_id.",
                            attach_kb=True)
            return True

        if cmd == "providers":
            lines = ["🧩 Подключённые провайдеры:"]
            if not self.cfg["providers"]:
                lines.append("— пока нет. Админ может добавить через /add")
            for pid, p in self.cfg["providers"].items():
                cached = self.models_cache.get(pid)
                n = str(len(cached[1])) + " мод." if cached else "модели не загружены"
                flag = "" if p.get("enabled", True) else " (выкл)"
                lines.append(
                    f"• {pid} — {p['name']} [{PROTOCOL_TITLES.get(p['protocol'], p['protocol'])}]{flag}\n"
                    f"  {p['base_url']} · ключ: {self.mask_key(p.get('api_key', ''))} · {n}")
            await self.send(chat_id, "\n".join(lines)[:MAX_TG], attach_kb=True)
            return True

        # ----- дальше только для админов -----
        admin_cmds = ("add", "del", "default", "key", "url", "sys", "lock")
        if cmd in admin_cmds and not self.is_admin(uid):
            return await deny()

        if cmd == "add":
            await self.wizard_start(chat_id)
            return True

        if cmd == "del":
            rows = [[(f"{p['name']} ({pid})", f"del:{pid}")] for pid, p in self.cfg["providers"].items()] \
                or [[("нет провайдеров", "noop")]]
            await self.send(chat_id, "🗑 Какого провайдера удалить?", self.kb(rows), attach_kb=True)
            return True

        if cmd == "default":
            await self.send(chat_id, "⭐ Провайдер по умолчанию:", self.kb_providers("defpv", "noop"),
                            attach_kb=True)
            return True

        if cmd == "key":
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
            parts = args.split(maxsplit=1)
            if len(parts) < 2 or parts[0] not in self.cfg["providers"]:
                await self.send(chat_id, "Формат: /url <id> <baseURL>\nНапример: /url p1 https://openrouter.ai/api/v1",
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
            if not args:
                cur = self.cfg.get("system_prompt", "")
                if cur:
                    await self.send(chat_id, f"💬 Текущий системный промпт:\n\n{cur[:1500]}\n\n"
                                             "Изменить: /sys <текст> · Выключить: /sys off", attach_kb=True)
                else:
                    await self.send(chat_id, "💬 Системный промпт выключен.\n\nЗадать: /sys <текст>",
                                    attach_kb=True)
                return True
            if args.lower() in ("off", "выкл", "выключить"):
                self.cfg["system_prompt"] = ""
                self.save_config()
                await self.send(chat_id, "Системный промпт выключен.", attach_kb=True)
            else:
                self.cfg["system_prompt"] = args
                self.save_config()
                await self.send(chat_id, f"✅ Системный промпт сохранён:\n{args[:500]}", attach_kb=True)
            return True

        if cmd == "lock":
            self.cfg["admins_only"] = not self.cfg.get("admins_only", False)
            self.save_config()
            state = ("включён — писать могут только администраторы" if self.cfg["admins_only"]
                     else "выключен — писать могут все")
            await self.send(chat_id, f"🔒 Режим «только администраторы» {state}.", attach_kb=True)
            return True

        return False

    async def send_font_sample(self, chat_id: int, font: str) -> None:
        """Пример текста в выбранном шрифте."""
        h = render(FONT_SAMPLE, font)
        try:
            await self.send(chat_id, h, parse_mode="HTML")
        except TgError:
            await self.send(chat_id, strip_html(h))

    # ---------- мастер добавления провайдера ----------

    async def wizard_start(self, chat_id: int) -> None:
        self.dialogs[chat_id] = {"action": "add", "step": "preset", "data": {}}
        rows: List[List[Tuple[str, str]]] = []
        items = list(PRESETS.items())
        for i in range(0, len(items), 2):
            rows.append([(p["name"], f"pre:{pid}") for pid, p in items[i:i + 2]])
        await self.send(
            chat_id,
            "🧩 Подключение провайдера.\n"
            "«🔗 Своя ссылка (авто)» — просто URL: бот сам определит тип API и загрузит список моделей. "
            "Ключ не обязателен — подойдёт самохостинг (Ollama / LM Studio / vLLM).\n"
            "Остальные — пресеты: выбрал → вставил только API-ключ:",
            self.kb(rows))

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
                note = f"🔌 Тип API определён: {PROTOCOL_TITLES.get(data['protocol'], data['protocol'])}\n"
            dlg["step"] = "key"
            await self.send(chat_id, f"{note}🔑 Теперь пришлите API-ключ (или /skip, если сервер без ключа):")
            return

        if step == "key":
            key = "" if text.strip().lower().lstrip("/") == "skip" else clean_secret(text)
            data["api_key"] = key
            pid = self.next_pid()
            name = data["name"]
            base = name
            n = 2
            while any(p["name"] == name for p in self.cfg["providers"].values()):
                name = f"{base} {n}"
                n += 1
            self.cfg["providers"][pid] = {"name": name, "protocol": data["protocol"],
                                          "base_url": data["base_url"], "api_key": key, "enabled": True}
            self.save_config()
            self.dialogs.pop(chat_id, None)
            await self.send(chat_id, f"✅ Провайдер «{name}» сохранён (id: {pid}).\n"
                                     "🔍 Проверяю ключ и загружаю список моделей…")
            try:
                models = await self.get_models(pid, force=True)
                if models:
                    await self.send(chat_id, f"📦 Доступно моделей: {len(models)} — ключ работает.\n\n"
                                             "Теперь: /model — выбрать модель или /default — модель по умолчанию.")
                else:
                    await self.send(chat_id, "⚠️ Список моделей пуст. Проверьте ключ (/key) и адрес (/url).")
            except ProviderError as e:
                await self.send(chat_id, f"⚠️ Не удалось получить модели:\n{e}\n\n"
                                         f"Провайдер сохранён. Ключ: /key {pid} <ключ>\nАдрес: /url {pid} <baseURL>")
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
        try:
            await self.handle_callback(cq, chat_id, uid, data)
        except TgError as e:
            try:
                await self.send(chat_id, f"⚠️ Кнопка не сработала: {e}\n\nОткройте меню заново: /model или /help")
            except TgError:
                pass

    async def handle_callback(self, cq: Dict[str, Any], chat_id: int, uid: int, data: str) -> None:
        msg = cq.get("message") or {}
        mid_ = msg.get("message_id", 0)

        # админские действия — проверка прав (callback_data можно подделать)
        if data.startswith(("pre:", "del:", "defpv:", "defsel", "defbk", "rm:defsel")) and not self.is_admin(uid):
            await self.send(chat_id, "⛔ Только для администраторов.")
            return
        if data.startswith("sel") or data.startswith("pv:") or data in ("bk", "seldef"):
            if not self.can_chat(uid):
                return

        # --- мастер /add ---
        if data.startswith("pre:"):
            preset_id = data[4:]
            preset = PRESETS.get(preset_id)
            if not preset:
                return
            dlg = self.dialogs.setdefault(chat_id, {"action": "add", "data": {}})
            dlg["action"] = "add"
            dlg["data"] = {"name": preset["name"], "protocol": preset["protocol"],
                           "base_url": preset["base_url"], "preset_id": preset_id,
                           "auto": preset_id == "custom-auto"}
            if preset["base_url"]:
                dlg["step"] = "key"
                docs = f"\nКлюч: {preset['docs']}" if preset.get("docs") else ""
                await self.edit(chat_id, mid_, f"✅ {preset['name']} — {preset['hint']}{docs}\n\n"
                                               "🔑 Пришлите API-ключ (или /skip):")
            else:
                dlg["step"] = "url"
                if preset_id == "custom-auto":
                    txt = ("Пришлите просто ссылку на API (baseURL), например:\nhttps://api.example.com/v1\n\n"
                           "Я сам определю тип API (OpenAI / Anthropic / Gemini), проверю ключ и загружу "
                           "список моделей. Ключ можно пропустить (/skip) — для Ollama, LM Studio и т.п.")
                else:
                    txt = "Пришлите baseURL, например:\nhttps://api.example.com/v1"
                await self.edit(chat_id, mid_, f"✅ {preset['name']}\n{txt}")
            return

        # --- /model ---
        if data == "seldef":
            st = self.chat_state(chat_id)
            st["provider"] = ""
            st["model"] = ""
            self.save_state()
            r = self.resolve(chat_id)
            if r:
                await self.edit(chat_id, mid_, "✅ Будет использоваться модель по умолчанию:\n"
                                               f"{r[1]['name']} → {self.model_display(r[0], r[2])}")
            else:
                await self.edit(chat_id, mid_, "Модель по умолчанию не задана. Админ может задать: /default")
            return

        if data.startswith("pv:"):
            pid = data[3:]
            provider = self.cfg["providers"].get(pid)
            if not provider:
                await self.edit(chat_id, mid_, "⚠️ Провайдер удалён. Откройте /model заново.")
                return
            kb, err = await self.kb_models(pid, 0, "sel", "bk")
            if err:
                await self.edit(chat_id, mid_, f"⚠️ {err}")
                return
            await self.edit(chat_id, mid_, f"🤖 {provider['name']} — выберите модель:", kb)
            return

        if data.startswith("sel:"):
            _, pid, idx_s = data.split(":", 2)
            if pid not in self.cfg["providers"]:
                await self.edit(chat_id, mid_, "⚠️ Провайдер удалён. Откройте /model заново.")
                return
            item = self.idx_to_model(pid, int(idx_s))
            if not item:
                await self.edit(chat_id, mid_, "⚠️ Список устарел, откройте /model заново.")
                return
            st = self.chat_state(chat_id)
            st["provider"] = pid
            st["model"] = item[0]
            self.save_state()
            await self.edit(chat_id, mid_, f"✅ Модель выбрана:\n{self.cfg['providers'][pid]['name']} → "
                                           f"{self.model_display(pid, item[0])}\n\nТеперь просто пишите сообщения.")
            return

        if data == "bk":
            await self.edit(chat_id, mid_, "🧩 Выберите провайдера:", self.kb_providers("pv"))
            return

        # --- кнопки под ответом (отправляем НОВЫЕ сообщения — ответ остаётся на месте) ---
        if data == "qmodel":
            st = self.chat_state(chat_id)
            kb = self.kb_providers("pv")
            if st.get("provider") and st.get("model"):
                kb["inline_keyboard"] = [[{"text": "⟲ Сбросить на модель по умолчанию",
                                           "callback_data": "seldef"}]] + kb["inline_keyboard"]
            await self.send(chat_id, "🧩 Выберите провайдера:", kb)
            return

        if data == "qfont":
            await self.send_font_menu(chat_id)
            return

        if data == "regen":
            if not self.can_chat(uid):
                return
            st = self.chat_state(chat_id)
            txt = st.get("last_user") or ""
            if not txt:
                await self.send(chat_id, "🔄 Повторить можно только текстовый запрос. Отправьте сообщение заново.")
                return
            h = st.get("history", [])
            if len(h) >= 2 and h[-2].get("role") == "user" and h[-1].get("role") == "assistant" \
                    and h[-2].get("content") == txt:
                st["history"] = h[:-2]
            fake = {"chat": {"id": chat_id, "type": "private" if chat_id > 0 else "group"},
                    "from": {"id": uid}}
            await self.generate_and_send(fake, txt)
            return

        # --- выбор шрифта (15 стилей) ---
        if data.startswith("font:"):
            f = data[5:]
            if f in FONTS:
                self.chat_state(chat_id)["font"] = f
                self.save_state()
                await self.edit(chat_id, mid_, f"🔤 Шрифт ответов ИИ: {FONTS[f]}\n"
                                               "Пример ниже — следующие ответы придут в этом стиле.",
                                self.kb_font(f))
                await self.send_font_sample(chat_id, f)
            return

        # --- обновление списка моделей (🔄) ---
        m = re.match(r"^rm:(sel|defsel):([^:]+)$", data)
        if m:
            pick, pid = m.group(1), m.group(2)
            back = "bk" if pick == "sel" else "defbk"
            provider = self.cfg["providers"].get(pid)
            if not provider:
                await self.edit(chat_id, mid_, "⚠️ Провайдер удалён. Откройте /model заново.")
                return
            try:
                await self.get_models(pid, force=True)
            except ProviderError as e:
                await self.edit(chat_id, mid_, f"⚠️ Не удалось обновить список моделей:\n{e}")
                return
            kb, err = await self.kb_models(pid, 0, pick, back)
            if err:
                await self.edit(chat_id, mid_, f"⚠️ {err}")
                return
            title = "🤖" if pick == "sel" else "⭐"
            n = len(self.models_cache.get(pid, (0, []))[1])
            await self.edit(chat_id, mid_, f"{title} {provider['name']} — список моделей обновлён ({n} шт.):", kb)
            return

        # --- /default ---
        if data.startswith("defpv:"):
            pid = data[6:]
            if pid not in self.cfg["providers"]:
                await self.edit(chat_id, mid_, "⚠️ Провайдер удалён.")
                return
            kb, err = await self.kb_models(pid, 0, "defsel", "defbk")
            if err:
                await self.edit(chat_id, mid_, f"⚠️ {err}")
                return
            await self.edit(chat_id, mid_, f"⭐ {self.cfg['providers'][pid]['name']} — модель по умолчанию:", kb)
            return

        if data == "defbk":
            await self.edit(chat_id, mid_, "⭐ Провайдер по умолчанию:", self.kb_providers("defpv", "noop"))
            return

        if data.startswith("defsel:"):
            _, pid, idx_s = data.split(":", 2)
            item = self.idx_to_model(pid, int(idx_s))
            if not item or pid not in self.cfg["providers"]:
                await self.edit(chat_id, mid_, "⚠️ Список устарел, откройте /default заново.")
                return
            self.cfg["default"] = {"provider": pid, "model": item[0]}
            self.save_config()
            await self.edit(chat_id, mid_, f"⭐ Модель по умолчанию:\n{self.cfg['providers'][pid]['name']} → "
                                           f"{self.model_display(pid, item[0])}")
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
                    if isinstance(st, dict) and st.get("provider") == pid:
                        st["provider"] = ""
                        st["model"] = ""
                self.save_config()
                self.save_state()
                await self.edit(chat_id, mid_, f"🗑 Провайдер «{p['name']}» удалён.")
            return

        # --- пагинация ---
        m = re.match(r"^(sel|defsel)pg:([^:]+):(\d+)$", data)
        if m:
            prefix, pid, page = m.group(1), m.group(2), int(m.group(3))
            back = "bk" if prefix == "sel" else "defbk"
            kb, err = await self.kb_models(pid, page, prefix, back)
            if err:
                await self.edit(chat_id, mid_, f"⚠️ {err}")
                return
            try:
                await self.api("editMessageReplyMarkup", chat_id=chat_id, message_id=mid_, reply_markup=kb)
            except TgError:
                pass
            return

    # ---------- апдейты ----------

    async def handle_update(self, up: Dict[str, Any]) -> None:
        try:
            if "callback_query" in up:
                await self.on_callback(up["callback_query"])
            elif "message" in up:
                await self.on_message(up["message"])
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Ошибка обработки апдейта: {type(e).__name__}: {e}")

    # ---------- webhook-режим ----------

    async def run_webhook(self, base_url: str, port: int) -> None:
        if not base_url.startswith(("http://", "https://")):
            sys.exit("❌ WEBHOOK_URL / RENDER_EXTERNAL_URL должен начинаться с http(s)://")
        timeout = aiohttp.ClientTimeout(total=None)
        hook_path = "hook-" + hashlib.sha256(self.cfg["bot_token"].encode()).hexdigest()[:16]
        secret = secrets.token_hex(24)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.session = session
            try:
                me = await self.api_with_retry("getMe", attempts=3, delay=3.0,
                                               fatal_markers=("401", "unauthorized", "404", "not found"))
            except TgError as e:
                sys.exit(f"❌ Telegram отклонил токен: {e}\n💡 {tg_hint(str(e))}")
            self.me = me or {}

            async def telegram_update(request: web.Request) -> web.Response:
                if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != secret:
                    return web.Response(status=403, text="forbidden")
                try:
                    up = await request.json()
                except Exception:
                    return web.Response(status=400, text="bad json")
                self.spawn(self.handle_update(up))
                return web.Response(text="ok")

            async def health(request: web.Request) -> web.Response:
                return web.Response(text=f"🤖 Бот работает: @{self.me.get('username', '?')}",
                                    content_type="text/plain", charset="utf-8")

            webapp = web.Application()
            webapp.router.add_get("/", health)
            webapp.router.add_get("/healthz", health)
            webapp.router.add_post("/" + hook_path, telegram_update)

            runner = web.AppRunner(webapp, access_log=None)
            await runner.setup()
            await web.TCPSite(runner, "0.0.0.0", port).start()

            try:
                await self.api_with_retry(
                    "setWebhook", attempts=4, delay=6.0, fatal_markers=("400", "bad request"),
                    http_timeout=30, url=f"{base_url}/{hook_path}", secret_token=secret,
                    allowed_updates=["message", "callback_query"], max_connections=40,
                    drop_pending_updates=False)
            except TgError as e:
                await runner.cleanup()
                sys.exit(f"❌ Не удалось зарегистрировать webhook: {e}\n💡 {tg_hint(str(e))}")

            uname = self.me.get("username", "?")
            print("=" * 60)
            print(f"✅ Бот @{uname} запущен (webhook): https://t.me/{uname}")
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
            print("⏹ Останавливаюсь…")
            await runner.cleanup()

    # ---------- polling ----------

    async def run(self) -> None:
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.session = session
            try:
                me = await self.api("getMe")
            except TgError as e:
                sys.exit(f"❌ Telegram отклонил токен: {e}\n💡 {tg_hint(str(e))}")
            self.me = me or {}
            username = self.me.get("username", "?")
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
                    updates = await self.api("getUpdates", http_timeout=45, offset=offset, timeout=25,
                                             allowed_updates=["message", "callback_query"])
                    for up in updates or []:
                        offset = up["update_id"] + 1
                        self.spawn(self.handle_update(up))
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
                        print("❌ Неверный токен! Исправьте BOT_TOKEN / config.json и перезапустите.")
                        await asyncio.sleep(10)
                    else:
                        print(f"⚠️ Telegram: {desc}")
                        await asyncio.sleep(3)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    print(f"⚠️ Сеть: {e} — переподключаюсь…")
                    await asyncio.sleep(3)


def main() -> None:
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
