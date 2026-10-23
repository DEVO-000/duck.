"""
Telegram-бот на DeepSeek: окно «Думаю Nс», доступ в интернет,
история диалога, красивое HTML-форматирование.

Токены берутся ТОЛЬКО из переменных окружения:
    TELEGRAM_TOKEN
    DEEPSEEK_API_KEY

Установка:
    pip install -U aiogram openai aiohttp ddgs

Запуск локально:
    export TELEGRAM_TOKEN=...
    export DEEPSEEK_API_KEY=...
    python main.py

На Render: добавить обе переменные в Environment → Save.
"""
import asyncio
import html
import json
import os
import random
import re
import sys
import time
from datetime import datetime

import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, MessageEntity
from aiogram.utils.keyboard import InlineKeyboardBuilder
from openai import AsyncOpenAI

try:
    from ddgs import DDGS
except ImportError:  # старое имя пакета
    from duckduckgo_search import DDGS


# ====================== ТОКЕНЫ (только из окружения) ======================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()

if not TELEGRAM_TOKEN or ":" not in TELEGRAM_TOKEN:
    print("❌ TELEGRAM_TOKEN не задан или имеет неверный формат.", file=sys.stderr)
    print("   Укажите переменную окружения TELEGRAM_TOKEN (токен от @BotFather).", file=sys.stderr)
    sys.exit(1)

if not DEEPSEEK_API_KEY:
    print("❌ DEEPSEEK_API_KEY не задан.", file=sys.stderr)
    print("   Укажите переменную окружения DEEPSEEK_API_KEY.", file=sys.stderr)
    sys.exit(1)

deepseek = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com", timeout=180.0)
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# ====================== НАСТРОЙКИ ======================
SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
STATUS_LINES = 5
STATUS_WIDTH = 38
STATUS_INTERVAL = 0.45
ANSWER_INTERVAL = 0.35
HISTORY_LIMIT = 12
MAX_SEARCH_ROUNDS = 3

REASONING: dict[int, bool] = {}
SEARCH: dict[int, bool] = {}
HISTORY: dict[int, list[dict]] = {}
LOCKS: dict[int, asyncio.Lock] = {}


# ====================== УТИЛИТЫ ======================
def utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        while len(w) > width:
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(w[:width])
            w = w[width:]
        if len(cur) + len(w) + (1 if cur else 0) <= width:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ====================== ОКНО «ДУМАЮ» ======================
def render_status(title: str, secs: int, frame: int, text: str) -> tuple[str, list[MessageEntity]]:
    header = f"{SPINNER[frame % len(SPINNER)]} {title} {secs}с"
    lines: list[str] = []
    for ln in text.splitlines():
        if ln.strip():
            lines.extend(wrap(ln.strip(), STATUS_WIDTH))
    shown = lines[-STATUS_LINES:]
    shown += ["\u00A0"] * (STATUS_LINES - len(shown))
    body = "\n".join(shown)
    full = f"{header}\n{body}"
    entities = [
        MessageEntity(type="bold", offset=0, length=utf16_len(header)),
        MessageEntity(type="blockquote", offset=utf16_len(header) + 1, length=utf16_len(body)),
    ]
    return full, entities


# ====================== ИНТЕРНЕТ ======================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Поиск в интернете. Используй для новостей, курсов, погоды, фактов после 2024 года, "
                           "цен, версий ПО и всего, что могло измениться.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Короткий поисковый запрос"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Открыть страницу по ссылке и получить её текст (для деталей из результатов поиска).",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


def _ddgs_search(query: str, n: int = 6) -> list[dict]:
    with DDGS() as d:
        return list(d.text(query, region="ru-ru", max_results=n))


async def web_search(query: str) -> list[dict]:
    try:
        raw = await asyncio.to_thread(_ddgs_search, query)
    except Exception as e:
        print(f"search error: {e}")
        return []
    return [
        {"title": r.get("title", ""), "url": r.get("href") or r.get("url", ""), "snippet": r.get("body", "")}
        for r in raw
    ]


async def open_url(url: str) -> str:
    try:
        async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status != 200:
                    return f"Ошибка {r.status}"
                raw = await r.text(errors="ignore")
    except Exception as e:
        return f"Не удалось открыть: {e}"
    raw = re.sub(r"(?is)<(script|style|noscript|svg|header|footer|nav)[^>]*>.*?</\1>", " ", raw)
    text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))
    return re.sub(r"\s+", " ", text).strip()[:3500]


# ====================== HTML-ФОРМАТИРОВАНИЕ ======================
def format_html(text: str) -> str:
    stash: list[str] = []

    def put(block: str) -> str:
        stash.append(block)
        return f"\x00P{len(stash) - 1}\x00"

    def code_block(m):
        lang, body = (m.group(1) or "").strip(), m.group(2)
        safe = html.escape(body.rstrip("\n"))
        if lang:
            return put(f'<pre><code class="language-{html.escape(lang)}">{safe}</code></pre>')
        return put(f"<pre>{safe}</pre>")

    text = re.sub(r"```(\w*)\n([\s\S]*?)```", code_block, text)
    text = re.sub(r"```(\w*)\n?([\s\S]*)$", code_block, text)
    text = re.sub(r"`([^`\n]+)`", lambda m: put(f"<code>{html.escape(m.group(1))}</code>"), text)

    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\*\w])\*(?!\*)(.+?)(?<!\*)\*(?![\*\w])", r"<i>\1</i>", text)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^&gt;\s?(.+)$", r"<blockquote>\1</blockquote>", text, flags=re.MULTILINE)
    text = text.replace("</blockquote>\n<blockquote>", "\n")

    for i, block in enumerate(stash):
        text = text.replace(f"\x00P{i}\x00", block)
    return text


def split_html(text: str, limit: int = 3900) -> list[str]:
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    parts.append(text)
    return parts


# ====================== КЛАВИАТУРА ======================
def keyboard(uid: int) -> InlineKeyboardMarkup:
    r = REASONING.get(uid, False)
    s = SEARCH.get(uid, True)
    b = InlineKeyboardBuilder()
    b.button(text=f"🧠 Думать: {'ВКЛ' if r else 'ВЫКЛ'}", callback_data="toggle:reasoning")
    b.button(text=f"🌐 Интернет: {'ВКЛ' if s else 'ВЫКЛ'}", callback_data="toggle:search")
    b.adjust(2)
    return b.as_markup()


@dp.callback_query(lambda c: c.data and c.data.startswith("toggle:"))
async def on_toggle(cb: types.CallbackQuery):
    uid = cb.from_user.id
    what = cb.data.split(":")[1]
    if what == "reasoning":
        REASONING[uid] = not REASONING.get(uid, False)
        note = "🧠 Режим размышлений " + ("включён" if REASONING[uid] else "выключен")
    else:
        SEARCH[uid] = not SEARCH.get(uid, True)
        note = "🌐 Интернет " + ("включён" if SEARCH[uid] else "выключен")
    try:
        await cb.message.edit_reply_markup(reply_markup=keyboard(uid))
    except Exception:
        pass
    await cb.answer(note)


# ====================== КОМАНДЫ ======================
@dp.message(Command("start"))
async def cmd_start(m: types.Message):
    await m.answer(
        "👋 <b>Привет!</b> Я отвечаю на вопросы, пишу код и умею искать в интернете.\n\n"
        "🧠 <b>Думать</b> — показывает ход рассуждений, как в Mira\n"
        "🌐 <b>Интернет</b> — свежие данные из сети\n"
        "🧹 /clear — забыть историю диалога",
        parse_mode="HTML",
        reply_markup=keyboard(m.from_user.id),
    )


@dp.message(Command("clear"))
async def cmd_clear(m: types.Message):
    HISTORY.pop(m.from_user.id, None)
    await m.answer("🧹 История очищена.", reply_markup=keyboard(m.from_user.id))


# ====================== ПОИСКОВАЯ ФАЗА ======================
async def run_search_phase(history: list[dict], user_text: str, ui: dict) -> tuple[str, list[dict]]:
    sources: list[dict] = []
    ctx_parts: list[str] = []
    today = datetime.now().strftime("%d.%m.%Y")
    messages = [
        {
            "role": "system",
            "content": (
                f"Сегодня {today}. Ты решаешь, нужен ли поиск в интернете для ответа. "
                "Если вопрос про актуальные события, цены, курсы, погоду, новости, свежие версии, "
                "конкретных людей/организации или любые факты, которые могли измениться — вызови web_search "
                "(можно несколько раз с разными запросами) и при необходимости open_url. "
                "Если поиск не нужен — ответь одним словом: OK. Сам ответ на вопрос НЕ пиши."
            ),
        },
        *history[-6:],
        {"role": "user", "content": user_text},
    ]

    for _ in range(MAX_SEARCH_ROUNDS):
        try:
            resp = await deepseek.chat.completions.create(
                model="deepseek-chat", messages=messages, tools=TOOLS, tool_choice="auto"
            )
        except Exception as e:
            print(f"planner error: {e}")
            break
        msg = resp.choices[0].message
        if not msg.tool_calls:
            break

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": t.id, "type": "function",
                 "function": {"name": t.function.name, "arguments": t.function.arguments}}
                for t in msg.tool_calls
            ],
        })

        for t in msg.tool_calls:
            try:
                args = json.loads(t.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if t.function.name == "web_search":
                q = str(args.get("query", user_text))[:150]
                ui["text"] += f"\n🔎 {q}"
                results = await web_search(q)
                lines = []
                for r in results:
                    sources.append(r)
                    n = len(sources)
                    lines.append(f"[{n}] {r['title']}\n{r['url']}\n{r['snippet'][:300]}")
                    ui["text"] += f"\n   ↳ {r['title'][:60]}"
                out = "\n\n".join(lines) or "Ничего не найдено."
                ctx_parts.append(f"Поиск «{q}»:\n{out}")

            elif t.function.name == "open_url":
                url = str(args.get("url", ""))
                ui["text"] += f"\n📄 {re.sub(r'^https?://', '', url)[:60]}"
                page = await open_url(url)
                sources.append({"title": url, "url": url, "snippet": ""})
                out = f"Страница [{len(sources)}] {url}:\n{page}"
                ctx_parts.append(out)
            else:
                out = "Неизвестный инструмент"

            messages.append({"role": "tool", "tool_call_id": t.id, "content": out[:6000]})

    return "\n\n".join(ctx_parts), sources


# ====================== ОСНОВНОЙ ХЕНДЛЕР ======================
@dp.message()
async def handle_message(message: types.Message):
    if message.chat.type != "private" or not message.text:
        return

    uid = message.from_user.id
    lock = LOCKS.setdefault(uid, asyncio.Lock())
    if lock.locked():
        await message.answer("⏳ Подожди, я ещё отвечаю на предыдущий вопрос.")
        return

    async with lock:
        await process(message, uid)


async def process(message: types.Message, uid: int):
    chat_id = message.chat.id
    user_text = message.text
    draft_id = random.randint(1, 2**31 - 1)
    use_reasoning = REASONING.get(uid, False)
    use_search = SEARCH.get(uid, True)
    history = HISTORY.setdefault(uid, [])

    ui = {"phase": None, "text": "", "frame": 0, "t0": time.monotonic()}
    think_secs = 0

    if use_search:
        ui["phase"] = "search"
    elif use_reasoning:
        ui["phase"] = "think"

    async def animate():
        while ui["phase"] in ("search", "think"):
            title = "Ищу в интернете" if ui["phase"] == "search" else "Думаю"
            secs = int(time.monotonic() - ui["t0"])
            text, ents = render_status(title, secs, ui["frame"], ui["text"])
            try:
                await bot.send_message_draft(chat_id=chat_id, draft_id=draft_id, text=text, entities=ents)
            except Exception:
                pass
            ui["frame"] += 1
            await asyncio.sleep(STATUS_INTERVAL)

    anim_task = asyncio.create_task(animate()) if ui["phase"] else None

    search_ctx, sources = "", []
    if use_search:
        search_ctx, sources = await run_search_phase(history, user_text, ui)
        ui["phase"] = "think" if use_reasoning else "write"
        ui["t0"] = time.monotonic()
        ui["text"] = ""

    today = datetime.now().strftime("%d.%m.%Y")
    system = (
        f"Ты — умный дружелюбный ассистент. Сегодня {today}. Отвечай по существу и кратко, на языке пользователя. "
        "Код — в блоках ``` с указанием языка. Не пиши код, если вопрос не про код. Используй Markdown."
    )
    if search_ctx:
        system += (
            "\n\nНиже данные из интернета. Опирайся на них для актуальных фактов и ссылайся на источники "
            "номерами вида [1], [2]. Если данные противоречат друг другу или их мало — скажи об этом.\n\n"
            + search_ctx
        )

    model = "deepseek-reasoner" if use_reasoning else "deepseek-chat"
    messages = [{"role": "system", "content": system}, *history[-HISTORY_LIMIT:],
                {"role": "user", "content": user_text}]

    try:
        stream = await deepseek.chat.completions.create(model=model, messages=messages, stream=True)
    except Exception as e:
        ui["phase"] = None
        if anim_task:
            anim_task.cancel()
        await message.answer(f"⚠️ Ошибка: {html.escape(str(e))}", parse_mode="HTML")
        return

    answer, reasoning_text = "", ""
    last_render, last_sent = 0.0, ""

    try:
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            r = getattr(delta, "reasoning_content", None)
            if r:
                reasoning_text += r
                ui["text"] = reasoning_text

            d = getattr(delta, "content", None)
            if d:
                if not answer:
                    think_secs = int(time.monotonic() - ui["t0"]) if use_reasoning else 0
                    ui["phase"] = "write"
                answer += d

                now = time.monotonic()
                if now - last_render > ANSWER_INTERVAL:
                    preview = format_html(answer)
                    if len(preview) <= 4000 and preview != last_sent:
                        try:
                            await bot.send_message_draft(
                                chat_id=chat_id, draft_id=draft_id, text=preview, parse_mode="HTML"
                            )
                        except Exception:
                            try:
                                plain = re.sub(r"[`*#_~>]", "", answer)[-3800:]
                                await bot.send_message_draft(chat_id=chat_id, draft_id=draft_id, text=plain)
                            except Exception:
                                pass
                        last_sent, last_render = preview, now
    except Exception as e:
        print(f"stream error: {e}")
    finally:
        ui["phase"] = "write"
        if anim_task:
            try:
                await anim_task
            except Exception:
                pass

    if not answer:
        await message.answer("🤷 Модель не вернула текст.", reply_markup=keyboard(uid))
        return

    history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}]
    del history[:-HISTORY_LIMIT]

    final = format_html(answer)

    if use_reasoning and reasoning_text.strip():
        snippet = html.escape(reasoning_text.strip()[-1500:])
        final = (f"<blockquote expandable><b>🧠 Думал {think_secs}с</b>\n{snippet}</blockquote>\n\n" + final)

    if sources:
        seen, links = set(), []
        for i, s in enumerate(sources, 1):
            if s["url"] in seen or not s["url"]:
                continue
            seen.add(s["url"])
            title = html.escape((s["title"] or s["url"])[:60])
            links.append(f'{i}. <a href="{html.escape(s["url"], quote=True)}">{title}</a>')
        if links:
            final += "\n\n🌐 <b>Источники</b>\n<blockquote expandable>" + "\n".join(links[:8]) + "</blockquote>"

    parts = split_html(final)
    for i, part in enumerate(parts):
        kb = keyboard(uid) if i == len(parts) - 1 else None
        try:
            await bot.send_message(chat_id, part, parse_mode="HTML", reply_markup=kb,
                                   link_preview_options=types.LinkPreviewOptions(is_disabled=True))
        except Exception:
            await bot.send_message(chat_id, re.sub(r"<[^>]+>", "", part), reply_markup=kb)


async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
