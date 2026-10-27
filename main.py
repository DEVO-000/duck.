"""
Telegram-бот «Сканер»: присылаешь фото листа — получаешь выровненный ч/б скан.
Если лист определился неточно, есть ручная подгонка углов кнопками.

Файлы рядом: bot.py и scanner.py.

Локальный запуск:
    pip install -r requirements.txt
    BOT_TOKEN=123:ABC python bot.py        (Windows: set BOT_TOKEN=123:ABC)
"""

import asyncio
import logging
import os
import secrets
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial

import cv2
import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiohttp import web

from scanner import find_document_contour, four_point_transform, order_points, to_black_white

logging.basicConfig(level=logging.INFO)

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise SystemExit("Задай переменную окружения BOT_TOKEN")

bot = Bot(TOKEN)
dp = Dispatcher()

MAX_FILE_MB = 20
MAX_SESSIONS = 20          # сколько последних фото помним для ручной подгонки
PREVIEW_SIDE = 1000        # размер превью с рамкой
STEPS = [0.005, 0.02, 0.06]  # шаг сдвига угла в долях от большей стороны фото
CORNER_NAMES = ["левый верхний", "правый верхний", "правый нижний", "левый нижний"]

user_mode: dict[int, str] = {}  # "bw" | "gray"


# ───────────────────────── обработка изображений ─────────────────────────

@dataclass
class Session:
    sid: str
    data: bytes            # исходный файл
    preview: np.ndarray    # уменьшенная копия для рисования рамки
    ratio: float           # во сколько раз оригинал больше превью
    size: tuple            # (ширина, высота) оригинала
    pts: np.ndarray        # 4 угла в координатах оригинала
    auto_pts: np.ndarray   # то, что нашёл автомат
    mode: str
    sel: int = 0
    step_idx: int = 1


sessions: "OrderedDict[int, Session]" = OrderedDict()


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("не удалось прочитать изображение")
    return img


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("не удалось закодировать результат")
    return buf.tobytes()


def prepare(data: bytes, mode: str):
    """Автоскан + подготовка сессии для ручной подгонки."""
    image = decode(data)
    h, w = image.shape[:2]

    pts = find_document_contour(image)
    found = pts is not None
    if found:
        pts = order_points(pts)
        warped = four_point_transform(image, pts)
    else:
        m = 0.05
        pts = np.array(
            [[w * m, h * m], [w * (1 - m), h * m], [w * (1 - m), h * (1 - m)], [w * m, h * (1 - m)]],
            dtype="float32",
        )
        warped = image

    png = encode_png(to_black_white(warped, mode))

    k = PREVIEW_SIDE / max(h, w)
    preview = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else image.copy()
    ratio = max(h, w) / max(preview.shape[:2])

    s = Session(
        sid=secrets.token_hex(3),
        data=data,
        preview=preview,
        ratio=ratio,
        size=(w, h),
        pts=pts.copy(),
        auto_pts=pts.copy(),
        mode=mode,
    )
    return s, png, found


def render_manual(s: Session) -> bytes:
    """Итоговый скан по вручную подогнанной рамке."""
    w, h = s.size
    if cv2.contourArea(order_points(s.pts)) < 0.01 * w * h:
        raise ValueError("рамка слишком маленькая или перекручена")
    image = decode(s.data)
    warped = four_point_transform(image, s.pts)
    return encode_png(to_black_white(warped, s.mode))


def draw_preview(s: Session) -> bytes:
    img = s.preview.copy()
    p = (s.pts / s.ratio).astype(int)
    cv2.polylines(img, [p.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (x, y) in enumerate(p):
        x, y = int(x), int(y)
        selected = i == s.sel
        cv2.circle(img, (x, y), 13 if selected else 9, (0, 0, 255) if selected else (0, 200, 0), -1)
        cv2.circle(img, (x, y), 13 if selected else 9, (255, 255, 255), 2)
        cv2.putText(img, str(i + 1), (x + 16, y - 16), font, 0.9, (0, 0, 0), 4)
        cv2.putText(img, str(i + 1), (x + 16, y - 16), font, 0.9, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes()


# ───────────────────────── клавиатуры ─────────────────────────

def cbd(s: Session, cmd: str, arg="") -> str:
    return f"m:{s.sid}:{cmd}:{arg}"


def manual_keyboard(s: Session) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.row(*[
        InlineKeyboardButton(text=("🔴" if i == s.sel else "⚪") + str(i + 1), callback_data=cbd(s, "sel", i))
        for i in range(4)
    ])
    b.row(
        InlineKeyboardButton(text="⬅️", callback_data=cbd(s, "mv", "l")),
        InlineKeyboardButton(text="⬆️", callback_data=cbd(s, "mv", "u")),
        InlineKeyboardButton(text="⬇️", callback_data=cbd(s, "mv", "d")),
        InlineKeyboardButton(text="➡️", callback_data=cbd(s, "mv", "r")),
    )
    b.row(
        InlineKeyboardButton(text=f"Шаг: {STEPS[s.step_idx] * 100:g}%", callback_data=cbd(s, "step")),
        InlineKeyboardButton(text="↩️ Сброс", callback_data=cbd(s, "reset")),
    )
    b.row(
        InlineKeyboardButton(text="✅ Готово", callback_data=cbd(s, "ok")),
        InlineKeyboardButton(text="❌ Отмена", callback_data=cbd(s, "x")),
    )
    return b.as_markup()


def manual_caption(s: Session) -> str:
    return (
        f"Угол {s.sel + 1} ({CORNER_NAMES[s.sel]}). Двигай стрелками, пока зелёная рамка "
        "не совпадёт с краями листа. Шаг меняется кнопкой «Шаг»."
    )


# ───────────────────────── хэндлеры ─────────────────────────

@dp.message(Command("start", "help"))
async def cmd_start(m: Message):
    await m.answer(
        "📄 Пришли фото листа — я найду его, выровняю и сделаю ч/б скан.\n\n"
        "Лучше отправлять «как файл» (без сжатия) — качество будет выше.\n"
        "Если лист нашёлся неточно, под результатом есть кнопка «✋ Поправить вручную».\n\n"
        "Режимы:\n"
        "/bw — чёрно-белый (по умолчанию)\n"
        "/gray — серый, лучше для фото и рисунков"
    )


@dp.message(Command("bw"))
async def cmd_bw(m: Message):
    user_mode[m.from_user.id] = "bw"
    await m.answer("Режим: чёрно-белый ✅")


@dp.message(Command("gray"))
async def cmd_gray(m: Message):
    user_mode[m.from_user.id] = "gray"
    await m.answer("Режим: серый ✅")


async def handle_image(m: Message, file_id: str, size):
    if size and size > MAX_FILE_MB * 1024 * 1024:
        await m.answer(f"Файл слишком большой (максимум {MAX_FILE_MB} МБ).")
        return

    status = await m.answer("⏳ Сканирую...")
    try:
        buf = await bot.download(file_id)
        data = buf.read()
        mode = user_mode.get(m.from_user.id, "bw")

        loop = asyncio.get_running_loop()
        s, png, found = await loop.run_in_executor(None, partial(prepare, data, mode))

        sessions[m.from_user.id] = s
        sessions.move_to_end(m.from_user.id)
        while len(sessions) > MAX_SESSIONS:
            sessions.popitem(last=False)

        caption = "✅ Готово" if found else "⚠️ Лист не нашёл, обработал фото целиком"
        kb = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="✋ Поправить вручную", callback_data=cbd(s, "go"))]]
        )
        await m.answer_document(BufferedInputFile(png, filename="scan.png"), caption=caption, reply_markup=kb)
        await status.delete()
    except Exception as e:
        logging.exception("scan failed")
        await status.edit_text(f"❌ Не получилось обработать: {e}")


@dp.message(F.photo)
async def on_photo(m: Message):
    p = m.photo[-1]
    await handle_image(m, p.file_id, p.file_size)


@dp.message(F.document & F.document.mime_type.startswith("image/"))
async def on_document(m: Message):
    await handle_image(m, m.document.file_id, m.document.file_size)


@dp.callback_query(F.data.startswith("m:"))
async def on_manual(cb: CallbackQuery):
    _, sid, cmd, arg = cb.data.split(":", 3)
    s = sessions.get(cb.from_user.id)
    if not s or s.sid != sid:
        await cb.answer("Это фото уже устарело — пришли его заново", show_alert=True)
        return

    loop = asyncio.get_running_loop()

    if cmd == "go":
        await cb.answer()
        jpg = await loop.run_in_executor(None, draw_preview, s)
        await cb.message.answer_photo(
            BufferedInputFile(jpg, "preview.jpg"), caption=manual_caption(s), reply_markup=manual_keyboard(s)
        )
        return

    if cmd == "x":
        await cb.answer()
        await cb.message.delete()
        return

    if cmd == "ok":
        await cb.answer("Обрабатываю...")
        try:
            png = await loop.run_in_executor(None, render_manual, s)
        except Exception as e:
            await cb.message.answer(f"❌ {e}")
            return
        await cb.message.answer_document(BufferedInputFile(png, "scan.png"), caption="✅ Готово (ручная подгонка)")
        await cb.message.delete()
        return

    # изменения рамки
    w, h = s.size
    if cmd == "sel":
        s.sel = int(arg)
    elif cmd == "step":
        s.step_idx = (s.step_idx + 1) % len(STEPS)
    elif cmd == "reset":
        s.pts = s.auto_pts.copy()
    elif cmd == "mv":
        step = STEPS[s.step_idx] * max(w, h)
        dx, dy = {"l": (-step, 0), "r": (step, 0), "u": (0, -step), "d": (0, step)}[arg]
        s.pts[s.sel] += (dx, dy)
        s.pts[:, 0] = np.clip(s.pts[:, 0], 0, w - 1)
        s.pts[:, 1] = np.clip(s.pts[:, 1], 0, h - 1)

    await cb.answer()
    jpg = await loop.run_in_executor(None, draw_preview, s)
    try:
        await cb.message.edit_media(
            InputMediaPhoto(media=BufferedInputFile(jpg, "preview.jpg"), caption=manual_caption(s)),
            reply_markup=manual_keyboard(s),
        )
    except TelegramBadRequest as e:
        if "not modified" not in str(e):
            raise


@dp.message()
async def fallback(m: Message):
    await m.answer("Пришли фото листа 📷 (или /help)")


# ───────────────────────── запуск (Render) ─────────────────────────

async def start_web():
    """Мини-сервер для Render Web Service: отвечает на /health, чтобы сервис считался живым."""
    port = os.getenv("PORT")
    if not port:
        return

    async def ok(_):
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/", ok)
    app.router.add_get("/health", ok)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(port)).start()
    logging.info("health server on :%s", port)


async def main():
    await start_web()
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
