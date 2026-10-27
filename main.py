"""
Telegram-бот «Сканер»: присылаешь фото листа — получаешь выровненный ч/б скан.

Файлы рядом: bot.py и scanner.py (из прошлого сообщения).

Установка:
    pip install aiogram opencv-python-headless numpy

Запуск:
    BOT_TOKEN=123:ABC python bot.py        (Windows: set BOT_TOKEN=123:ABC)
"""

import asyncio
import logging
import os
from functools import partial

import cv2
import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, Message

from scanner import find_document_contour, four_point_transform, to_black_white

logging.basicConfig(level=logging.INFO)

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise SystemExit("Задай переменную окружения BOT_TOKEN")

bot = Bot(TOKEN)
dp = Dispatcher()

# режим на пользователя: "bw" или "gray" (хранится в памяти)
user_mode: dict[int, str] = {}

MAX_FILE_MB = 20


def process(data: bytes, mode: str) -> tuple[bytes, bool]:
    """Блокирующая обработка. Возвращает (png-байты, найден ли лист)."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("не удалось прочитать изображение")

    pts = find_document_contour(image)
    found = pts is not None
    warped = four_point_transform(image, pts) if found else image

    result = to_black_white(warped, mode)
    ok, buf = cv2.imencode(".png", result)
    if not ok:
        raise ValueError("не удалось закодировать результат")
    return buf.tobytes(), found


@dp.message(Command("start", "help"))
async def cmd_start(m: Message):
    await m.answer(
        "📄 Пришли фото листа — я найду его, выровняю и сделаю ч/б скан.\n\n"
        "Лучше отправлять «как файл» (без сжатия) — качество будет выше.\n\n"
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


async def handle_image(m: Message, file_id: str, size: int | None):
    if size and size > MAX_FILE_MB * 1024 * 1024:
        await m.answer(f"Файл слишком большой (максимум {MAX_FILE_MB} МБ).")
        return

    status = await m.answer("⏳ Сканирую...")
    try:
        buf = await bot.download(file_id)
        data = buf.read()
        mode = user_mode.get(m.from_user.id, "bw")

        loop = asyncio.get_running_loop()
        png, found = await loop.run_in_executor(None, partial(process, data, mode))

        caption = "✅ Готово" if found else "⚠️ Лист не нашёл, обработал фото целиком"
        await m.answer_document(BufferedInputFile(png, filename="scan.png"), caption=caption)
        await status.delete()
    except Exception as e:
        logging.exception("scan failed")
        await status.edit_text(f"❌ Не получилось обработать: {e}")


@dp.message(F.photo)
async def on_photo(m: Message):
    p = m.photo[-1]  # самое большое
    await handle_image(m, p.file_id, p.file_size)


@dp.message(F.document & F.document.mime_type.startswith("image/"))
async def on_document(m: Message):
    await handle_image(m, m.document.file_id, m.document.file_size)


@dp.message()
async def fallback(m: Message):
    await m.answer("Пришли фото листа 📷 (или /help)")


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
