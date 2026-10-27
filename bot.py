# -*- coding: utf-8 -*-
"""
Telegram-бот «Умный сканер документов + AI».

Повторяет ВСЕ функции оригинального Flask-приложения
Smart Document Scanner & AI Text Enhancer (TA-SCANNER):

  1. Загрузка изображения (PNG/JPG/JPEG/WEBP)      -> фото или файл в Telegram
  2. Перспективная коррекция (4-point transform)   -> scanner.process_document()
  3. Фильтры: Ч/Б (adaptive threshold) / серый / цвет -> инлайн-кнопки, пересчёт
  4. OCR (Tesseract)                               -> scanner.perform_ocr() (rus+eng)
  5. AI-улучшение текста                           -> ai_enhancer (GLM, как Llama в оригинале)
  6. Скачивание обработанного изображения          -> кнопка «Скачать JPG» (файлом)
  7. Копирование сырого и AI-текста                -> кнопки с текстом для копирования

Дополнительно:
  /pdf   — собрать все обработанные страницы сессии в один PDF-файл
  /clear — очистить очередь PDF и историю результатов
"""
import os
import sys
import html
import time
import uuid
import logging
import asyncio

from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ.get("BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
MAX_FILE_MB = 16  # как MAX_CONTENT_LENGTH в оригинале
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}  # как в оригинале

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

import cv2
import numpy as np
from PIL import Image
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

import scanner
from ai_enhancer import enhance_text_with_ai
from scanner import FILTER_KEYS, FILTER_LABELS_RU

DEFAULT_FILTER = "adaptive_threshold"  # как в оригинальной форме

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("doc-scanner-bot")

WELCOME = (
    "📄 <b>Умный сканер документов + AI</b>\n\n"
    "Пришлите мне <b>фото документа</b> или <b>файл</b> (PNG / JPG / WEBP) — я:\n\n"
    "1️⃣ Выровняю перспективу, как настоящий сканер\n"
    "2️⃣ Применю фильтр: ⚫ Ч/Б / ◻ серый / 🎨 цвет\n"
    "3️⃣ Распознаю текст (русский + английский)\n"
    "4️⃣ Улучшу и причешу текст с помощью AI\n"
    "5️⃣ Отдам JPG-файл, текст для копирования и соберу PDF\n\n"
    "<b>Команды:</b>\n"
    "• /filter — выбрать фильтр по умолчанию\n"
    "• /pdf — собрать очередь страниц в один PDF\n"
    "• /clear — очистить очередь PDF и историю\n"
    "• /help — эта справка"
)

FILTER_MENU = (
    "🎨 <b>Выберите фильтр по умолчанию</b>\n"
    "Он будет применяться к новым документам. "
    "Фильтр можно сменить и под конкретным результатом — кнопкой «Сменить фильтр»."
)


# ----------------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------------

def split_text(text: str, limit: int = 3800):
    """Режет длинный текст на части <= limit (лимит Telegram 4096)."""
    text = text or ""
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks or [""]


def result_keyboard(rid: str, pdf_count: int = 0) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton("⬇️ Скачать JPG", callback_data=f"jpg:{rid}"),
            InlineKeyboardButton("📋 Сырой текст", callback_data=f"raw:{rid}"),
        ],
        [
            InlineKeyboardButton("✨ AI-текст", callback_data=f"ai:{rid}"),
            InlineKeyboardButton("🎨 Сменить фильтр", callback_data=f"refilter:{rid}"),
        ],
        [
            InlineKeyboardButton(f"➕ В PDF ({pdf_count})", callback_data=f"pdfadd:{rid}"),
        ],
    ]
    if pdf_count > 0:
        rows.append([InlineKeyboardButton("📑 Собрать PDF сейчас", callback_data="pdfbuild")])
    return InlineKeyboardMarkup(rows)


def filter_keyboard(prefix: str, rid: str = "", current: str = "") -> InlineKeyboardMarkup:
    rows = []
    for key in FILTER_KEYS:
        label = FILTER_LABELS_RU.get(key, key)
        if key == current:
            label = f"✅ {label}"
        data = f"setf:{rid}:{key}" if rid else f"{prefix}:{key}"
        rows.append([InlineKeyboardButton(label, callback_data=data)])
    return InlineKeyboardMarkup(rows)


def store_result(chat_data: dict, rid: str, data: dict):
    results = chat_data.setdefault("results", {})
    results[rid] = data
    # держим не больше 20 последних результатов на чат
    if len(results) > 20:
        oldest = sorted(results.items(), key=lambda kv: kv[1].get("ts", 0))[:-20]
        for old_rid, _ in oldest:
            results.pop(old_rid, None)


async def send_text_chunks(message_target, text: str, parse_mode=None, keyboard=None):
    """Отправляет длинный текст по частям; клавиатура крепится к последней части."""
    chunks = split_text(text)
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        last = i == total - 1
        if last and keyboard is not None:
            await message_target(chunk, parse_mode=parse_mode, reply_markup=keyboard)
        else:
            await message_target(chunk, parse_mode=parse_mode)


# ----------------------------------------------------------------------------
# Основной конвейер обработки (аналог POST-запроса в оригинальном приложении)
# ----------------------------------------------------------------------------

async def process_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE,
                           data: bytes, ext: str):
    """Полный цикл: сохранение -> перспектива -> фильтр -> OCR -> AI -> отправка."""
    chat_id = update.effective_chat.id
    filt = context.chat_data.get("default_filter", DEFAULT_FILTER)

    progress = await update.message.reply_text("⏳ Принял изображение, обрабатываю…")
    try:
        # 1. Сохранение и чтение изображения (аналог file.save + cv2.imread)
        rid = uuid.uuid4().hex[:10]
        rdir = os.path.join(RESULTS_DIR, str(chat_id), rid)
        os.makedirs(rdir, exist_ok=True)
        orig_path = os.path.join(rdir, f"original.{ext}")
        with open(orig_path, "wb") as f:
            f.write(data)

        image_cv = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image_cv is None:
            await progress.edit_text(
                "❌ Не удалось прочитать изображение — файл повреждён или формат не поддерживается."
            )
            return

        # 2. Перспективная коррекция + фильтр (process_document из оригинала)
        await progress.edit_text(
            f"✂️ Выравниваю перспективу и применяю фильтр «{FILTER_LABELS_RU[filt]}»…"
        )
        processed, scanned, err = await asyncio.to_thread(
            scanner.process_document, image_cv, filt
        )
        if err:
            await progress.edit_text(f"❌ Ошибка обработки изображения: {err}")
            return

        proc_path = os.path.join(rdir, f"processed_{filt}.jpg")
        ok = cv2.imwrite(proc_path, processed, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            await progress.edit_text("❌ Не удалось сохранить обработанное изображение.")
            return

        # 3. OCR (Tesseract, rus+eng) — как в оригинале
        await progress.edit_text("🔍 Распознаю текст (OCR: русский + английский)…")
        await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
        raw_text, ocr_err = await asyncio.to_thread(scanner.perform_ocr, processed)

        # 4. AI-улучшение текста — как enhance_text_with_ai в оригинале
        ai_text, ai_err = None, None
        if raw_text and raw_text.strip():
            await progress.edit_text("✨ AI улучшает и форматирует текст… (10–40 сек)")
            ai_text, ai_err = await asyncio.to_thread(enhance_text_with_ai, raw_text)

        await progress.delete()

        # 5. Отправка результатов
        await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_PHOTO)
        caption_lines = ["✅ <b>Скан готов!</b>"]
        if scanned:
            caption_lines.append("✂️ Перспектива выровнена (4-point transform).")
        else:
            caption_lines.append("⚠️ Границы документа не найдены — обработал изображение целиком.")
        caption_lines.append(f"🎨 Фильтр: <b>{FILTER_LABELS_RU[filt]}</b>")
        if raw_text is not None:
            caption_lines.append(f"🔍 OCR: {len(raw_text.strip())} символов")
        if ocr_err:
            caption_lines.append(f"⚠️ OCR: {html.escape(ocr_err)}")
        if ai_err:
            caption_lines.append(f"⚠️ {html.escape(ai_err)}")
        if ai_text and raw_text is not None and raw_text.strip() == (ai_text or "").strip():
            caption_lines.append("ℹ️ AI счёл текст уже чистым — улучшения не требуются.")

        with open(proc_path, "rb") as f:
            await update.message.reply_photo(
                photo=f,
                caption="\n".join(caption_lines),
                parse_mode="HTML",
            )

        # AI-текст (для копирования) + клавиатура действий
        chat_data = context.chat_data
        pdf_count = len(chat_data.get("pdf_pages", []))
        store_result(chat_data, rid, {
            "dir": rdir,
            "orig": orig_path,
            "proc": proc_path,
            "filter": filt,
            "scanned": scanned,
            "raw": raw_text,
            "ai": ai_text,
            "ts": time.time(),
        })

        if ai_text and ai_text.strip():
            header = "✨ <b>AI-улучшенный текст</b> (можно копировать):\n\n"
            await send_text_chunks(
                update.message.reply_text,
                header + ai_text.strip(),
                parse_mode="HTML",
                keyboard=result_keyboard(rid, pdf_count),
            )
        elif raw_text and raw_text.strip():
            header = "📋 <b>Распознанный текст</b> (AI-улучшение недоступно):\n\n"
            await send_text_chunks(
                update.message.reply_text,
                header + html.escape(raw_text.strip()),
                parse_mode="HTML",
                keyboard=result_keyboard(rid, pdf_count),
            )
        else:
            await update.message.reply_text(
                "🤔 Текст на изображении не найден.\n"
                "Попробуйте фото с более чётким текстом или смените фильтр.",
                reply_markup=result_keyboard(rid, pdf_count),
            )

    except Exception as e:
        log.exception("Ошибка обработки: %s", e)
        try:
            await progress.edit_text(f"❌ Непредвиденная ошибка при обработке: {e}")
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Обработчики входных изображений
# ----------------------------------------------------------------------------

async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]  # самое большое разрешение
    tg_file = await photo.get_file()
    data = bytes(await tg_file.download_as_bytearray())
    await process_and_send(update, context, data, "jpg")


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    name = doc.file_name or "image"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if ext not in ALLOWED_EXTENSIONS:
        await update.message.reply_text(
            "❌ Неверный тип файла. Разрешены: png, jpg, jpeg, webp\n"
            "(как и в оригинальном приложении)"
        )
        return
    if (doc.file_size or 0) > MAX_FILE_MB * 1024 * 1024:
        await update.message.reply_text(f"❌ Файл больше {MAX_FILE_MB} МБ — это превышает лимит.")
        return

    tg_file = await doc.get_file()
    data = bytes(await tg_file.download_as_bytearray())
    await process_and_send(update, context, data, ext)


# ----------------------------------------------------------------------------
# Команды
# ----------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_html(WELCOME)


async def cmd_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current = context.chat_data.get("default_filter", DEFAULT_FILTER)
    await update.message.reply_html(
        FILTER_MENU,
        reply_markup=filter_keyboard("deffilter", current=current),
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data.pop("pdf_pages", None)
    context.chat_data.pop("results", None)
    chat_dir = os.path.join(RESULTS_DIR, str(update.effective_chat.id))
    if os.path.isdir(chat_dir):
        for root, dirs, files in os.walk(chat_dir, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except OSError:
                    pass
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except OSError:
                    pass
    await update.message.reply_text("🧹 Очередь PDF и история очищены.")


async def cmd_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await build_and_send_pdf(update, context)


async def build_and_send_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pages = context.chat_data.get("pdf_pages", [])
    if not pages:
        await update.message.reply_text(
            "📭 Очередь PDF пуста.\n\n"
            "Обработайте документы и нажимайте «➕ В PDF» под результатами — "
            "затем я соберу их в один файл командой /pdf."
        )
        return

    note = await update.message.reply_text(f"📑 Собираю PDF из {len(pages)} стр.…")
    try:
        images = []
        for p in pages:
            path = p["path"] if isinstance(p, dict) else p
            img = Image.open(path).convert("RGB")
            images.append(img)
        ts = time.strftime("%Y%m%d_%H%M%S")
        chat_dir = os.path.join(RESULTS_DIR, str(update.effective_chat.id))
        os.makedirs(chat_dir, exist_ok=True)
        pdf_path = os.path.join(chat_dir, f"scan_{ts}.pdf")
        images[0].save(
            pdf_path, save_all=True, append_images=images[1:],
            resolution=150.0,
        )
        await note.delete()
        with open(pdf_path, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename=os.path.basename(pdf_path),
                caption=f"📑 Ваш PDF: {len(pages)} стр.",
            )
        context.chat_data["pdf_pages"] = []
    except Exception as e:
        log.exception("Ошибка сборки PDF: %s", e)
        await note.edit_text(f"❌ Ошибка при сборке PDF: {e}")


# ----------------------------------------------------------------------------
# Инлайн-кнопки
# ----------------------------------------------------------------------------

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data or ""
    await query.answer()

    try:
        if data.startswith("raw:"):
            rid = data.split(":", 1)[1]
            res = context.chat_data.get("results", {}).get(rid)
            if not res or not res.get("raw") or not res["raw"].strip():
                await query.message.reply_text("📋 Сырой текст не сохранился или пуст.")
                return
            await send_text_chunks(
                query.message.reply_text,
                "📋 <b>Сырой текст (OCR)</b> — копируйте целиком:\n\n"
                + html.escape(res["raw"].strip()),
                parse_mode="HTML",
            )

        elif data.startswith("ai:"):
            rid = data.split(":", 1)[1]
            res = context.chat_data.get("results", {}).get(rid)
            if not res or not res.get("ai") or not res["ai"].strip():
                await query.message.reply_text("✨ AI-текст недоступен (пустой OCR или ошибка AI).")
                return
            await send_text_chunks(
                query.message.reply_text,
                "✨ <b>AI-улучшенный текст</b> — копируйте целиком:\n\n"
                + html.escape(res["ai"].strip()),
                parse_mode="HTML",
            )

        elif data.startswith("jpg:"):
            rid = data.split(":", 1)[1]
            res = context.chat_data.get("results", {}).get(rid)
            if not res or not os.path.exists(res.get("proc", "")):
                await query.message.reply_text("❌ Файл не найден — обработайте документ заново.")
                return
            await context.bot.send_chat_action(query.message.chat_id, ChatAction.UPLOAD_DOCUMENT)
            with open(res["proc"], "rb") as f:
                await query.message.reply_document(
                    document=f,
                    filename=f"scan_{rid}.jpg",
                    caption="⬇️ Обработанное изображение (без сжатия Telegram)",
                )

        elif data.startswith("refilter:"):
            rid = data.split(":", 1)[1]
            res = context.chat_data.get("results", {}).get(rid, {})
            await query.message.reply_text(
                f"🎨 <b>Сменить фильтр</b> — пересчитаю этот документ заново:",
                parse_mode="HTML",
                reply_markup=filter_keyboard("setf", rid=rid, current=res.get("filter", "")),
            )

        elif data.startswith("setf:"):
            _, rid, new_filter = data.split(":", 2)
            await refilter_result(query, context, rid, new_filter)

        elif data.startswith("deffilter:"):
            key = data.split(":", 1)[1]
            context.chat_data["default_filter"] = key
            await query.edit_message_text(
                f"✅ Фильтр по умолчанию: <b>{FILTER_LABELS_RU[key]}</b>",
                parse_mode="HTML",
            )

        elif data.startswith("pdfadd:"):
            rid = data.split(":", 1)[1]
            res = context.chat_data.get("results", {}).get(rid)
            if not res or not os.path.exists(res.get("proc", "")):
                await query.message.reply_text("❌ Страница недоступна — обработайте документ заново.")
                return
            pages = context.chat_data.setdefault("pdf_pages", [])
            pages.append({"rid": rid, "path": res["proc"]})
            kb = result_keyboard(rid, len(pages))
            try:
                await query.edit_message_reply_markup(reply_markup=kb)
            except Exception:
                await query.message.reply_text(
                    f"➕ Страница добавлена в очередь PDF (всего: {len(pages)})",
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("📑 Собрать PDF", callback_data="pdfbuild")]]
                    ),
                )

        elif data == "pdfbuild":
            await build_and_send_pdf_from_callback(query, context)

    except Exception as e:
        log.exception("Ошибка обработки кнопки %s: %s", data, e)
        try:
            await query.message.reply_text(f"❌ Ошибка: {e}")
        except Exception:
            pass


async def refilter_result(query, context: ContextTypes.DEFAULT_TYPE, rid: str, new_filter: str):
    """Пересчёт того же документа с новым фильтром (аналог повторной отправки
    формы в оригинальном приложении с другим фильтром)."""
    res = context.chat_data.get("results", {}).get(rid)
    if not res or not os.path.exists(res.get("orig", "")):
        await query.message.reply_text("❌ Исходное изображение не найдено — пришлите документ заново.")
        return

    status = await query.message.reply_text(
        f"🔄 Пересчитываю с фильтром «{FILTER_LABELS_RU.get(new_filter, new_filter)}»…"
    )
    image_cv = cv2.imread(res["orig"])
    if image_cv is None:
        await status.edit_text("❌ Не удалось перечитать исходное изображение.")
        return

    processed, scanned, err = await asyncio.to_thread(
        scanner.process_document, image_cv, new_filter
    )
    if err:
        await status.edit_text(f"❌ Ошибка обработки: {err}")
        return

    proc_path = os.path.join(res["dir"], f"processed_{new_filter}.jpg")
    cv2.imwrite(proc_path, processed, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    raw_text, ocr_err = await asyncio.to_thread(scanner.perform_ocr, processed)
    ai_text, ai_err = None, None
    if raw_text and raw_text.strip():
        await status.edit_text("✨ AI улучшает и форматирует текст… (10–40 сек)")
        ai_text, ai_err = await asyncio.to_thread(enhance_text_with_ai, raw_text)
    await status.delete()

    res.update({
        "proc": proc_path,
        "filter": new_filter,
        "scanned": scanned,
        "raw": raw_text,
        "ai": ai_text,
        "ts": time.time(),
    })

    caption_lines = ["✅ <b>Пересчитано!</b>"]
    caption_lines.append("✂️ Перспектива выровнена." if scanned
                         else "⚠️ Границы не найдены — обработан весь кадр.")
    caption_lines.append(f"🎨 Фильтр: <b>{FILTER_LABELS_RU[new_filter]}</b>")
    if ocr_err:
        caption_lines.append(f"⚠️ OCR: {html.escape(ocr_err)}")
    if ai_err:
        caption_lines.append(f"⚠️ {html.escape(ai_err)}")

    pdf_count = len(context.chat_data.get("pdf_pages", []))
    with open(proc_path, "rb") as f:
        await query.message.reply_photo(
            photo=f,
            caption="\n".join(caption_lines),
            parse_mode="HTML",
        )

    if ai_text and ai_text.strip():
        await send_text_chunks(
            query.message.reply_text,
            "✨ <b>AI-улучшенный текст</b> (можно копировать):\n\n" + html.escape(ai_text.strip()),
            parse_mode="HTML",
            keyboard=result_keyboard(rid, pdf_count),
        )
    elif raw_text and raw_text.strip():
        await send_text_chunks(
            query.message.reply_text,
            "📋 <b>Распознанный текст</b>:\n\n" + html.escape(raw_text.strip()),
            parse_mode="HTML",
            keyboard=result_keyboard(rid, pdf_count),
        )
    else:
        await query.message.reply_text(
            "🤔 Текст не найден и с этим фильтром.",
            reply_markup=result_keyboard(rid, pdf_count),
        )


async def build_and_send_pdf_from_callback(query, context: ContextTypes.DEFAULT_TYPE):
    pages = context.chat_data.get("pdf_pages", [])
    if not pages:
        await query.message.reply_text("📭 Очередь PDF пуста — сначала добавьте страницы кнопкой «➕ В PDF».")
        return
    note = await query.message.reply_text(f"📑 Собираю PDF из {len(pages)} стр.…")
    try:
        images = [Image.open(p["path"]).convert("RGB") for p in pages]
        ts = time.strftime("%Y%m%d_%H%M%S")
        chat_dir = os.path.join(RESULTS_DIR, str(query.message.chat_id))
        os.makedirs(chat_dir, exist_ok=True)
        pdf_path = os.path.join(chat_dir, f"scan_{ts}.pdf")
        images[0].save(pdf_path, save_all=True, append_images=images[1:], resolution=150.0)
        await note.delete()
        with open(pdf_path, "rb") as f:
            await query.message.reply_document(
                document=f,
                filename=os.path.basename(pdf_path),
                caption=f"📑 Ваш PDF: {len(pages)} стр.",
            )
        context.chat_data["pdf_pages"] = []
    except Exception as e:
        log.exception("Ошибка сборки PDF: %s", e)
        await note.edit_text(f"❌ Ошибка при сборке PDF: {e}")


# ----------------------------------------------------------------------------
# Точка входа
# ----------------------------------------------------------------------------

def main():
    if not BOT_TOKEN or "ВСТАВЬТЕ" in BOT_TOKEN:
        print(
            "\n" + "!" * 66 +
            "\n!  НЕ ЗАДАН ТОКЕН БОТА                                      !"
            "\n!                                                           !"
            "\n!  1. Откройте @BotFather в Telegram                         !"
            "\n!  2. Отправьте команду /newbot и следуйте инструкциям       !"
            "\n!  3. Полученный токен вида 123456789:AAH... впишите в .env: !"
            "\n!       BOT_TOKEN=ваш_токен                                 !"
            "\n!  4. Запустите снова: python bot.py                        !"
            "\n" + "!" * 66 + "\n"
        )
        sys.exit(1)

    app = Application.builder().token(BOT_TOKEN).concurrent_updates(True).build()

    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(CommandHandler("filter", cmd_filter))
    app.add_handler(CommandHandler("pdf", cmd_pdf))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))

    log.info("✅ Бот запущен и слушает обновления…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
