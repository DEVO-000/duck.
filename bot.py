# -*- coding: utf-8 -*-
"""
Telegram-бот «Сканер документов».

Повторяет функции оригинального Flask-приложения
Smart Document Scanner & AI Text Enhancer (TA-SCANNER) — БЕЗ AI-обработки:

  1. Загрузка изображения (PNG/JPG/JPEG/WEBP, до 16 МБ) -> фото или файл в Telegram
  2. Идеальное выравнивание: перспектива (4-point transform) + доворот
     до строго горизонтального текста
  3. Фильтры: Ч/Б (без точек, со сглаживанием) / серый / цвет -> инлайн-кнопки
  4. OCR (Tesseract, рус+англ) — чистый текст ЦЕЛИКОМ, как в веб-версии
  5. Если на листе замечена таблица — бот СПРАШИВАЕТ, что распознать:
     «📊 Распознать таблицу» или «📄 Весь лист»
  6. Скачивание обработанного изображения -> кнопка «⬇️ Скачать JPG»
  7. Сборка страниц в PDF -> /pdf и кнопка «➕ В PDF»
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
from scanner import FILTER_KEYS, FILTER_LABELS_RU, detect_content_type, extract_table_text

DEFAULT_FILTER = "adaptive_threshold"  # как в оригинальной форме

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("doc-scanner-bot")

WELCOME = (
    "📄 <b>Сканер документов</b>\n\n"
    "Пришлите мне <b>фото документа</b> или <b>файл</b> (PNG / JPG / WEBP) — я:\n\n"
    "1️⃣ Идеально выровняю документ — перспектива + доворот текста\n"
    "    до строго горизонтального положения, как настоящий сканер\n"
    "2️⃣ Применю фильтр: ⚫ Ч/Б (без точек, со сглаживанием) / ◻ серый / 🎨 цвет\n"
    "3️⃣ Распознаю текст (русский + английский) и пришлю его целиком,\n"
    "    чистыми строками — как в веб-версии, без изменений\n"
    "4️⃣ Если замечу на листе таблицу — спрошу, что распознать:\n"
    "    📊 таблицу по ячейкам или 📄 весь лист целиком\n"
    "5️⃣ Отдам JPG-файл и соберу страницы в PDF\n\n"
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
            InlineKeyboardButton("📋 Текст", callback_data=f"text:{rid}"),
        ],
        [
            InlineKeyboardButton("🎨 Сменить фильтр", callback_data=f"refilter:{rid}"),
            InlineKeyboardButton(f"➕ В PDF ({pdf_count})", callback_data=f"pdfadd:{rid}"),
        ],
    ]
    if pdf_count > 0:
        rows.append([InlineKeyboardButton("📑 Собрать PDF сейчас", callback_data="pdfbuild")])
    return InlineKeyboardMarkup(rows)


def ask_keyboard(rid: str) -> InlineKeyboardMarkup:
    """Вопрос при обнаружении таблицы: распознать таблицу или весь лист."""
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("📊 Распознать таблицу", callback_data=f"tabocr:{rid}"),
        InlineKeyboardButton("📄 Весь лист", callback_data=f"pageocr:{rid}"),
    ]])


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


async def a_send_chunks(message_target, text: str, keyboard=None):
    chunks = split_text(text)
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        if i == total - 1 and keyboard is not None:
            await message_target(chunk, reply_markup=keyboard)
        else:
            await message_target(chunk)


# ----------------------------------------------------------------------------
# Основной конвейер обработки (аналог POST-запроса в оригинальном приложении)
# ----------------------------------------------------------------------------

def scan_caption(res: dict, scanned, filt: str, kind: str = None,
                 chars: int = None, ocr_err: str = None,
                 awaiting: bool = False) -> str:
    """Подпись к скану: что выровнено, какой фильтр, что определено."""
    lines = ["✅ <b>Скан готов!</b>"]
    if scanned is True:
        lines.append("✂️ Документ выровнен: перспектива + доворот текста.")
    elif scanned == "deskewed":
        lines.append("🧭 Границы не найдены — выровнял наклон текста и обработал весь лист.")
    else:
        lines.append("⚠️ Границы документа не найдены — обработал изображение целиком.")
    lines.append(f"🎨 Фильтр: <b>{FILTER_LABELS_RU.get(filt, filt)}</b>")
    if awaiting:
        lines.append(
            f"📊 <b>На листе похоже есть таблица</b> — {res.get('rows', 0)} × {res.get('cols', 0)}."
        )
        lines.append("Что распознать? Выберите кнопкой под сканом 👇")
        return "\n".join(lines)
    if kind == "table":
        lines.append(
            f"📊 <b>Распознано: таблица</b> — {res.get('rows', 0)} строк × {res.get('cols', 0)} столбцов"
        )
    elif kind == "page":
        lines.append("📄 <b>Определено: страница с текстом</b>")
    if chars is not None:
        lines.append(f"🔍 Распознано текста: {chars} символов" if chars else "🔍 Текст не найден")
    if ocr_err:
        lines.append(f"⚠️ OCR: {html.escape(ocr_err)}")
    return "\n".join(lines)


async def recognize_and_send(message, context: ContextTypes.DEFAULT_TYPE, rid: str,
                             res: dict, processed, content: dict, mode: str):
    """OCR выбранным способом + отправка чистого текста целиком.
    Возвращает (kind, text, ocr_err)."""
    ocr_err = None
    if mode == "table":
        await context.bot.send_chat_action(message.chat_id, ChatAction.TYPING)
        out_text = await asyncio.to_thread(
            extract_table_text, processed, content["h_lines"], content["v_lines"]
        )
        kind = "table"
        if len(out_text.strip()) < 10:
            # ячейки не распознались — OCR всей страницы (без линий сетки)
            ocr_src = processed
            if content.get("is_table"):
                ocr_src = await asyncio.to_thread(scanner.remove_grid_lines, processed)
            raw, ocr_err = await asyncio.to_thread(scanner.perform_ocr, ocr_src)
            if raw and raw.strip():
                out_text, kind = raw, "page"
    else:
        await context.bot.send_chat_action(message.chat_id, ChatAction.TYPING)
        ocr_src = processed
        if content.get("is_table"):
            # На листе есть таблица, но нужен весь лист: убираем линии сетки,
            # чтобы OCR прочитал и текст вокруг таблицы, и содержимое ячеек
            ocr_src = await asyncio.to_thread(scanner.remove_grid_lines, processed)
        raw, ocr_err = await asyncio.to_thread(scanner.perform_ocr, ocr_src)
        out_text = raw or ""
        kind = "page"

    res.update({"text": out_text, "kind": kind, "ts": time.time()})

    pdf_count = len(context.chat_data.get("pdf_pages", []))
    if out_text.strip():
        # Текст — ЦЕЛИКОМ, простым сообщением (копируется как есть)
        await a_send_chunks(
            message.reply_text,
            out_text.rstrip(),
            keyboard=result_keyboard(rid, pdf_count),
        )
    else:
        await message.reply_text(
            "🤔 Текст на изображении не найден.\n"
            "Попробуйте другой режим или смените фильтр.",
            reply_markup=result_keyboard(rid, pdf_count),
        )
    return kind, out_text, ocr_err


async def handle_mode_choice(query, context: ContextTypes.DEFAULT_TYPE,
                             rid: str, mode: str):
    """Пользователь выбрал: распознавать таблицу или весь лист."""
    res = context.chat_data.get("results", {}).get(rid)
    if not res or not os.path.exists(res.get("proc", "")):
        await query.message.reply_text("❌ Скан не найден — пришлите документ заново.")
        return
    processed = cv2.imread(res["proc"])
    if processed is None:
        await query.message.reply_text("❌ Не удалось перечитать обработанное изображение.")
        return
    content = res.get("content") or {}
    if mode == "table" and not content.get("is_table"):
        mode = "page"

    status = await query.message.reply_text(
        "📊 Распознаю таблицу по ячейкам…"
        if mode == "table" else
        "🔍 Распознаю весь лист целиком (OCR: русский + английский)…"
    )
    try:
        kind, out_text, ocr_err = await recognize_and_send(
            query.message, context, rid, res, processed, content, mode
        )
    finally:
        try:
            await status.delete()
        except Exception:
            pass

    # Обновляем подпись на скане (кнопки остаются — можно выбрать и другой режим)
    try:
        caption = scan_caption(
            res, res.get("scanned"), res.get("filter", DEFAULT_FILTER),
            kind=kind, chars=len(out_text.strip()), ocr_err=ocr_err,
        )
        await query.edit_message_caption(caption=caption, parse_mode="HTML")
    except Exception:
        pass


async def finalize_scan(message, context: ContextTypes.DEFAULT_TYPE, rid: str,
                        res: dict, processed, scanned, filt: str, content: dict):
    """Отправка скана с подписью. Если таблица — спрашивает, что распознать,
    иначе сразу распознаёт и присылает текст."""
    chat_id = message.chat_id
    await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_PHOTO)
    with open(res["proc"], "rb") as f:
        if content.get("is_table"):
            # Спрашиваем пользователя: таблица или весь лист
            caption = scan_caption(res, scanned, filt, awaiting=True)
            await message.reply_photo(
                photo=f, caption=caption, parse_mode="HTML",
                reply_markup=ask_keyboard(rid),
            )
        else:
            caption = scan_caption(res, scanned, filt, kind="page")
            await message.reply_photo(photo=f, caption=caption, parse_mode="HTML")
            await recognize_and_send(message, context, rid, res, processed, content, "page")


async def process_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE,
                           data: bytes, ext: str):
    """Цикл: сохранение -> выравнивание -> фильтр -> тип -> OCR -> отправка."""
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

        # 2. Выравнивание (перспектива + доворот) + фильтр
        await progress.edit_text(
            f"✂️ Выравниваю документ и применяю фильтр «{FILTER_LABELS_RU[filt]}»…"
        )
        processed, scanned, err = await asyncio.to_thread(
            scanner.process_document, image_cv, filt
        )
        if err:
            await progress.edit_text(f"❌ Ошибка обработки изображения: {err}")
            return

        proc_path = os.path.join(rdir, f"processed_{filt}.jpg")
        ok = cv2.imwrite(proc_path, processed, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            await progress.edit_text("❌ Не удалось сохранить обработанное изображение.")
            return

        # 3. Определение типа содержимого: страница или таблица
        await progress.edit_text("🔎 Определяю, что это: страница с текстом или таблица…")
        content = await asyncio.to_thread(detect_content_type, processed)

        chat_data = context.chat_data
        res = {
            "dir": rdir,
            "orig": orig_path,
            "proc": proc_path,
            "filter": filt,
            "scanned": scanned,
            "text": "",
            "kind": None,
            "rows": content.get("rows", 0),
            "cols": content.get("cols", 0),
            "content": content if content.get("is_table") else None,
            "ts": time.time(),
        }
        store_result(chat_data, rid, res)
        await progress.delete()

        # 4. Скан + подпись (+ вопрос, если таблица; + текст, если страница)
        await finalize_scan(update.message, context, rid, res, processed,
                            scanned, filt, content)

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
        if data.startswith("text:"):
            rid = data.split(":", 1)[1]
            res = context.chat_data.get("results", {}).get(rid)
            if not res or not res.get("text") or not res["text"].strip():
                await query.message.reply_text("📋 Текст не сохранился или пуст.")
                return
            await a_send_chunks(query.message.reply_text, res["text"].strip())

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

        elif data.startswith("tabocr:") or data.startswith("pageocr:"):
            mode = "table" if data.startswith("tabocr:") else "page"
            rid = data.split(":", 1)[1]
            await handle_mode_choice(query, context, rid, mode)

        elif data.startswith("refilter:"):
            rid = data.split(":", 1)[1]
            res = context.chat_data.get("results", {}).get(rid, {})
            await query.message.reply_text(
                "🎨 <b>Сменить фильтр</b> — пересчитаю этот документ заново:",
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
    cv2.imwrite(proc_path, processed, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    content = await asyncio.to_thread(detect_content_type, processed)
    await status.delete()

    res.update({
        "proc": proc_path,
        "filter": new_filter,
        "scanned": scanned,
        "text": "",
        "kind": None,
        "rows": content.get("rows", 0),
        "cols": content.get("cols", 0),
        "content": content if content.get("is_table") else None,
        "ts": time.time(),
    })

    await finalize_scan(query.message, context, rid, res, processed,
                        scanned, new_filter, content)


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
