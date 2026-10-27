# -*- coding: utf-8 -*-
"""
ai_enhancer.py — AI-улучшение распознанного (OCR) текста.

Аналог функции enhance_text_with_ai() из оригинального app-ocr.py,
но вместо Replicate (meta/llama-2-13b-chat) используется встроенная
LLM (GLM) через z-ai CLI — работает без сторонних API-ключей.

Системный промпт сохранён из оригинального приложения.
Возвращает (улучшенный_текст, ошибка) — при ошибке отдаёт исходный текст.
"""
import os
import json
import subprocess
import tempfile

# Промпт из оригинального приложения (enhance_text_with_ai), сохранён
# и дополнен требованием сохранять язык исходного текста.
SYSTEM_PROMPT = (
    "You are a helpful assistant that meticulously cleans and formats OCR text. "
    "Your primary goal is to correct errors and make the text highly readable and "
    "easily copy-pasteable, as if it were typed manually. Output *only* the cleaned "
    "text, without any conversational preamble, summaries, or explanations about your "
    "process. Keep the original language of the text unchanged. "
    "Correct OCR errors (misrecognized characters, incorrect spacing, formatting "
    "issues, fragmented sentences) and format the text logically with standard "
    "paragraph breaks. If you detect lists or clear headings, format them "
    "appropriately (e.g., using '-' for list items)."
)

# Ограничение аргумента командной строки Linux (~128 КиБ); кириллица в UTF-8
# занимает 2 байта/символ, поэтому безопасный порог — 40 000 символов.
MAX_INPUT_CHARS = 40_000

AI_TIMEOUT = int(os.environ.get("AI_TIMEOUT", "240"))


def enhance_text_with_ai(text: str, timeout: int = AI_TIMEOUT):
    """Очищает и форматирует OCR-текст через AI. Возвращает (text, error)."""
    if not text or not text.strip():
        return "", None

    payload = text.strip()
    truncated = False
    if len(payload) > MAX_INPUT_CHARS:
        payload = payload[:MAX_INPUT_CHARS]
        truncated = True

    fd, out_path = tempfile.mkstemp(prefix="ai_enh_", suffix=".json")
    os.close(fd)
    try:
        cmd = [
            "z-ai", "chat",
            "-p", payload,
            "-s", SYSTEM_PROMPT,
            "-o", out_path,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "неизвестная ошибка")[:300]
            return text, f"Ошибка AI: {err}"

        with open(out_path, encoding="utf-8") as f:
            resp = json.load(f)

        content = resp["choices"][0]["message"]["content"].strip()

        # Модель иногда оборачивает ответ в кодовые блоки — убираем
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith(("text", "cleaned")):
                content = content.split("\n", 1)[-1].strip()

        if not content:
            return text, None

        if truncated:
            content += "\n\n⚠️ Входной текст был слишком длинным и обрезан перед AI-обработкой."

        return content, None

    except subprocess.TimeoutExpired:
        return text, f"Ошибка AI: превышено время ожидания ({timeout} с)."
    except FileNotFoundError:
        return text, "Ошибка AI: утилита z-ai не найдена в системе."
    except Exception as e:
        return text, f"Ошибка AI: {e}"
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass
