# -*- coding: utf-8 -*-
"""
scanner.py — обработка изображений документов.

Функции перенесены 1-в-1 из оригинального Flask-приложения
Smart Document Scanner & AI Text Enhancer (app-ocr.py):

  * order_points()          — упорядочивание 4 углов документа
  * four_point_transform()  — перспективная коррекция («вид сканера»)
  * process_document()      — поиск контура документа + фильтр
  * perform_ocr()           — распознавание текста (Tesseract)

Отличия от оригинала:
  * языки OCR: rus+eng (авто-определение) вместо только eng
  * если границы документа не найдены — применяется фильтр
    ко всему кадру (флаг scanned=False), бот не падает с ошибкой
"""
import os
import cv2
import numpy as np
import imutils
import pytesseract

# --- Tesseract: локальные языковые данные (rus + eng) -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TESSDATA = os.path.join(BASE_DIR, "tessdata")
if os.path.isdir(LOCAL_TESSDATA):
    os.environ["TESSDATA_PREFIX"] = LOCAL_TESSDATA

OCR_LANG = os.environ.get("OCR_LANG", "rus+eng")  # авто-определение: русский + английский

# --- Фильтры (как в оригинальном веб-интерфейсе) -----------------------------
FILTER_KEYS = ("adaptive_threshold", "grayscale", "color")

FILTER_LABELS_RU = {
    "adaptive_threshold": "Ч/Б документ",
    "grayscale": "Серый",
    "color": "Цвет",
}


# --- Helper Functions (Image Processing) — из оригинала ----------------------
def order_points(pts):
    """ Orders points: top-left, top-right, bottom-right, bottom-left """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """ Applies perspective transform to obtain bird's-eye view """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def find_document_contour(image_cv):
    """Ищет 4-угольный контур документа. Возвращает точки в масштабе
    оригинала либо None (адаптировано из process_document оригинала)."""
    ratio = image_cv.shape[0] / 500.0
    image_resized = imutils.resize(image_cv, height=500)
    if image_resized.size == 0:
        return None

    # Preprocessing for contour detection
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            # Basic area check to avoid tiny contours
            if cv2.contourArea(approx) > (image_resized.shape[0] * image_resized.shape[1] * 0.05):
                return approx.reshape(4, 2) * ratio
    return None


def apply_filter(warped, selected_filter):
    """Применяет выбранный фильтр (как в оригинале)."""
    if selected_filter == "adaptive_threshold":
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        processed_output = cv2.adaptiveThreshold(
            warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 10)
        processed_output = cv2.cvtColor(processed_output, cv2.COLOR_GRAY2BGR)
    elif selected_filter == "grayscale":
        processed_output = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        processed_output = cv2.cvtColor(processed_output, cv2.COLOR_GRAY2BGR)
    elif selected_filter == "color":
        processed_output = warped
    else:
        processed_output = warped  # Default to color
    return processed_output


# --- Core Image Processing Function ------------------------------------------
def process_document(image_cv, selected_filter="adaptive_threshold", allow_fallback=True):
    """Находит контур документа, выравнивает перспективу, применяет фильтр.

    Возвращает (processed_image, scanned, error):
      processed_image — np.ndarray (BGR) либо None при ошибке
      scanned         — True, если перспектива выровнена; False, если
                        обработан весь кадр (границы не найдены)
      error           — текст ошибки либо None
    """
    if image_cv is None or image_cv.size == 0:
        return None, False, "Входное изображение пустое."

    if selected_filter not in FILTER_KEYS:
        selected_filter = "adaptive_threshold"

    orig = image_cv.copy()
    pts = find_document_contour(image_cv)

    if pts is not None:
        try:
            warped = four_point_transform(orig, pts)
        except Exception as e:
            return None, False, f"Ошибка перспективного преобразования: {e}."
        if warped is None or warped.size == 0:
            return None, False, "Перспективное преобразование дало пустое изображение."
        scanned = True
    else:
        # Границы документа не найдены: как в оригинале это ошибка,
        # но для удобства в боте обрабатываем всё изображение целиком.
        if not allow_fallback:
            return None, False, "Не удалось найти 4-угольный контур документа."
        warped = orig
        scanned = False

    try:
        processed_output = apply_filter(warped, selected_filter)
    except cv2.error as e:
        return None, scanned, f"Ошибка OpenCV при наложении фильтра: {e}"

    if processed_output is None or processed_output.size == 0:
        return None, scanned, "Обработка не дала итогового изображения."

    return processed_output, scanned, None


# --- OCR Function (из оригинала, расширен языками) ----------------------------
def perform_ocr(image_cv, lang=None):
    """Performs OCR on an OpenCV image using Tesseract."""
    lang = lang or OCR_LANG
    if image_cv is None or image_cv.size == 0:
        return None, "Входное изображение для OCR пустое."
    try:
        # Tesseract предпочитает grayscale
        if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
            gray_for_ocr = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray_for_ocr = image_cv

        text = pytesseract.image_to_string(gray_for_ocr, lang=lang)
        return text, None
    except pytesseract.TesseractNotFoundError:
        error_msg = "Ошибка Tesseract: исполняемый файл не найден. Установите tesseract-ocr."
        return None, error_msg
    except Exception as e:
        return None, f"Ошибка OCR: {str(e)}"
