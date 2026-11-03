# -*- coding: utf-8 -*-
"""
scanner.py — обработка изображений документов (v3, максимальное качество).

База перенесена из оригинального Flask-приложения
Smart Document Scanner & AI Text Enhancer (app-ocr.py):

  * order_points()          — упорядочивание 4 углов документа
  * four_point_transform()  — перспективная коррекция («вид сканера»)
  * process_document()      — поиск контура документа + фильтр
  * perform_ocr()           — распознавание текста (Tesseract)

Улучшения v3 (по отзывам пользователя):
  * ВЫРАВНИВАНИЕ: поиск контура по 3 вариантам детекции краёв + отбор
    лучшего четырёхугольника по площади/прямоугольности/углам;
    после перспективы — финишный поворот, чтобы текст был строго
    горизонтален. Если границы не найдены — выравнивается наклон
    текста поворотом (scanned="deskewed").
  * ЧИСТОТА: нормализация освещения (убирает тени/пересветы),
    шумоподавление, фильтр «плавающие точки» (морфология + медиана),
    сглаживание краёв — результат без мусорных точек, ровный и чёткий.
  * ОПРЕДЕЛЕНИЕ ТИПА: страница или таблица (более строгие критерии
    сетки — меньше ложных срабатываний).
  * БЕЗ AI: отдаётся чистый OCR-текст, как в веб-версии.
"""
import os
import math
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


# --- Выравнивание: поиск контура документа (v3) ------------------------------

def _corner_angles(quad):
    """Углы (в градусах) четырёх вершин четырёхугольника."""
    angs = []
    for i in range(4):
        p0 = quad[i]
        p1 = quad[(i - 1) % 4]
        p2 = quad[(i + 1) % 4]
        v1 = p0 - p1
        v2 = p0 - p2
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        if n < 1e-9:
            return [0.0] * 4
        angs.append(float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))))
    return angs


def _quad_quality(quad, work_area):
    """Оценка качества четырёхугольника-кандидата в документ.
    Чем ближе к прямоугольнику с углами 90°, тем выше оценка."""
    area = cv2.contourArea(quad.astype(np.float32))
    if area <= 0:
        return 0.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(quad.astype(np.float32))
    rect_area = max(float(rw) * float(rh), 1.0)
    rectangularity = area / rect_area          # 1.0 = идеальный прямоугольник
    if rectangularity < 0.70:
        return 0.0
    area_frac = area / float(work_area)
    angs = _corner_angles(quad)
    max_dev = max(abs(a - 90.0) for a in angs)
    if max_dev > 38.0:
        return 0.0
    angle_score = 1.0 - max_dev / 45.0
    sides = sorted(float(np.linalg.norm(quad[i] - quad[(i + 1) % 4])) for i in range(4))
    if sides[0] < 0.20 * sides[3]:             # вырожденный (игла/треугольник)
        return 0.0
    return area_frac * (rectangularity ** 1.5) * angle_score


def find_document_contour(image_cv):
    """Ищет 4-угольный контур документа.

    v3: пробует несколько вариантов детекции краёв (авто-Canny с
    расширением, классический Canny, адаптивная бинаризация) и выбирает
    лучший четырёхугольник по площади, прямоугольности и углам.
    Возвращает точки в масштабе оригинала либо None.
    """
    work_h = 700  # рабочее разрешение выше, чем в оригинале (было 500) — точнее углы
    ratio = image_cv.shape[0] / float(work_h)
    work = imutils.resize(image_cv, height=work_h)
    if work.size == 0:
        return None
    wh, ww = work.shape[:2]
    work_area = float(wh * ww)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    b3 = cv2.GaussianBlur(gray, (3, 3), 0)
    b5 = cv2.GaussianBlur(gray, (5, 5), 0)

    # Вариант 1: авто-Canny по медиане + расширение (соединяет разрывы границ)
    v = float(np.median(b5))
    lo, hi = int(max(0, 0.67 * v)), int(min(255, 1.33 * v))
    edged1 = cv2.dilate(cv2.Canny(b5, lo, hi), np.ones((3, 3), np.uint8))
    # Вариант 2: классические пороги (как в оригинале)
    edged2 = cv2.dilate(cv2.Canny(b3, 75, 200), np.ones((3, 3), np.uint8))
    # Вариант 3: адаптивная бинаризация как «края» (помогает при слабом контрасте)
    edged3 = cv2.adaptiveThreshold(
        b3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8)

    best_quad, best_score = None, 0.0
    for edged in (edged1, edged2, edged3):
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if not cnts:
            continue
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:12]
        for c in cnts:
            if cv2.contourArea(c) < work_area * 0.08:
                continue
            peri = cv2.arcLength(c, True)
            # перебираем от мелкого эпсилон к крупному: мелкий даёт точные углы,
            # крупный — помогает, если граница «рваная»; оценку делает _quad_quality
            for eps in (0.008, 0.012, 0.02, 0.03, 0.045):
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                quad = approx.reshape(4, 2).astype("float32")
                score = _quad_quality(quad, work_area)
                if score > best_score:
                    best_score, best_quad = score, quad
    if best_quad is None or best_score <= 0.0:
        return None
    return best_quad * ratio


# --- Выравнивание: поворот по наклону текста (deskew) ------------------------

def _ink_mask(image_cv):
    """Бинаризация: чернила -> белый (255)."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if image_cv.ndim == 3 else image_cv
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def estimate_skew_angle(image_cv, max_angle=15.0):
    """Оценка угла наклона текста в градусах.

    Положительный угол = текст наклонён по часовой стрелке.
    Сначала ищутся длинные линии (таблицы, рамки) — по ним угол самый
    надёжный; если линий нет — проекционный метод по строкам текста.
    """
    h, w = image_cv.shape[:2]
    if h < 80 or w < 80:
        return 0.0
    try:
        bw = _ink_mask(image_cv)
    except cv2.error:
        return 0.0

    # 1) длинные линии через Hough
    edges = cv2.Canny(cv2.GaussianBlur(bw, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150,
                            minLineLength=max(80, w // 3), maxLineGap=25)
    angs = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            a = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
            if abs(a) <= 25.0:
                angs.append(a)
    if len(angs) >= 4:
        ang = float(np.median(angs))
        return ang if 0.25 < abs(ang) <= max_angle else 0.0

    # 2) проекционный метод: перебираем углы, ищем максимум «строчности»
    scale = 500.0 / float(h)
    small = cv2.resize(bw, (max(60, int(w * scale)), 500), interpolation=cv2.INTER_AREA)
    small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)[1]
    hh, ww = small.shape
    center = (ww / 2.0, hh / 2.0)

    def _score(angle_deg):
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rot = cv2.warpAffine(small, M, (ww, hh), flags=cv2.INTER_NEAREST, borderValue=0)
        proj = rot.astype(np.float32).sum(axis=1)
        return float(np.sum(np.diff(proj) ** 2))

    best, best_score = 0.0, _score(0.0)
    for cand in np.arange(-8.0, 8.01, 0.5):
        s = _score(float(cand))
        if s > best_score:
            best_score, best = s, float(cand)
    for cand in np.arange(best - 0.5, best + 0.55, 0.1):
        s = _score(float(cand))
        if s > best_score:
            best_score, best = s, float(cand)
    if abs(best) < 0.25 or abs(best) > max_angle:
        return 0.0
    return best


def deskew_image(image_cv, max_angle=12.0):
    """Поворот изображения так, чтобы текст был строго горизонтален.
    Возвращает (изображение, применённый_угол)."""
    ang = estimate_skew_angle(image_cv, max_angle=max_angle)
    if ang == 0.0:
        return image_cv, 0.0
    h, w = image_cv.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int(h * sin + w * cos) + 2
    nH = int(h * cos + w * sin) + 2
    M[0, 2] += nW / 2.0 - w / 2.0
    M[1, 2] += nH / 2.0 - h / 2.0
    border = (255, 255, 255) if image_cv.ndim == 3 else 255
    rot = cv2.warpAffine(image_cv, M, (nW, nH), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return rot, ang


# --- Качество: нормализация освещения, шумоподавление, сглаживание ----------

def _is_inner_rectangle(image_cv, pts):
    """Похож ли найденный 4-угольник на ВНУТРЕННИЙ элемент листа (таблицу,
    рамку, картинку), а не на границы документа?

    Признаки: кадр по краям светлый (лист бумаги занимает весь кадр),
    а четырёхугольник покрывает меньше половины кадра. Тогда обрезать
    нельзя — иначе потеряем текст вокруг таблицы."""
    if pts is None or image_cv is None or image_cv.size == 0:
        return False
    h, w = image_cv.shape[:2]
    try:
        quad_area = float(cv2.contourArea(pts.astype(np.float32)))
    except cv2.error:
        return False
    if quad_area <= 0:
        return False
    if quad_area / float(w * h) >= 0.55:
        return False  # занимает большую часть кадра — это документ
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    t = max(2, int(min(h, w) * 0.05))
    ring = np.concatenate([
        gray[:t, :].ravel(), gray[-t:, :].ravel(),
        gray[:, :t].ravel(), gray[:, -t:].ravel(),
    ])
    return bool(float(np.mean(ring)) > 165.0)

def _inset_quad(pts, frac=0.025):
    """Слегка поджимает четырёхугольник к центру (~1% меньшей стороны).

    Контур ищется по внешнему краю линии границы; без поджатия по краям
    скана остаются тонкие тёмные полосы фона и «мохнатые» края бумаги."""
    rect = order_points(np.asarray(pts, dtype="float32"))
    center = rect.mean(axis=0)
    min_side = min(float(np.linalg.norm(rect[i] - rect[(i + 1) % 4])) for i in range(4))
    d = frac * min_side
    out = rect.copy()
    for i in range(4):
        v = center - rect[i]
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            out[i] = rect[i] + v * (d / n)
    return out


def _trim_dark_margins(image_cv, max_trim_frac=0.10):
    """Обрезает со сторон выровненного скана полосы, где явно виден тёмный фон
    (остатки стола/фона из-за неточности контура). Срезаются только стороны,
    где БОЛЬШИНСТВО пикселей тёмные; контент с тёмными элементами не трогаем."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if image_cv.ndim == 3 else image_cv
    h, w = gray.shape[:2]
    light = gray > 170
    top = bottom = left = right = 0
    max_t = int(h * max_trim_frac)
    max_s = int(w * max_trim_frac)
    while top < max_t and light[top, :].mean() < 0.60:
        top += 1
    while bottom < max_t and light[h - 1 - bottom, :].mean() < 0.60:
        bottom += 1
    while left < max_s and light[:, left].mean() < 0.60:
        left += 1
    while right < max_s and light[:, w - 1 - right].mean() < 0.60:
        right += 1
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image_cv
    return image_cv[top:h - bottom if bottom else h,
                    left:w - right if right else w]


def _odd_kernel(min_dim):
    """Ядро для medianBlur: нечётное, адаптивное к размеру изображения."""
    if min_dim >= 160:
        return 35
    if min_dim >= 80:
        return 21
    if min_dim >= 40:
        return 15
    return max(3, min_dim | 1)


def enhance_gray(gray):
    """Улучшение серого изображения: ровный свет + без шума + чётко.
    Убирает тени и неравномерную засветку, плавающие точки и мусор."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # 1) Нормализация освещения: делим на размытый фон (убирает тени)
    bg = cv2.medianBlur(gray, _odd_kernel(min(h, w)))
    norm = cv2.divide(gray, bg, scale=255)
    # 2) Шумоподавление (дорогой фильтр — только для разумных размеров)
    if max(h, w) <= 2200:
        norm = cv2.fastNlMeansDenoising(norm, None, 8, 7, 21)
    else:
        norm = cv2.bilateralFilter(norm, 7, 50, 50)
    # 3) Лёгкая резкость (unsharp mask)
    blur = cv2.GaussianBlur(norm, (0, 0), 2.0)
    sharp = cv2.addWeighted(norm, 1.3, blur, -0.3, 0)
    return sharp


def enhance_bgr(bgr):
    """Улучшение цветного изображения: сглаживание шума при чётких краях."""
    h, w = bgr.shape[:2]
    if max(h, w) <= 2200:
        den = cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)
    else:
        den = cv2.bilateralFilter(bgr, 7, 60, 60)
    blur = cv2.GaussianBlur(den, (0, 0), 2.0)
    return cv2.addWeighted(den, 1.25, blur, -0.25, 0)


def apply_filter(warped, selected_filter):
    """Применяет выбранный фильтр (v3: чисто и сглажено, без точек)."""
    if selected_filter == "color":
        return enhance_bgr(warped)

    gray = enhance_gray(warped)

    if selected_filter == "grayscale":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # adaptive_threshold — «Ч/Б документ»
    h, w = gray.shape[:2]
    block = max(31, min(71, (min(h, w) // 18) | 1))  # адаптивный размер окна
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 12)
    # медиана убирает одиночные «плавающие» точки шума
    bw = cv2.medianBlur(bw, 3)
    # морфологическое открытие чернил: удаляет мусор мельче штриха
    ink = cv2.bitwise_not(bw)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    bw = cv2.bitwise_not(ink)
    # сглаживание «рваных» краёв (лёгкий антиалиасинг)
    bw = cv2.GaussianBlur(bw, (3, 3), 0)
    _, bw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


# --- Core Image Processing Function ------------------------------------------
def process_document(image_cv, selected_filter="adaptive_threshold", allow_fallback=True):
    """Находит контур документа, выравнивает перспективу, применяет фильтр.

    Возвращает (processed_image, scanned, error):
      processed_image — np.ndarray (BGR) либо None при ошибке
      scanned         — True, если перспектива выровнена;
                        "deskewed", если границы не найдены, но наклон
                        текста выровнен поворотом;
                        False, если обработан весь кадр как есть
      error           — текст ошибки либо None
    """
    if image_cv is None or image_cv.size == 0:
        return None, False, "Входное изображение пустое."

    if selected_filter not in FILTER_KEYS:
        selected_filter = "adaptive_threshold"

    orig = image_cv.copy()
    pts = find_document_contour(image_cv)
    # Если «документ» — это на самом деле таблица/рамка на листе, который
    # занимает весь кадр, обрезаем НЕЛЬЗЯ: обработаем весь лист целиком.
    if pts is not None and _is_inner_rectangle(image_cv, pts):
        pts = None
    if pts is not None:
        # поджимаем контур внутрь: без этого по краям остаются полосы фона
        pts = _inset_quad(pts)

    if pts is not None:
        try:
            warped = four_point_transform(orig, pts)
            # финишное выравнивание: убираем остаточный наклон текста
            warped, _ = deskew_image(warped, max_angle=8.0)
            # срезаем остатки тёмного фона по краям (неточность контура)
            warped = _trim_dark_margins(warped)
        except Exception as e:
            return None, False, f"Ошибка перспективного преобразования: {e}."
        if warped is None or warped.size == 0:
            return None, False, "Перспективное преобразование дало пустое изображение."
        scanned = True
    else:
        # Границы документа не найдены. В оригинале это ошибка, но для
        # удобства выравниваем наклон текста и обрабатываем весь кадр.
        if not allow_fallback:
            return None, False, "Не удалось найти 4-угольный контур документа."
        warped, ang = deskew_image(orig, max_angle=14.0)
        scanned = "deskewed" if ang else False

    try:
        processed_output = apply_filter(warped, selected_filter)
    except cv2.error as e:
        return None, scanned, f"Ошибка OpenCV при наложении фильтра: {e}"

    if processed_output is None or processed_output.size == 0:
        return None, scanned, "Обработка не дала итогового изображения."

    return processed_output, scanned, None


# --- OCR Function (из оригинала, расширен языками) ----------------------------
def prepare_ocr_image(image_cv):
    """Подготовка изображения к OCR: серое, достаточно крупное, чистое."""
    if image_cv.ndim == 3 and image_cv.shape[2] == 3:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_cv.copy() if image_cv.ndim == 2 else cv2.cvtColor(image_cv, cv2.COLOR_BGRA2GRAY)
    h, w = gray.shape[:2]
    short = min(h, w)
    if short < 1200:  # мелкие фото увеличиваем — OCR заметно точнее
        scale = 1200.0 / float(short)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = enhance_gray(gray)
    return gray


def perform_ocr(image_cv, lang=None):
    """Performs OCR on an OpenCV image using Tesseract."""
    lang = lang or OCR_LANG
    if image_cv is None or image_cv.size == 0:
        return None, "Входное изображение для OCR пустое."
    try:
        gray_for_ocr = prepare_ocr_image(image_cv)
        text = pytesseract.image_to_string(gray_for_ocr, lang=lang)
        return text, None
    except pytesseract.TesseractNotFoundError:
        error_msg = "Ошибка Tesseract: исполняемый файл не найден. Установите tesseract-ocr."
        return None, error_msg
    except Exception as e:
        return None, f"Ошибка OCR: {str(e)}"


# --- Определение типа содержимого: страница или таблица -----------------------

def _binarize_ink(image_cv):
    """Бинаризация: чернила -> белый (255)."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _cluster_positions(proj, min_val, min_gap):
    """Кластеризует координаты линий из проекции, возвращает центры кластеров."""
    idx = np.where(proj >= min_val)[0]
    if len(idx) == 0:
        return []
    clusters = []
    start = prev = int(idx[0])
    for p in idx[1:]:
        p = int(p)
        if p - prev > min_gap:
            clusters.append((start + prev) // 2)
            start = p
        prev = p
    clusters.append((start + prev) // 2)
    return clusters


def _slanted_line_kernels(length, horizontal, angles=range(-4, 5)):
    """Набор структурных элементов «линия» с небольшими наклонами.

    Строгое вертикальное/горизонтальное ядро не выдерживает даже 0.2–0.5°
    остаточного наклона (линия уползает из колонки за длину ядра).
    Перебор наклонов ±4° делает поиск линий устойчивым к малым перекосам."""
    L = int(max(3, length))
    kernels = []
    for a in angles:
        t = int(round(math.tan(math.radians(a)) * L / 2.0))
        k = np.zeros((L, L), np.uint8)
        if horizontal:
            cv2.line(k, (0, L // 2 - t), (L - 1, L // 2 + t), 1, 1)
        else:
            cv2.line(k, (L // 2 - t, L - 1), (L // 2 + t, 0), 1, 1)
        kernels.append(k)
    return kernels


def _open_lines(bw, length, horizontal):
    """Морфологическое открытие длинными линиями, устойчивое к наклону."""
    out = None
    for k in _slanted_line_kernels(length, horizontal):
        op = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
        out = op if out is None else cv2.bitwise_or(out, op)
    return out


def _line_components(mask, min_len, vertical):
    """Координаты линий на маске как связных компонент.

    Компонентный подход (в отличие от проекций) не «размазывает»
    наклонные линии: каждая линия — одна компонента с центроидом.
    Дубликаты одной линии (линию могли захватить несколько наклонных
    ядер) сливаются по перекрытию диапазонов координат.
    vertical=True -> ищем вертикальные линии (высота >= min_len),
    иначе горизонтальные (ширина >= min_len).
    """
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for i in range(1, n):
        left = int(stats[i, cv2.CC_STAT_LEFT])
        top = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        if vertical:
            if ch >= min_len and cw <= max(6, min_len // 3):
                # [начало, конец диапазона X, центроид X, длина линии]
                comps.append([left, left + cw, float(cents[i][0]), ch])
        else:
            if cw >= min_len and ch <= max(6, min_len // 3):
                comps.append([top, top + ch, float(cents[i][1]), cw])
    comps.sort(key=lambda t: t[0])
    merged = []
    for c in comps:
        if merged and c[0] <= merged[-1][1] + 3:
            m = merged[-1]
            # та же линия: объединяем диапазон, центроид взвешиваем длиной
            m[1] = max(m[1], c[1])
            m[2] = (m[2] * m[3] + c[2] * c[3]) / (m[3] + c[3])
            m[3] = max(m[3], c[3])
        else:
            merged.append(list(c))
    return [m[2] for m in merged]


def _snap_line(bw_full, coord, vertical, half=4):
    """Уточняет координату линии по максимуму чернил в полном разрешении."""
    h, w = bw_full.shape[:2]
    best, best_val = coord, -1
    for c in range(max(0, coord - half), min((h if vertical else w), coord + half + 1)):
        if vertical:
            val = int((bw_full[:, max(0, c - 1):c + 2] > 0).sum())
        else:
            val = int((bw_full[max(0, c - 1):c + 2, :] > 0).sum())
        if val > best_val:
            best_val, best = val, c
    return best


def detect_content_type(image_cv):
    """Определяет, что на изображении: страница с текстом или таблица.

    Таблица ищется по сетке длинных горизонтальных и вертикальных линий
    (морфологическое открытие с допуском наклона ±4°) и их пересечениям.
    Поиск линий идёт на уменьшенной копии (скорость), координаты
    уточняются на полном разрешении. Критерии строгие: линии длинные,
    узлов много, сетка занимает заметную часть листа.

    Возвращает dict:
      is_table        — True, если обнаружена таблица
      rows, cols      — число строк и столбцов (для таблицы)
      h_lines, v_lines— координаты линий сетки (y и x соответственно)
    """
    result = {"is_table": False, "rows": 0, "cols": 0, "h_lines": [], "v_lines": []}
    if image_cv is None or image_cv.size == 0:
        return result
    h, w = image_cv.shape[:2]
    try:
        bw_full = _binarize_ink(image_cv)
    except cv2.error:
        return result
    # Небольшое расширение перекрывает разрывы линий (шум, пережатый JPEG)
    bw_full = cv2.dilate(bw_full, np.ones((3, 3), np.uint8), iterations=1)

    # Рабочая копия для быстрого поиска линий
    scale = min(1.0, 800.0 / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(bw_full, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        bw = cv2.threshold(small, 40, 255, cv2.THRESH_BINARY)[1]
        bw = cv2.dilate(bw, np.ones((2, 2), np.uint8), iterations=1)
    else:
        bw = bw_full
    sh, sw = bw.shape[:2]

    horiz = _open_lines(bw, max(20, sw // 8), horizontal=True)
    vert = _open_lines(bw, max(20, sh // 8), horizontal=False)

    # Горизонтальные линии: длина >= трети ширины листа
    h_lines_s = _line_components(horiz, min_len=max(16, sw // 3), vertical=False)
    if len(h_lines_s) < 3:
        return result  # для таблицы нужно >= 2 строк
    # Вертикальные линии: длина >= половины высоты таблицы (адаптивно)
    th_s = h_lines_s[-1] - h_lines_s[0]
    v_lines_s = _line_components(vert, min_len=max(16, int(0.5 * th_s)), vertical=True)
    if len(v_lines_s) < 3:
        return result  # нужно >= 2 столбцов

    # Проверяем пересечения сетки (узлы): их должно быть большинство
    grid = cv2.bitwise_and(horiz, vert)
    n_nodes = cv2.connectedComponents(grid)[0] - 1
    if n_nodes < max(6, int(0.45 * len(h_lines_s) * len(v_lines_s))):
        return result  # узлов мало — это не сетка таблицы

    # Сетка должна занимать заметную часть листа и быть достаточно широкой
    tw_s = v_lines_s[-1] - v_lines_s[0]
    if tw_s < 0.25 * sw or tw_s * th_s < 0.06 * float(sw * sh):
        return result

    # Координаты: масштаб оригинала + уточнение по полному разрешению
    inv = 1.0 / scale
    h_lines = [_snap_line(bw_full, int(round(c * inv)), vertical=False)
               for c in h_lines_s]
    v_lines = [_snap_line(bw_full, int(round(c * inv)), vertical=True)
               for c in v_lines_s]

    result["is_table"] = True
    result["rows"] = len(h_lines) - 1
    result["cols"] = len(v_lines) - 1
    result["h_lines"] = h_lines
    result["v_lines"] = v_lines
    return result


def remove_grid_lines(image_cv):
    """Убирает длинные линии сетки (таблицы), оставляя текст.

    Используется в режиме «📄 Весь лист», когда на странице есть таблица:
    без линий Tesseract читает и текст вокруг таблицы, и содержимое ячеек.
    Поиск линий идёт на уменьшенной копии (быстро), маска линий
    масштабируется обратно и слегка расширяется.
    """
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if image_cv.ndim == 3 else image_cv
    h, w = gray.shape[:2]
    _, bw_full = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    scale = min(1.0, 800.0 / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(bw_full, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        bw = cv2.threshold(small, 40, 255, cv2.THRESH_BINARY)[1]
        bw = cv2.dilate(bw, np.ones((2, 2), np.uint8), iterations=1)
    else:
        bw = bw_full

    lines = cv2.bitwise_or(
        _open_lines(bw, max(20, bw.shape[1] // 8), horizontal=True),
        _open_lines(bw, max(20, bw.shape[0] // 8), horizontal=False),
    )
    if scale < 1.0:
        lines = cv2.resize(lines, (w, h), interpolation=cv2.INTER_NEAREST)
    # расширяем маску линий, чтобы захватить кромку букв, прилипших к линиям
    lines = cv2.dilate(lines, np.ones((5, 5), np.uint8), iterations=1)
    cleaned = np.where(lines > 0, 255, gray).astype(np.uint8)
    if image_cv.ndim == 3:
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    return cleaned


def _ocr_cell(cell_bgr, lang):
    """OCR одной ячейки таблицы, возвращает однострочный текст."""
    if cell_bgr is None or cell_bgr.size == 0:
        return ""
    img = cell_bgr
    ch, cw = img.shape[:2]
    # Мелкие ячейки увеличиваем для надёжного распознавания
    if ch < 28 or cw < 60:
        scale = max(2.0, 28.0 / max(ch, 1), 60.0 / max(cw, 1))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Белые поля вокруг ячейки улучшают распознавание
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    try:
        text = pytesseract.image_to_string(gray, lang=lang, config="--psm 6")
        # случайно попавший в срез обрезок линии сетки OCR читает как «|» — убираем
        return " ".join(text.replace("|", " ").split())
    except Exception:
        return ""


def extract_table_text(image_cv, h_lines, v_lines, max_cells=160, lang=None):
    """OCR таблицы по ячейкам.

    Возвращает текст вида:
        Название | Кол-во | Цена
        Стол     | 2      | 5000
    Каждая строка таблицы — отдельная строка текста (как в документе).
    """
    lang = lang or OCR_LANG
    # отступ от линий сетки: край линии не должен попадать в срез ячейки
    margin = 5
    lines_out = []
    used = 0
    for r in range(len(h_lines) - 1):
        y1 = h_lines[r] + margin
        y2 = h_lines[r + 1] - margin
        if y2 - y1 < 10:
            continue
        cells = []
        for c in range(len(v_lines) - 1):
            x1 = v_lines[c] + margin
            x2 = v_lines[c + 1] - margin
            if x2 - x1 < 10:
                cells.append("")
                continue
            used += 1
            if used > max_cells:
                cells.append("")
                continue
            cell = image_cv[y1:y2, x1:x2]
            cells.append(_ocr_cell(cell, lang))
        lines_out.append(" | ".join(cells))
    return "\n".join(lines_out)
