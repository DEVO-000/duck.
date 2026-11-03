# -*- coding: utf-8 -*-
"""
scanner.py — обработка изображений документов (v4, «идеальный сканер»).

Наследует базу v3 (перспективная коррекция, фильтры, OCR, таблицы) и
добавляет:

  * EXIF-корректное чтение (decode_image_bytes) — фото с телефона
    больше не приходят повёрнутыми из-за метаданных;
  * АВТООПРЕДЕЛЕНИЕ ОРИЕНТАЦИИ 0/90/180/270:
      - ось текста (горизонталь/вертикаль) по форме слов-блобов
        + пиковости проекции чернил;
      - «вверх или вниз» — скоринг слов Tesseract (только слова,
        содержащие кириллицу; для англ. документов — резервный
        англ. скоринг);
  * ДОВОРОТ ДО СТРОГО ГОРИЗОНТАЛЬНОГО ТЕКСТА с точностью до 0.05°
    (линии Хафа + проекционный метод, полярно-инвариантная маска
    чернил — работает и для мела на доске);
  * ВЫПРАВЛЕНИЕ ДЕФОРМАЦИИ (изгиба) страницы: полосовая взаимная
    корреляция проекций чернил -> кривизна -> обратный remap.
    Изогнутые строки текста становятся идеально ровными;
  * ОБРЕЗКА ОСТАТКОВ ФОНА с учётом полярности (светлая страница /
    тёмная доска);
  * КАЧЕСТВО: нормализация освещения + CLAHE + двухмасштабное
    повышение резкости; Ч/Б-фильтр без «плавающих точек»
    (компонентный анализ вместо морфологии, тонкие штрихи
    и точки «ё» не съедаются); авто-инверсия тёмных документов
    (мел на доске -> чёрный текст на белом).

Публичный API:
  decode_image_bytes(data) -> np.ndarray BGR
  process_document(image_cv, selected_filter, allow_fallback=True)
      -> (processed, scanned, error, info)
  perform_ocr(image_cv, lang=None)
  detect_content_type(image_cv) / extract_table_text(...) / remove_grid_lines(...)
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

_CYR = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")


# ==============================================================================
# Чтение изображения (EXIF-aware)
# ==============================================================================
def decode_image_bytes(data):
    """Декодирует PNG/JPEG/WEBP из байтов с учётом EXIF-ориентации.

    cv2.imdecode игнорирует EXIF Orientation — фото с телефона могли
    прийти «налёжу». Здесь поворот применяется по метаданным."""
    if not data:
        return None
    try:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
    except Exception:
        return None
    # EXIF Orientation (0x0112 = 274)
    try:
        from PIL import Image
        import io
        with Image.open(io.BytesIO(data)) as pil:
            exif = pil.getexif()
            orient = exif.get(274, 1) if exif else 1
        if orient in (2, 4):
            img = cv2.flip(img, 1)
        if orient in (3, 4):
            img = cv2.rotate(img, cv2.ROTATE_180)
        if orient in (5, 6):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if orient in (7, 8):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if orient in (5, 7):
            img = cv2.flip(img, 1)
    except Exception:
        pass
    return img


# ==============================================================================
# Маски чернил
# ==============================================================================
def _ink_mask(image_cv):
    """Бинаризация: тёмные чернила -> белый (255). Для светлой бумаги."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if image_cv.ndim == 3 else image_cv
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _ink_mask_minority(gray):
    """Маска «меток текста» независимо от полярности.

    Чернила/мел = класс меньшинства: для светлой бумаги это тёмные
    пиксели, для чёрной доски — светлые (мел)."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if (bw > 0).mean() > 0.5:
        bw = cv2.bitwise_not(bw)
    return bw


# ==============================================================================
# 4-точечная перспективная коррекция
# ==============================================================================
def order_points(pts):
    """ Orders points: top-left, top-right, bottom-right, bottom-left.

    v4: сортировка по углу вокруг центроида с опорным направлением
    «верх-лево» (-135°). Классическая эвристика sum/diff ОШИБАЕТСЯ на
    портретных прямоугольниках (меняет местами tr и bl) — из-за этого
    перспективное преобразование «транспонировало» кадр. """
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])  # y вниз
    ref = math.atan2(-1.0, -1.0)  # -135°: направление к верхнему-левому углу
    rel = np.mod(angles - ref, 2.0 * np.pi)
    order = np.argsort(rel)
    rect = pts[order].astype("float32")
    return rect


def four_point_transform(image, pts):
    """ Applies perspective transform to obtain bird's-eye view """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)   # нижняя сторона
    widthB = np.linalg.norm(tr - tl)   # верхняя сторона
    maxWidth = max(int(widthA), int(widthB))
    # v4.1: ВЫСОТА — это БОКОВЫЕ стороны (|tr-br|, |tl-bl|), а не диагонали!
    # Прежний вариант (|tr-bl|, |tl-br|) растягивал документ по вертикали
    # на фактор «диагональ/высота» (~1.2–1.25x) — вносил деформацию вместо
    # её исправления и портил автоопределение ориентации.
    heightA = np.linalg.norm(tr - br)  # правая сторона
    heightB = np.linalg.norm(tl - bl)  # левая сторона
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))
    return warped


# ==============================================================================
# Поиск границ документа (v4: края + яркостный компонент)
# ==============================================================================
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


def _iter_quad_candidates(mask, work_area, top=12):
    """Перебирает контуры маски и отдаёт (score, quad) лучших 4-угольников."""
    cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        return []
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:top]
    out = []
    for c in cnts:
        if cv2.contourArea(c) < work_area * 0.08:
            continue
        peri = cv2.arcLength(c, True)
        # от мелкого эпсилона к крупному: мелкий даёт точные углы,
        # крупный помогает, если граница «рваная»
        for eps in (0.008, 0.012, 0.02, 0.03, 0.045):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = approx.reshape(4, 2).astype("float32")
            score = _quad_quality(quad, work_area)
            if score > 0.0:
                out.append((score, quad))
    return out


def _edge_masks(work):
    """Три варианта карты краёв (v3): авто-Canny, классика, адаптивная порог."""
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    b3 = cv2.GaussianBlur(gray, (3, 3), 0)
    b5 = cv2.GaussianBlur(gray, (5, 5), 0)
    v = float(np.median(b5))
    lo, hi = int(max(0, 0.67 * v)), int(min(255, 1.33 * v))
    edged1 = cv2.dilate(cv2.Canny(b5, lo, hi), np.ones((3, 3), np.uint8))
    edged2 = cv2.dilate(cv2.Canny(b3, 75, 200), np.ones((3, 3), np.uint8))
    edged3 = cv2.adaptiveThreshold(
        b3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8)
    return [edged1, edged2, edged3]


def _bright_paper_mask(work):
    """Маска яркой бумаги (лист на тёмном/пёстром фоне).

    Возвращает маску замкнутой ярковой области или None, если яркостное
    разделение не информативно (например, лист занимает весь кадр)."""
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    frac = float((bw > 0).mean())
    if frac < 0.10 or frac > 0.90:
        # попробуем адаптивный вариант (неравномерное освещение)
        bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, -12)
        frac = float((bw > 0).mean())
        if frac < 0.10 or frac > 0.90:
            return None
    # закрываем дырки (текст, рисунки на листе) и склейки края
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return bw


def _largest_component_mask(bw):
    """Маска крупнейшей связной компоненты (внутри контура)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8))
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (labels == i).astype(np.uint8) * 255
    # вычищаем мелкие «островки» фона внутри листа
    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return comp


def _quad_ring_ink(bw_ink, quad, band_px):
    """Плотность чернил в тонком кольце ВНУТРИ четырёхугольника.

    Настоящие границы листа/доски окружены пустыми полями: кольцо
    сразу внутри контура почти чистое. Если же четырёхугольник — это
    внутренний «прямоугольник текста» (таблица, текстовый блок),
    кольцо пересекает строки и плотность чернил высокая."""
    # v4.1: кольцо считается между 1 и 2 полосами ВНУТРЬ от границы —
    # самая кромка (0..1%) не смотрится: там живёт тень/градиент на стыке
    # листа и фона, которая ложно засчитывалась как «чернила»
    h, w = bw_ink.shape[:2]
    q = np.asarray(quad, dtype=np.float32)
    min_side = float(min(
        np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1]),
        np.linalg.norm(q[3] - q[2]), np.linalg.norm(q[0] - q[3]),
    ))
    if min_side <= 6 * band_px:
        return 0.0
    outer = np.zeros((h, w), np.uint8)
    cv2.fillPoly(outer, [_inset_quad(q, frac=band_px / min_side).astype(np.int32)], 255)
    inner = np.zeros((h, w), np.uint8)
    cv2.fillPoly(inner, [_inset_quad(q, frac=2.0 * band_px / min_side).astype(np.int32)], 255)
    ring = cv2.subtract(outer, inner)
    area = float((ring > 0).sum())
    if area < 20.0:
        return 0.0
    return float(((ring > 0) & (bw_ink > 0)).sum()) / area


def _quad_ring_bg_mismatch(gray, quad, band_px):
    """Отличается ли кольцо внутри квада от внутренней области по яркости.

    True = кольцо похоже на ФОН, а не на лист: так ведут себя
    габаритные боксы вокруг наклонного листа (minAreaRect) — внутри
    остаются треугольники фона. У настоящего квада листа кольцо —
    пустые поля того же цвета, что и страница."""
    h, w = gray.shape[:2]
    q = np.asarray(quad, dtype=np.float32)
    min_side = float(min(
        np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1]),
        np.linalg.norm(q[3] - q[2]), np.linalg.norm(q[0] - q[3]),
    ))
    if min_side <= 10 * band_px:
        return False
    masks = []
    for f1, f2 in ((1.0, 2.0), (3.0, 9.0)):
        m_out = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m_out, [_inset_quad(q, frac=f1 * band_px / min_side).astype(np.int32)], 255)
        m_in = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m_in, [_inset_quad(q, frac=f2 * band_px / min_side).astype(np.int32)], 255)
        masks.append(cv2.subtract(m_out, m_in))
    ring, interior = masks
    if (ring > 0).sum() < 20 or (interior > 0).sum() < 20:
        return False
    med_ring = float(np.median(gray[ring > 0]))
    med_int = float(np.median(gray[interior > 0]))
    return abs(med_ring - med_int) > 35.0


def find_document_contour(image_cv):
    """Ищет 4-угольный контур документа (v4).

    Стратегии:
      1) края (3 варианта детекции, как в v3);
      2) яркая бумага: Otsu/адаптивная яркость -> крупнейшая яркая
         компонента -> выпуклая оболочка -> 4-угольник.
         Работает там, где края тонут в пёстром фоне (скатерть, ковёр).
    Возвращает точки в масштабе оригинала либо None.
    """
    work_h = 900  # выше рабочее разрешение — точнее углы (v3 было 700)
    ratio = image_cv.shape[0] / float(work_h)
    work = imutils.resize(image_cv, height=work_h)
    if work.size == 0:
        return None
    wh, ww = work.shape[:2]
    work_area = float(wh * ww)

    candidates = []
    # --- стратегия 1: края ---
    try:
        for edged in _edge_masks(work):
            candidates.extend(_iter_quad_candidates(edged, work_area))
    except cv2.error:
        pass
    # --- стратегия 2: яркая бумага ---
    try:
        paper = _bright_paper_mask(work)
        if paper is not None:
            comp = _largest_component_mask(paper)
            if comp is not None:
                found = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)[0]
                if found:
                    hull = cv2.convexHull(found[0])
                    peri = cv2.arcLength(hull, True)
                    approxed = False
                    for eps in (0.006, 0.01, 0.02, 0.035, 0.05):
                        approx = cv2.approxPolyDP(hull, eps * peri, True)
                        if len(approx) == 4 and cv2.isContourConvex(approx):
                            quad = approx.reshape(4, 2).astype("float32")
                            score = _quad_quality(quad, work_area)
                            if score > 0.0:
                                # бонус: яркостная стратегия стабильнее на пёстром фоне
                                candidates.append((score * 1.15, quad))
                            approxed = True
                            break
                    # v4.1: изогнутый край листа -> 4-точечная аппроксимация
                    # не выходит; габаритный прямоугольник бумажной области
                    # всё равно покрывает лист С полями (лучше лишний фон,
                    # чем срезанный текст)
                    rect = cv2.minAreaRect(found[0])
                    box = cv2.boxPoints(rect).astype("float32")
                    score = _quad_quality(box, work_area)
                    if score > 0.0:
                        candidates.append((score * (1.15 if not approxed else 0.95), box))
    except (cv2.error, IndexError, ValueError):
        pass

    if not candidates:
        return None
    # v4.1: штраф за «рамку режет текст» — кольцо внутри квада должно
    # быть чистым (пустые поля страницы). Ложные внутренние прямоугольники
    # (текстовый блок, таблица) этим отсекаются.
    try:
        gray_work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        ink_work = _ink_mask_minority(gray_work)
        band_px = max(3, int(0.012 * min(wh, ww)))
        adjusted = []
        for score, quad in candidates:
            ri = _quad_ring_ink(ink_work, quad, band_px)
            if ri > 0.02:
                score = score * max(0.30, 1.0 - 3.0 * ri)
            if _quad_ring_bg_mismatch(gray_work, quad, band_px):
                # габаритный бокс вокруг наклонного листа: внутри треугольники
                # фона — сильно понижаем, но не убиваем (для изогнутых краёв
                # это единственный безопасный кандидат)
                score = score * 0.55
            adjusted.append((score, quad))
        candidates = adjusted
    except (cv2.error, ValueError):
        pass

    best_score, best_quad = max(candidates, key=lambda t: t[0])
    if best_quad is None or best_score <= 0.0:
        return None
    return best_quad * ratio


# ==============================================================================
# Автоопределение ориентации (0 / 90 / 180 / 270)
# ==============================================================================
def _rot(img, deg):
    if deg == 0:
        return img
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _word_blob_axis_score(image_cv, work_h=800):
    """Ось текста по форме слов: слова — широкие капли (w/h ~ 1..14).

    Линии, рамки, крупные объекты исключаются. Возвращает площадь
    «словесных» блобов (чем больше, тем вероятнее текст горизонтален)."""
    s = work_h / image_cv.shape[0]
    small = cv2.resize(image_cv, (max(1, int(image_cv.shape[1] * s)), work_h),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bw = _ink_mask_minority(gray)
    # соединяем буквы внутри слов (слова разделяются большими пробелами)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    n, _, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8),
                                                      connectivity=8)
    H, W = bw.shape[:2]
    area = 0.0
    for i in range(1, n):
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if h < 5 or h > 0.09 * H:
            continue
        if w < 0.008 * W or w > 0.45 * W:
            continue
        ar = w / max(h, 1)
        if 1.1 <= ar <= 14.0:
            area += float(stats[i, cv2.CC_STAT_AREA])
    return area


def _proj_axis_score(image_cv, work_h=800):
    """Пиковость горизонтальной проекции чернил (полярно-инвариантная)."""
    s = work_h / image_cv.shape[0]
    small = cv2.resize(image_cv, (max(1, int(image_cv.shape[1] * s)), work_h),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bw = _ink_mask_minority(gray)
    proj = bw.astype(np.float32).sum(axis=1)
    proj = cv2.GaussianBlur(proj.reshape(-1, 1), (1, 15), 0).ravel()
    d = np.diff(proj) ** 2
    return float(np.sum(d)) / max(1.0, float(proj.sum()))


def _ocr_word_score(image_cv, deg, lang, work_h=1000, min_conf=40):
    """Скоринг ориентации: слова, распознанные Tesseract с conf>=min_conf.

    Для кириллических языков считаются только слова, содержащие
    кириллицу (перевёрнутая кириллица читается как латиница-двойник
    и не должна засчитываться). Возвращает (кол-во_символов, ср_conf)."""
    r = _rot(image_cv, deg)
    s = work_h / r.shape[0]
    small = cv2.resize(r, (max(1, int(r.shape[1] * s)), work_h),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # нормализация освещения — сильно поднимает качество распознавания
    k = 35 if min(gray.shape[:2]) >= 160 else 21
    gray = cv2.divide(gray, cv2.medianBlur(gray, k), scale=255)
    try:
        d = pytesseract.image_to_data(gray, lang=lang, config="--psm 3",
                                      output_type=pytesseract.Output.DICT)
    except Exception:
        return 0, 0.0
    words, chars, confsum = 0, 0, 0.0
    for txt, cf in zip(d["text"], d["conf"]):
        t = txt.strip().lower()
        if len(t) < 2 or sum(ch.isalnum() for ch in t) < 2:
            continue
        try:
            cfv = float(cf)
        except (TypeError, ValueError):
            continue
        if cfv < min_conf:
            continue
        if any(ch in _CYR for ch in lang) and not any(ch in _CYR for ch in t):
            continue  # слово без кириллицы не засчитываем (латиница-двойник)
        words += 1
        chars += len(t)
        confsum += cfv
    return chars, (confsum / words if words else 0.0)


def detect_orientation(image_cv):
    """Определяет, на сколько градусов повёрнут текст (0/90/180/270).

    Схема (v4.1):
      1) ось текста (0/180 vs 90/270) — блобы-слова × пиковость проекции;
      2) «вверх/вниз» внутри оси — СТРОГИЙ скоринг слов Tesseract
         (conf>=60, >=30 символов, перевес x2.5). Печатный текст
         читается уверенно только в верной ориентации; на рукописи
         скоринги малы, решение НЕ принимается (мягкий порог conf>=40
         давал ложные срабатывания: Tesseract «читал» перевёрнутую
         рукопись — 20 символов против 3 у верной);
      3) при нехватке текста: уверенная вертикальная ось -> 90° CW,
         иначе 0° (на эталонных фото верно для всей рукописи).

    Возвращает (угол_поворота_по_часовой, пояснение)."""
    if image_cv is None or image_cv.size == 0:
        return 0, "пустое изображение"

    # --- 1) ось ---
    blob0 = _word_blob_axis_score(image_cv)
    blob90 = _word_blob_axis_score(_rot(image_cv, 90))
    proj0 = _proj_axis_score(image_cv)
    proj90 = _proj_axis_score(_rot(image_cv, 90))
    blob_ratio = (blob90 + 1.0) / (blob0 + 1.0)
    proj_ratio = (proj90 + 1e-6) / (proj0 + 1e-6)
    axis_prod = blob_ratio * proj_ratio
    if axis_prod > 1.25:
        cands = (90, 270)
        axis_why = f"ось вертикальна ({axis_prod:.2f})"
    elif axis_prod < 0.8:
        cands = (0, 180)
        axis_why = f"ось горизонтальна ({axis_prod:.2f})"
    else:
        cands = (0, 90, 180, 270)
        axis_why = f"ось неясна ({axis_prod:.2f})"

    # --- 2) строгий скоринг слов по кандидатам оси ---
    def _score_all(lang):
        return {deg: _ocr_word_score(image_cv, deg, lang, min_conf=60)
                for deg in cands}

    def _decide(stats):
        ranked = sorted(cands, key=lambda d: stats[d][0] * max(stats[d][1], 1.0),
                        reverse=True)
        best, second = ranked[0], ranked[1]
        b_chars, b_conf = stats[best]
        s_chars, s_conf = stats[second]
        b_score = b_chars * max(b_conf, 1.0)
        s_score = s_chars * max(s_conf, 1.0)
        if b_chars >= 30 and b_score >= 2.5 * max(s_score, 1.0):
            return best, (f"печатный текст распознан уверенно: {b_chars} симв. "
                          f"против {s_chars} ({axis_why})")
        return None, (f"OCR не уверен ({b_chars} против {s_chars} симв., {axis_why})")

    for lang, tag in (("rus", ""), ("eng", " [eng]")):
        try:
            best, why = _decide(_score_all(lang))
        except Exception:
            continue
        if best is not None:
            return best, why + tag

    # --- 3) резерв по оси ---
    if cands == (90, 270):
        return 90, f"рукопись/неясный текст, {axis_why} -> 90° по часовой"
    return 0, f"рукопись/неясный текст, оставляю как есть ({axis_why})"


def auto_orient(image_cv):
    """Поворачивает изображение так, чтобы текст был «вверх головой».

    Возвращает (изображение, применённый_угол)."""
    deg, why = detect_orientation(image_cv)
    if deg == 0:
        return image_cv, 0
    return _rot(image_cv, deg), deg


# ==============================================================================
# Выравнивание наклона (deskew, точность до 0.05°)
# ==============================================================================
def estimate_skew_angle(image_cv, max_angle=15.0):
    """Оценка угла наклона текста в градусах (v4).

    Положительный угол = текст наклонён по часовой стрелке.
    1) длинные линии (таблицы, рамки, линовка) через Хафа — медиана
       с отбраковкой по MAD;
    2) проекционный метод: перебор ±6° с шагом 0.25°, затем
       уточнение с шагом 0.05°.
    Маска чернил полярно-инвариантная (работает и для мела на доске)."""
    h, w = image_cv.shape[:2]
    if h < 80 or w < 80:
        return 0.0
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if image_cv.ndim == 3 else image_cv
    try:
        bw = _ink_mask_minority(gray)
    except cv2.error:
        return 0.0

    # --- 1) длинные линии ---
    edges = cv2.Canny(cv2.GaussianBlur(bw, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150,
                            minLineLength=max(80, w // 3), maxLineGap=25)
    angs = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            a = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
            if abs(a) <= 20.0:
                angs.append(a)
    if len(angs) >= 4:
        angs = np.array(angs, dtype=np.float64)
        med = float(np.median(angs))
        mad = float(np.median(np.abs(angs - med))) * 1.4826
        if mad > 1e-6:
            keep = np.abs(angs - med) <= 3.0 * mad
            if keep.sum() >= 3:
                med = float(np.median(angs[keep]))
        if 0.10 < abs(med) <= max_angle:
            return med
        return 0.0

    # --- 2) проекционный метод ---
    scale = 600.0 / float(h)
    small = cv2.resize(bw, (max(60, int(w * scale)), 600),
                       interpolation=cv2.INTER_AREA)
    small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)[1]
    hh, ww = small.shape
    center = (ww / 2.0, hh / 2.0)

    def _score(angle_deg):
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rot = cv2.warpAffine(small, M, (ww, hh), flags=cv2.INTER_NEAREST,
                             borderValue=0)
        proj = rot.astype(np.float32).sum(axis=1)
        proj = cv2.GaussianBlur(proj.reshape(-1, 1), (1, 7), 0).ravel()
        return float(np.sum(np.diff(proj) ** 2))

    best, best_score = 0.0, _score(0.0)
    for cand in np.arange(-6.0, 6.01, 0.25):
        s = _score(float(cand))
        if s > best_score:
            best_score, best = s, float(cand)
    for cand in np.arange(best - 0.3, best + 0.31, 0.05):
        s = _score(float(cand))
        if s > best_score:
            best_score, best = s, float(cand)
    if abs(best) < 0.10 or abs(best) > max_angle:
        return 0.0
    return best


def deskew_image(image_cv, max_angle=12.0):
    """Поворот изображения так, чтобы текст был строго горизонтален.

    Возвращает (изображение, применённый_угол). Кадр расширяется,
    полярность заливки подбирается под документ."""
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
    # заливка углов поворота: цвет фона страницы. Яркой странице — яркая
    # заливка (иначе серые «углы» провоцируют _trim_dark_margins срезать
    # до 12% содержимого с каждой стороны), тёмной доске — цвет доски
    # (медиана кромки). Чёрные/белые заливки сбивали и полярность мела.
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if image_cv.ndim == 3 else image_cv
    t = max(4, min(h, w) // 50)
    border = np.concatenate([
        gray[:t, :].ravel(), gray[-t:, :].ravel(),
        gray[:, :t].ravel(), gray[:, -t:].ravel(),
    ])
    if _ink_is_bright(gray):
        fill_val = int(np.median(border)) if border.size else 255
    else:
        fill_val = int(np.clip(np.percentile(border, 90), 190, 255)) if border.size else 255
    fill = (fill_val, fill_val, fill_val)
    rot = cv2.warpAffine(image_cv, M, (nW, nH), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return rot, ang


# ==============================================================================
# Выправление деформации (изгиба) страницы
# ==============================================================================
def _strip_profiles(bw, k_strips):
    """Проекции чернил по вертикальным полосам. [(профиль, масса), ...]"""
    h, w = bw.shape[:2]
    strip_w = max(8, int(math.ceil(w / k_strips)))
    out = []
    x = 0
    while x < w:
        x2 = min(w, x + strip_w)
        p = bw[:, x:x2].sum(axis=1).astype(np.float32)
        out.append((p, float(p.sum())))
        x = x2
    return out


def _best_lag_ncc(p1, p2, lag_range):
    """Сдвиг с максимальной нормированной корреляцией + суб-пиксель."""
    n = len(p1)
    if n != len(p2) or n < 4 * lag_range:
        return None
    a = p1 - p1.mean()
    b = p2 - p2.mean()
    sa = float((a * a).sum()) ** 0.5
    sb = float((b * b).sum()) ** 0.5
    if sa < 1e-6 or sb < 1e-6:
        return None
    ccs = {}
    for lag in range(-lag_range, lag_range + 1):
        if lag >= 0:
            aa, bb = a[lag:], b[:n - lag]
        else:
            aa, bb = a[:n + lag], b[-lag:]
        if len(aa) < 8:
            continue
        cc = float((aa * bb).sum()) / max(1e-9, float((aa * aa).sum()) ** 0.5 *
                                          float((bb * bb).sum()) ** 0.5)
        ccs[lag] = cc
    if not ccs:
        return None
    best_lag = max(ccs, key=lambda k: ccs[k])
    best_cc = ccs[best_lag]
    if best_cc < 0.30 or len(ccs) < 3:
        return None
    # суб-пиксель по параболе через соседей
    sub = 0.0
    if best_lag - 1 in ccs and best_lag + 1 in ccs:
        y0, y1, y2 = ccs[best_lag - 1], ccs[best_lag], ccs[best_lag + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-9:
            sub = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    return best_lag + sub, best_cc


def dewarp_page(image_cv, work_h=1400):
    """Выправляет деформацию (изгиб) страницы по строкам текста.

    Метод: изображение делится на вертикальные полосы; проекции чернил
    соседних полос взаимно коррелируются -> кумулятивные вертикальные
    сдвиги -> линейный тренд (остаточный наклон) вычитается -> по
    остатку (кривизне) строится обратное поле смещений и делается remap.
    Изогнутые строки становятся строго горизонтальными.

    Возвращает (изображение, амплитуда_коррекции_в_пикселях)."""
    if image_cv is None or image_cv.size == 0:
        return image_cv, 0.0
    h0, w0 = image_cv.shape[:2]
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if image_cv.ndim == 3 else image_cv
    scale = min(1.0, work_h / float(h0))
    if scale < 1.0:
        small = cv2.resize(gray, (int(w0 * scale), int(h0 * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray
    h, w = small.shape[:2]
    try:
        bw = _ink_mask_minority(small)
    except cv2.error:
        return image_cv, 0.0
    if float((bw > 0).mean()) < 0.004:
        return image_cv, 0.0  # текста нет — нечего выправлять

    k = 24
    profiles = _strip_profiles(bw, k)
    lag_range = max(4, int(0.03 * h))
    max_mass = max(m for _, m in profiles) if profiles else 0.0
    if max_mass <= 0:
        return image_cv, 0.0

    # кумулятивные сдвиги полос (по цепочке корреляций)
    # знак: lag = (строка фичи в левой полосе) - (строка фичи в правой),
    # поэтому offs должен накопить d_{i+1} - d_i, а не наоборот
    # (v4.1: прежний знак разворачивал коррекцию — изгиб удваивался)
    offs = {0: 0.0}
    good = {0}
    cur = 0.0
    for i in range(len(profiles) - 1):
        p1, a1 = profiles[i]
        p2, a2 = profiles[i + 1]
        if a1 < 0.10 * max_mass or a2 < 0.10 * max_mass:
            continue  # полоса почти без чернил — пропускаем связку
        res = _best_lag_ncc(p1, p2, lag_range)
        if res is None:
            continue
        lag, cc = res
        cur -= float(lag)
        offs[i + 1] = cur
        good.add(i + 1)
    if len(good) < max(6, len(profiles) // 3):
        return image_cv, 0.0  # корреляций слишком мало

    # заполняем пропуски линейной интерполяцией
    xs_all = np.arange(len(profiles), dtype=np.float64)
    xs_good = np.array(sorted(good), dtype=np.float64)
    ys_good = np.array([offs[int(i)] for i in xs_good], dtype=np.float64)
    ys = np.interp(xs_all, xs_good, ys_good)

    # вычитаем линейный тренд (наклон уже убран deskew)
    A = np.vstack([xs_good, np.ones_like(xs_good)]).T
    try:
        coef, _, _, _ = np.linalg.lstsq(A, ys_good, rcond=None)
    except np.linalg.LinAlgError:
        return image_cv, 0.0
    trend = coef[0] * xs_all + coef[1]
    resid = ys - trend
    amp = float(np.max(np.abs(resid)))
    if amp < 1.5 or amp > 0.06 * h:
        return image_cv, 0.0  # страница плоская или что-то пошло не так

    # сглаживаем остаток параболой (изгиб страницы гладкий)
    try:
        c2 = np.polyfit(xs_good, ys_good - (coef[0] * xs_good + coef[1]), 2)
    except np.linalg.LinAlgError:
        return image_cv, 0.0
    resid_fit = np.polyval(c2, xs_all)
    # качество фитировки
    ss_res = float(np.sum((resid[xs_all.astype(int)] - resid_fit) ** 2))
    ss_tot = float(np.sum((resid - resid.mean()) ** 2)) + 1e-9
    if ss_res / ss_tot > 0.35:
        return image_cv, 0.0  # кривизна не гладкая — не доверяем

    # поле смещений: y' = y + R(x), строим В ПОЛНОМ разрешении
    # (v4.1: раньше r_full считался в ширину РАБОЧЕЙ копии w, а карта
    #  строилась на полную ширину w0 — remap падал с assertion на любом
    #  изображении выше work_h, и этап молча отключался через except)
    inv = 1.0 / scale
    xs_px_full = (xs_all + 0.5) / len(profiles) * w0    # центры полос, полн. масштаб
    r_full = np.interp(np.arange(w0, dtype=np.float64), xs_px_full,
                       resid_fit * inv).astype(np.float32)
    map_x = np.tile(np.arange(w0, dtype=np.float32), (h0, 1))
    map_y = (np.arange(h0, dtype=np.float32).reshape(-1, 1) +
             r_full.reshape(1, -1))
    warped = cv2.remap(image_cv, map_x, map_y, cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REPLICATE)
    return warped, amp / scale


# ==============================================================================
# Качество: нормализация освещения, резкость, чистота
# ==============================================================================
def _is_inner_rectangle(image_cv, pts):
    """Похож ли найденный 4-угольник на ВНУТРЕННИЙ элемент листа (таблицу,
    рамку, картинку), а не на границы документа?"""
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


def _inset_quad(pts, frac=0.02):
    """Слегка поджимает четырёхугольник к центру.

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


def _page_is_bright(image_cv):
    """Преимущественно светлый документ (бумага) или тёмный (доска)."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if image_cv.ndim == 3 else image_cv
    return float((gray > 170).mean()) > 0.45


def _border_dark_fraction(image_cv):
    """Доля тёмных пикселей в кромочной рамке изображения (для светлых страниц).

    Большая доля = вокруг листа в кадре остался тёмный фон (например,
    габаритный бокс вокруг наклонного листа после перспективной коррекции)."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if image_cv.ndim == 3 else image_cv
    h, w = gray.shape[:2]
    t = max(4, int(0.02 * min(h, w)))
    ring = np.concatenate([
        gray[:t, :].ravel(), gray[-t:, :].ravel(),
        gray[:, :t].ravel(), gray[:, -t:].ravel(),
    ])
    if ring.size == 0:
        return 0.0
    return float((ring < 170).mean())


def _trim_dark_margins(image_cv, max_trim_frac=0.12):
    """Обрезает со сторон выровненного скана полосы, где явно виден тёмный фон.

    Срезаются только стороны, где БОЛЬШИНСТВО пикселей тёмные; контент
    с тёмными элементами не трогаем. Тёмные документы (доска) не трогаем."""
    if not _page_is_bright(image_cv):
        return image_cv
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


def _trim_light_margins(image_cv, max_trim_frac=0.10):
    """Обрезает светлые поля, добавленные поворотом (после deskew).

    Режет только строки/столбцы, которые почти целиком белые И лежат
    у самой границы. Контент (белые поля страницы) защищён порогом
    «почти целиком» (>= 97%) и лимитом 10%."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if image_cv.ndim == 3 else image_cv
    h, w = gray.shape[:2]
    white = gray >= 244
    top = bottom = left = right = 0
    max_t = int(h * max_trim_frac)
    max_s = int(w * max_trim_frac)
    while top < max_t and white[top, :].mean() >= 0.97:
        top += 1
    while bottom < max_t and white[h - 1 - bottom, :].mean() >= 0.97:
        bottom += 1
    while left < max_s and white[:, left].mean() >= 0.97:
        left += 1
    while right < max_s and white[:, w - 1 - right].mean() >= 0.97:
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


def _ink_is_bright(gray):
    """Чернила — яркий класс (мел на доске)?

    Решает полярность для Ч/Б-скана: Otsu делит пиксели на тёмный/яркий
    класс; чернила почти всегда в меньшинстве. Инвертировать нужно,
    только если ЯРКИЙ класс в явном меньшинстве — тогда именно он
    чернила (мел, белый маркер). Тёмное фото белой бумаги сюда НЕ
    попадает: там в меньшинстве тёмный текст, полярность обычная.

    v4.1: решение по КЛАССУ БОЛЬШИНСТВА (фон поверхности): инверсия
    нужна, если большинство пикселей — тёмные (доска), а яркий класс
    заметно ярче. Прежнее правило «яркий класс в меньшинстве (<0.35)»
    было неустойчиво: лист бумаги на доске раздувал яркое меньшинство
    до 0.29…0.37, и решение переворачивалось от кадра к кадру."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (max(1, gray.shape[1] // 4), max(1, gray.shape[0] // 4)),
                       interpolation=cv2.INTER_AREA)
    vals = small[(small > 8) & (small < 248)]
    if vals.size < 1000:
        return False
    # Otsu по гистограмме отфильтрованных значений
    hist, _ = np.histogram(vals, bins=256, range=(0, 256))
    total = float(vals.size)
    sum_all = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    best_t, best_var = 128, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f <= 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    dark = vals[vals <= best_t]
    bright = vals[vals > best_t]
    if dark.size == 0 or bright.size == 0:
        return False
    # контраст классов: близкие средние = почти однородное изображение,
    # полярность неопределима — считаем обычным документом
    if float(bright.mean()) - float(dark.mean()) < 25.0:
        return False
    # фон (класс большинства) тёмный -> чернила яркие (мел) -> инверсия
    return dark.size > bright.size


def enhance_gray(gray, invert_auto=True, clahe=True, sharpen=True):
    """Улучшение серого изображения (v4): ровный свет + контраст + чётко.

    1) полярность: тёмный документ (мел на доске) инвертируется ДО
       нормализации — иначе «деление на фон» уводит и доску, и мел
       в 255 и контраст гибнет;
    2) нормализация освещения: фон = уменьшенная+размытая копия
       (быстро даже на 20 Мпкс, тени/пересветы убирает так же);
    3) шумоподавление с сохранением краёв;
    4) CLAHE — локальный контраст текста ( отключается для бинаризации);
    5) двухмасштабное повышение резкости (мелкие штрихи + контуры).

    Параметры clahe/sharpen выключаются в Ч/Б-пути: Sauvola самодостаточен
    на выровненном свете, а CLAHE+резкость перед порогом усугубляют шум."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if invert_auto and _ink_is_bright(gray):
        # мел на доске: инвертируем ДО нормализации, иначе «деление на фон»
        # уводит и доску, и мел в 255 и контраст гибнет
        gray = cv2.bitwise_not(gray)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    h, w = gray.shape[:2]
    # 1) нормализация освещения (фон через downscale — в ~50 раз быстрее median35)
    sw, sh = max(1, w // 8), max(1, h // 8)
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    bg_s = cv2.GaussianBlur(small, (0, 0), max(3.0, sw * 0.02))
    bg = cv2.resize(bg_s, (w, h), interpolation=cv2.INTER_LINEAR)
    norm = cv2.divide(gray, bg, scale=255)
    if not clahe and not sharpen:
        return norm
    # 2) шумоподавление (edge-preserving; размеры выше 8 Мпкс — не масштабируем)
    if h * w <= 8_000_000:
        norm = cv2.bilateralFilter(norm, 5, 40, 40)
    else:
        norm = cv2.GaussianBlur(norm, (3, 3), 0)
    # 3) локальный контраст
    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm = clahe_obj.apply(norm)
    # 4) резкость: мелкая + средняя
    blur_f = cv2.GaussianBlur(norm, (0, 0), 0.8)
    fine = cv2.addWeighted(norm, 1.7, blur_f, -0.7, 0)
    blur_m = cv2.GaussianBlur(fine, (0, 0), 2.2)
    sharp = cv2.addWeighted(fine, 1.35, blur_m, -0.35, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def enhance_bgr(bgr):
    """Улучшение цветного изображения (v4): тени долой, цвет сочнее, чётко.

    Нормализация освещения идёт по каждому каналу отдельно (фон —
    уменьшенная размытая версия канала), поэтому цвет сохраняется,
    а тени и пересветы убираются. Быстро даже на 20 Мпкс."""
    h, w = bgr.shape[:2]
    sw, sh = max(1, w // 8), max(1, h // 8)
    out = []
    for c in range(3):
        ch = bgr[:, :, c]
        small = cv2.resize(ch, (sw, sh), interpolation=cv2.INTER_AREA)
        bg_s = cv2.GaussianBlur(small, (0, 0), max(3.0, sw * 0.02))
        bg = cv2.resize(bg_s, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(cv2.divide(ch, bg, scale=255))
    norm = cv2.merge(out)
    # лёгкая автоконтрастность цвета
    norm = cv2.convertScaleAbs(norm, alpha=1.06, beta=-6)
    # шумоподавление с сохранением краёв (только для разумных размеров)
    if h * w <= 8_000_000:
        den = cv2.fastNlMeansDenoisingColored(norm, None, 5, 5, 7, 21)
    else:
        den = cv2.bilateralFilter(norm, 5, 45, 45)
    # резкость
    blur = cv2.GaussianBlur(den, (0, 0), 1.6)
    sharp = cv2.addWeighted(den, 1.4, blur, -0.4, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _remove_specks(bw, max_area):
    """Удаляет «плавающие точки» как связные компоненты-крохи.

    В отличие от морфологического открытия НЕ съедает тонкие штрихи
    (дефисы, точки «ё», засечки).

    Реализация векторизована через LUT: на фото 15 Мп может быть
    ~50 000 компонентов, и покомпонентное присваивание `out[labels == i]`
    (полноразмерная маска на каждый) занимало минуты. Теперь один
    проход по изображению."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 2:
        return bw
    # LUT: какие id компонентов убрать (фон id=0 всегда крупный — не трогаем)
    drop = np.zeros(n, dtype=bool)
    areas = stats[:, cv2.CC_STAT_AREA]
    drop[1:] = areas[1:] <= max_area
    if not drop.any():
        return bw
    return np.where(drop[labels], 0, bw).astype(np.uint8)


def _remove_border_artifacts(bw):
    """Убирает «фоновый» мусор, прижатый к рамке кадра (v4.1).

    Два механизма:
      1) полосная зачистка: пока кромочная полоса со стороны рамки
         занята чернилами плотнее 40% — это фон (оправа доски, срез
         соседней страницы, тёмный предмет у края, остатки выравнивания);
         полоса очищается и проверяется следующая (не глубже 8% стороны);
      2) покомпонентная: большие/толстые компоненты, касающиеся рамки,
         удаляются; мелкий текст и тонкие линии таблиц не трогаются.
    Векторизовано через LUT."""
    h, w = bw.shape[:2]
    if not (bw == 0).any():
        return bw
    out = bw.copy()

    # --- 1) полосная зачистка кромок ---
    strip = max(10, int(0.03 * min(h, w)))
    max_eat = int(0.08 * min(h, w))

    def _eat(view_fn):
        eaten = 0
        while eaten < max_eat:
            view = view_fn(eaten)
            if view.size == 0 or float((view == 0).mean()) <= 0.40:
                break
            view[...] = 255
            eaten += strip

    _eat(lambda e: out[e:e + strip, :])            # верх
    _eat(lambda e: out[h - e - strip:h - e, :])    # низ
    _eat(lambda e: out[:, e:e + strip])            # лево
    _eat(lambda e: out[:, w - e - strip:w - e])    # право

    # --- 2) покомпонентная зачистка ---
    ink = (out == 0)
    if not ink.any():
        return out
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8)
    if n <= 2:
        return out
    mx = max(4, int(0.015 * min(h, w)))          # толщина «кольца» рамки
    area_thr = 0.004 * float(h * w)                # 0.4% площади кадра
    lefts = stats[1:, cv2.CC_STAT_LEFT].astype(np.int64)
    tops = stats[1:, cv2.CC_STAT_TOP].astype(np.int64)
    cws = stats[1:, cv2.CC_STAT_WIDTH].astype(np.int64)
    chs = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.int64)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    touches = (lefts < mx) | (tops < mx) | \
              (lefts + cws > w - mx) | (tops + chs > h - mx)
    # полоса-кандидат обязана быть ТОЛСТОЙ (рамка/клякса),
    # тонкую линию таблицы/подчёркивание не трогаем
    band = ((cws > 0.55 * w) & (chs > 0.02 * h)) | \
           ((chs > 0.55 * h) & (cws > 0.02 * w))
    drop = touches & ((areas > area_thr) | band)
    if drop.any():
        lut = np.zeros(n, dtype=bool)
        lut[1:] = drop
        out = np.where(lut[labels], 255, out).astype(np.uint8)
    return out


def _sauvola_binarize(gray, window=101, k=0.22, r=128.0):
    """Бинаризация Сауволы: T = m·(1 + k·(s/R − 1)).

    Локальный порог зависит и от среднего m, и от дисперсии s: в ровном
    фоне окно с малой дисперсией поднимает порог выше шума (фон чист),
    в области текста порог уверенно отделяет штрих. Устойчива к неравномерному
    освещению и слабому контрасту (карандаш, мел). Реализована через
    boxFilter — быстро даже на 15 Мпкс. Чернила = 0 (тёмный)."""
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, -1, (window, window),
                         borderType=cv2.BORDER_REPLICATE)
    mean2 = cv2.boxFilter(f * f, -1, (window, window),
                          borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
    T = mean * (1.0 + k * (std / r - 1.0))
    return np.where(f < T, 0, 255).astype(np.uint8)


def apply_filter(warped, selected_filter):
    """Применяет выбранный фильтр (v4.1: Sauvola для Ч/Б, чище и чётче)."""
    if selected_filter == "color":
        return enhance_bgr(warped)

    if selected_filter == "grayscale":
        gray = enhance_gray(warped)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # adaptive_threshold — «Ч/Б документ»
    # 1) выровненный свет БЕЗ CLAHE/резкости: локальная статистика
    #    Sauvola должна видеть настоящие штрихи, а не усиленный шум;
    # 2) медиана 3×3 убивает зерно камеры/JPEG;
    # 3) Sauvola с окном ~1/20 короткой стороны;
    # 4) полярность: тёмный документ (мел на доске уже инвертирован
    #    в enhance_gray) — просто проверяем долю чернил.
    norm = enhance_gray(warped, clahe=False, sharpen=False)
    norm = cv2.medianBlur(norm, 3)
    h, w = norm.shape[:2]
    win = max(31, min(121, (min(h, w) // 20) | 1))
    bw = _sauvola_binarize(norm, window=win, k=0.22)
    if float((bw == 0).mean()) > 0.55:
        bw = cv2.bitwise_not(bw)
    # «плавающие точки»: порог площади масштабируется с разрешением
    # (при 4000+ px зерно крупнее, а точки «ё» ещё крупнее — выживают)
    max_area = max(4, int((min(h, w) / 700.0) ** 2))
    ink = cv2.bitwise_not(bw)
    ink = _remove_specks(ink, max_area=max_area)
    bw = cv2.bitwise_not(ink)
    # оправа доски / остатки фона по краям -> сплошные пятна: убираем
    bw = _remove_border_artifacts(bw)
    # сглаживание «рваных» краёв (лёгкий антиалиасинг)
    bw = cv2.GaussianBlur(bw, (3, 3), 0)
    _, bw = cv2.threshold(bw, 170, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


# ==============================================================================
# Главный конвейер
# ==============================================================================
def process_document(image_cv, selected_filter="adaptive_threshold",
                     allow_fallback=True, fast=False, orientation=None):
    """Полный конвейер «фото -> идеальный скан» (v4).

    Шаги: границы документа -> перспективная коррекция -> автоориентация
    (0/90/180/270) -> доворот до строго горизонтального текста ->
    выправление деформации -> обрезка остатков фона -> фильтр качества.

    Возвращает (processed, scanned, error, info):
      processed — np.ndarray (BGR) либо None при ошибке
      scanned   — True (перспектива выровнена) / "deskewed" (границы
                  не найдены, но выравнивание и ориентация сделаны) /
                  False (обработан весь кадр как есть)
      error     — текст ошибки либо None
      info      — dict: orientation, skew, dewarp, reason (для подписи)

    fast=True отключает OCR-скоринг ориентации и dewarp (для
    пересчёта фильтра кнопкой «Сменить фильтр»);
    orientation=N — применить заранее известный поворот (кеш из
    предыдущего прохода, чтобы пересчёт фильтра не менял геометрию)."""
    info = {"orientation": 0, "skew": 0.0, "dewarp": 0.0, "reason": ""}
    if image_cv is None or image_cv.size == 0:
        return None, False, "Входное изображение пустое.", info

    if selected_filter not in FILTER_KEYS:
        selected_filter = "adaptive_threshold"

    orig = image_cv.copy()
    pts = find_document_contour(image_cv)
    # Если «документ» — это на самом деле таблица/рамка на листе, который
    # занимает весь кадр, обрезать НЕЛЬЗЯ: обработаем весь лист целиком.
    if pts is not None and _is_inner_rectangle(image_cv, pts):
        pts = None
    if pts is not None:
        pts = _inset_quad(pts)

    if pts is not None:
        try:
            warped = four_point_transform(orig, pts)
        except Exception as e:
            return None, False, f"Ошибка перспективного преобразования: {e}.", info
        if warped is None or warped.size == 0:
            return None, False, "Перспективное преобразование дало пустое изображение.", info
        scanned = True
        # v4.1: ВТОРОЙ ПРОХОД поиска границ. Если первый quad был габаритным
        # боксом вокруг наклонного листа (minAreaRect — безопасен для изогнутых
        # краёв, но оставляет треугольники фона), то теперь лист внутри рамки
        # уже выпрямлен и имеет ровные края — находим его точно.
        try:
            if _page_is_bright(warped) and _border_dark_fraction(warped) > 0.20:
                pts2 = find_document_contour(warped)
                if pts2 is not None and not _is_inner_rectangle(warped, pts2):
                    h2, w2 = warped.shape[:2]
                    frac2 = float(cv2.contourArea(pts2.astype(np.float32))) / float(h2 * w2)
                    if 0.40 <= frac2 <= 0.97:
                        cand = four_point_transform(warped, _inset_quad(pts2))
                        if (cand is not None and cand.size > 0 and
                                _border_dark_fraction(cand) < _border_dark_fraction(warped)):
                            warped = cand
        except Exception:
            pass
    else:
        if not allow_fallback:
            return None, False, "Не удалось найти 4-угольный контур документа.", info
        warped = orig
        scanned = False

    # --- автоориентация (90/180/270) ---
    try:
        if orientation is not None:
            deg = int(orientation) % 360
            if deg:
                warped = _rot(warped, deg)
        elif fast:
            deg = 0
        else:
            warped, deg = auto_orient(warped)
        info["orientation"] = deg
    except Exception:
        deg = 0
    if scanned is False and deg != 0:
        scanned = "deskewed"

    # --- доворот до строго горизонтального текста ---
    try:
        warped, ang = deskew_image(warped, max_angle=10.0)
        info["skew"] = round(float(ang), 2)
        if ang:
            warped = _trim_light_margins(warped)
            warped = _trim_dark_margins(warped)
            if scanned is False:
                scanned = "deskewed"
    except Exception:
        pass

    # --- выправление деформации ---
    if not fast:
        try:
            warped, amp = dewarp_page(warped)
            info["dewarp"] = round(float(amp), 1)
        except Exception:
            pass

    # --- срезаем остатки тёмного фона по краям ---
    try:
        warped = _trim_dark_margins(warped)
    except Exception:
        pass

    if warped is None or warped.size == 0:
        return None, scanned, "Выравнивание дало пустое изображение.", info

    # --- фильтр качества ---
    try:
        processed_output = apply_filter(warped, selected_filter)
    except cv2.error as e:
        return None, scanned, f"Ошибка OpenCV при наложении фильтра: {e}", info

    if processed_output is None or processed_output.size == 0:
        return None, scanned, "Обработка не дала итогового изображения.", info

    return processed_output, scanned, None, info


# ==============================================================================
# OCR
# ==============================================================================
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


# ==============================================================================
# Определение типа содержимого: страница или таблица
# ==============================================================================
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
