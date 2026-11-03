# -*- coding: utf-8 -*-
"""
Ядро бота обработки изображений.

Полный конвейер:
1) автоматический поиск границ документа/объекта и коррекция перспективы
   и деформации (4-точечное преобразование);
2) автоматическое определение угла наклона и выравнивание по горизонтали
   И вертикали (поворот без потери содержимого);
3) повышение чёткости и качества: шумоподавление, выравнивание
   освещённости (CLAHE) + нормализация белизны фона, обрезка артефактов
   поворота, усиление резкости (маска нерезкости).

Запуск из командной строки:
    python bot_processor.py вход.jpg результат.jpg
"""
import sys

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Геометрия: порядок точек и коррекция перспективы
# ----------------------------------------------------------------------------

def order_points(pts):
    """Упорядочивает 4 точки: левый-верхний, правый-верхний, правый-нижний, левый-нижний."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]          # левый верхний
    rect[2] = pts[np.argmax(s)]          # правый нижний
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]       # правый верхний
    rect[3] = pts[np.argmax(diff)]       # левый нижний
    return rect


def _expand_quad(pts, margin=0.02):
    """Слегка расширяет четырёхугольник наружу, чтобы не срезать края документа."""
    c = pts.mean(axis=0)
    return (c + (pts - c) * (1.0 + margin)).astype("float32")


def four_point_transform(image, pts):
    """Выпрямляет четырёхугольник pts в ровный прямоугольник (вид «сверху»)."""
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect
    width = max(50, int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = max(50, int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(
        image, m, (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


# ----------------------------------------------------------------------------
# Поиск границ документа / объекта
# ----------------------------------------------------------------------------

def _quad_from_contour(c, allow_fallback):
    """Пытается получить 4-угольник из контура (approx → выпуклая оболочка → minAreaRect)."""
    peri = cv2.arcLength(c, True)
    for eps in (0.02, 0.04):
        ap = cv2.approxPolyDP(c, eps * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            return ap.reshape(4, 2).astype("float32")
    if not allow_fallback:
        return None
    hull = cv2.convexHull(c)
    peri2 = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03):
        ap = cv2.approxPolyDP(hull, eps * peri2, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype("float32")
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype("float32")


def _quads_from_mask(mask, img_shape, min_area_ratio, allow_fallback=False):
    """Ищет самый большой выпуклый 4-угольник в бинарной маске."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    img_area = float(img_shape[0] * img_shape[1])
    for c in cnts[:12]:
        area = cv2.contourArea(c)
        if area < min_area_ratio * img_area or area > 0.99 * img_area:
            continue
        quad = _quad_from_contour(c, allow_fallback)
        if quad is not None:
            return quad
    return None


def _paper_mask(image, close_size=15):
    """Маска «бумаги»: светлые пиксели с низкой насыщенностью (страница, тетрадь).

    Работает даже на пёстрой скатерти или тёмном столе: цветной фон отсекается
    по насыщенности, тёмный — по яркости.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((s < 95) & (v > 120)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return mask


def _validate_quad(image, quad):
    """Проверяет, что квадрант действительно лежит на странице (бумаге).

    Обучено на фото пользователя: отбрасывает рамки «на весь кадр» от
    мусорных контуров (край стола, переплёт, фон). Критерии:
    - не менее 60% площади внутри рамки занимает «бумага»;
    - углы между сторонами близки к 90 град. (60..120);
    - отношение сторон не длиннее 3.2 (страница/разворот).
    """
    rect = order_points(quad)
    tl, tr, br, bl = rect
    sides = [tr - tl, br - tr, br - bl, tl - bl]
    # прямоугольность: угол между соседними сторонами
    for i in range(4):
        v1 = sides[i]
        v2 = sides[(i + 1) % 4]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 20 or n2 < 20:
            return False
        cosang = float(np.dot(v1, v2) / (n1 * n2))
        # cos близко к 0 для 90 град.; допускаем 60..120 град.
        if abs(cosang) > 0.5:
            return False
    w = np.linalg.norm(tr - tl)
    h = np.linalg.norm(bl - tl)
    if w < 40 or h < 40:
        return False
    if max(w, h) / max(min(w, h), 1e-3) > 3.2:
        return False
    # перекрытие с бумагой
    h_img, w_img = image.shape[:2]
    paper = _paper_mask(image) > 0
    qmask = np.zeros((h_img, w_img), np.uint8)
    cv2.fillPoly(qmask, [np.round(rect).astype(np.int32)], 1)
    qarea = float(qmask.sum())
    if qarea < 0.05 * h_img * w_img:
        return False
    overlap = float((paper & (qmask > 0)).sum()) / qarea
    return overlap >= 0.70


def find_document_contour(image, min_area_ratio=0.15):
    """Находит главный четырёхугольник (страницу/объект) на фото.

    Собирает кандидатов несколькими путями и возвращает первый, прошедший
    проверку «рамка лежит на странице» (_validate_quad):
    1 — контуры рёбер Canny (обычные фото, строгий 4-угольник);
    2 — порог Отсу после CLAHE (светлый документ, тени/виньетка);
    3 — маска «бумаги» по насыщенности и яркости (обучено на 70 фото
        пользователя: тетради в клетку и распечатки на пёстрой скатерти,
        деревянном столе, ткани; изогнутые края достраиваются выпуклой
        оболочкой и minAreaRect);
    4 — инвертированный порог Отсу (тёмный объект на светлом фоне).
    Возвращает 4x2 точки или None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    paper = _paper_mask(image)

    edged = cv2.Canny(blur, 60, 160)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=2)
    edged = cv2.erode(edged, np.ones((3, 3), np.uint8), iterations=1)

    eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(blur)
    _, thr = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    thr_inv = cv2.morphologyEx(cv2.bitwise_not(thr), cv2.MORPH_CLOSE,
                               np.ones((9, 9), np.uint8))

    candidates = (
        (edged, False),   # Canny: только строгий 4-угольник
        (thr, True),      # Otsu: допустим fallback
        (paper, True),    # бумага: допустим fallback
        (thr_inv, False),
    )
    best = None
    best_overlap = -1.0
    for mask, allow_fallback in candidates:
        quad = _quads_from_mask(mask, image.shape, min_area_ratio,
                                allow_fallback=allow_fallback)
        if quad is None or not _validate_quad(image, quad):
            continue
        # из подходящих берём с максимальным перекрытием бумагой
        rect = order_points(quad)
        h_img, w_img = image.shape[:2]
        qmask = np.zeros((h_img, w_img), np.uint8)
        cv2.fillPoly(qmask, [np.round(rect).astype(np.int32)], 1)
        paper_b = paper > 0
        overlap = float((paper_b & (qmask > 0)).sum()) / float(qmask.sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best = quad
        if allow_fallback and best is not None and overlap >= 0.85:
            break
    return best


# ----------------------------------------------------------------------------
# Автоопределение и исправление наклона
# ----------------------------------------------------------------------------

def detect_skew_angle(image, max_angle=15.0):
    """Оценивает угол наклона по линиям текста/объекта (преобразование Хафа).

    Вертикальные линии приводятся к горизонтальному эквиваленту, поэтому
    выравнивание идёт сразу и по горизонтали, и по вертикали. Вклад линии
    пропорционален её длине: длинные края и клетки «весят» больше росчерков
    рукописного почерка (обучено на фото тетрадей пользователя).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(60, w // 15),
        minLineLength=max(60, w // 8),
        maxLineGap=15,
    )
    if lines is None or len(lines) < 4:
        return 0.0
    angles = []
    weights = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        a = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
        if a > 45.0:
            a -= 90.0
        elif a < -45.0:
            a += 90.0
        if abs(a) <= max_angle:
            angles.append(a)
            weights.append(float(np.hypot(x2 - x1, y2 - y1)))
    if len(angles) < 4:
        return 0.0
    # взвешенная медиана
    order = np.argsort(angles)
    angles = np.asarray(angles)[order]
    weights = np.asarray(weights)[order]
    cum = np.cumsum(weights)
    half = cum[-1] / 2.0
    idx = int(np.searchsorted(cum, half))
    return float(angles[min(idx, len(angles) - 1)])


def rotate_bound(image, angle, bg=(255, 255, 255)):
    """Поворот на угол (в градусах) без обрезки содержимого."""
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    m[0, 2] += nw / 2.0 - cx
    m[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(
        image, m, (nw, nh),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg,
    )


# ----------------------------------------------------------------------------
# Повышение чёткости и качества
# ----------------------------------------------------------------------------

def denoise(image, h=4.0):
    """Лёгкое шумоподавление (0 = выключено). Возвращает (изображение, применено ли)."""
    hh, ww = image.shape[:2]
    if h > 0 and hh * ww <= 4_500_000:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21), True
    return image, False


def balance_light(image, clahe_clip=2.0):
    """Выравнивает освещённость (CLAHE) и нормализует фон документа к белому.

    Для светлых «документных» кадров — flat-field: делим яркость на сглаженную
    оценку фона (убирает виньетку, тени, разводы). Для обычных фото — мягкое
    растягивание контраста по перцентилям.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    h, w = l.shape
    bright_frac = float((l > 150).mean())
    if bright_frac > 0.45:
        # похоже на документ: выравниваем фон (flat-field)
        k = int(min(51, max(21, min(h, w) // 16)))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bg = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
        bg = cv2.GaussianBlur(bg, (0, 0), k / 3.0, borderType=cv2.BORDER_REPLICATE)
        bgf = np.maximum(bg.astype(np.float32), 30.0)
        l = np.clip(l.astype(np.float32) / bgf * 252.0, 0, 255).astype(np.uint8)
    else:
        # обычное фото: мягкая нормализация контраста
        lo, hi = np.percentile(l, (1, 99))
        if hi - lo > 50:
            gain = min(255.0 / float(hi - lo), 1.8)
            l = np.clip((l.astype(np.float32) - float(lo)) * gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def trim_dark_edges(image, dark_thresh=95, max_strip=0.13, min_strip=0.015):
    """Обрезает тёмные однородные полосы по краям (переплёт, рука, тень).

    Обучено на фото пользователя: смотрит среднюю яркость полос у каждой
    стороны; если полоса у края стабильно тёмная — она не «бумага», а мусор.
    С каждой стороны срезается не более max_strip длины стороны.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    top = int(max(4, min(h * max_strip, 30)))
    side = int(max(4, min(w * max_strip, 30)))

    def strip_mean(row_slice, col_slice):
        s = gray[row_slice, col_slice]
        if s.size == 0:
            return 255.0
        return float(s.mean())

    x0 = 0
    while x0 < side and strip_mean(slice(0, h), slice(x0, x0 + 6)) < dark_thresh:
        x0 += 6
    x1 = w
    while x1 > w - side and strip_mean(slice(0, h), slice(x1 - 6, x1)) < dark_thresh:
        x1 -= 6
    y0 = 0
    while y0 < top and strip_mean(slice(y0, y0 + 6), slice(x0, x1)) < dark_thresh:
        y0 += 6
    y1 = h
    while y1 > h - top and strip_mean(slice(y1 - 6, y1), slice(x0, x1)) < dark_thresh:
        y1 -= 6

    if x1 - x0 < w * (1 - 2 * max_strip) or y1 - y0 < h * (1 - 2 * max_strip):
        return image  # срезали бы слишком много — фон тёмный, это не документ
    if (x1 - x0) * (y1 - y0) < (w * h) * 0.5:
        return image
    if x0 < min_strip * w and y0 < min_strip * h and x1 > w * (1 - min_strip) \
            and y1 > h * (1 - min_strip):
        return image
    return image[y0:y1, x0:x1]


def trim_white_margins(image, thresh=244, pad=14, min_keep=0.50):
    """Обрезает белые поля и клинья, оставшиеся после поворота."""
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    content = gray < thresh
    frac = float(content.mean())
    if frac < 0.02 or frac > 0.98:
        return image
    ys, xs = np.where(content)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image.shape[1], x1 + pad)
    y1 = min(image.shape[0], y1 + pad)
    if (x1 - x0) * (y1 - y0) < min_keep * image.shape[0] * image.shape[1]:
        return image  # обрезали бы слишком много — не трогаем
    if x1 - x0 < 100 or y1 - y0 < 100:
        return image
    return image[y0:y1, x0:x1]


def whiten_background(image, tolerance=60):
    """Заливает белым фоновые области, соединённые с краями кадра.

    Так убираются остатки стола вокруг документа, виньетка и клинья
    после поворота. Текст и тёмные детали не трогаются: заливка идёт
    только в диапазоне цвета точки старта (fixed range). Для светлых
    «документных» кадров фону разрешено занимать почти весь кадр;
    для тёмных/обычных фото — не более 30%, чтобы не тронуть небо и т.п.
    """
    h, w = image.shape[:2]
    total = float(h * w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_frac = float((gray > 180).mean())
    max_area = 0.95 if bright_frac > 0.5 else 0.30
    out = image.copy()
    seeds = (
        (0, 0), (w // 2, 0), (w - 1, 0),
        (w - 1, h // 2), (w - 1, h - 1),
        (w // 2, h - 1), (0, h - 1), (0, h // 2),
    )
    for seed in seeds:
        mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
        cv2.floodFill(out, mask, seed, (255, 255, 255),
                      loDiff=(tolerance,) * 3, upDiff=(tolerance,) * 3,
                      flags=flags)
        area = float((mask == 255).sum())
        if area == 0 or area > max_area * total:
            continue
        sel = mask[1:-1, 1:-1] == 255
        out[sel] = (255, 255, 255)
    return out


def balance_white(image):
    """Убирает цветовой оттенок бумаги (желтизну, синеву).

    Обучено на фото пользователя: выравнивает каналы B/G/R так, чтобы
    светлые «бумажные» пиксели стали нейтрально-серыми.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] > 150) & (hsv[:, :, 1] < 128)
    if int(mask.sum()) < image.shape[0] * image.shape[1] // 20:
        return image
    gains = []
    vals = []
    for ch in range(3):
        p = float(np.percentile(image[:, :, ch][mask], 60))
        vals.append(p)
    target = sum(vals) / 3.0
    for p in vals:
        gains.append(float(np.clip(target / max(p, 1e-3), 0.85, 1.18)))
    out = image.astype(np.float32)
    for ch in range(3):
        out[:, :, ch] *= gains[ch]
    return np.clip(out, 0, 255).astype(np.uint8)


def scan_clean(image, whiten_from=208, sat_keep=0.35):
    """Режим «сканера»: бумага становится однотонно-белой.

    Светлые пиксели плавно тянутся к чистому белому, насыщенность светлых
    участков снижается (клетка тетради и лёгкие пятна бледнеют), а тёмный
    текст, линии и рисунки сохраняют контраст.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    w = np.clip((v - whiten_from) / max(255.0 - whiten_from, 1.0), 0.0, 1.0)
    v2 = v + w * (255.0 - v)
    s2 = s * (1.0 - w * (1.0 - sat_keep))
    hsv[:, :, 2] = np.clip(v2, 0, 255).astype(np.uint8)
    hsv[:, :, 1] = np.clip(s2, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def sharpen(image, amount=0.6, sigma=3.0):
    """Повышение резкости (маска нерезкости)."""
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)


def enhance_quality(image, denoise_h=4.0, clahe_clip=2.0, sharpen_amt=0.6):
    """Полное улучшение качества одним вызовом (для совместимости)."""
    notes = []
    image, den_ok = denoise(image, denoise_h)
    notes.append("шум убран" if den_ok else "шумоподавление пропущено (большой размер)")
    image = balance_light(image, clahe_clip)
    image = sharpen(image, sharpen_amt)
    notes.append("освещённость и контраст выровнены, резкость повышена")
    return image, notes


# ----------------------------------------------------------------------------
# Главный конвейер
# ----------------------------------------------------------------------------

def process_image(image, do_perspective=True, do_skew=True, do_enhance=True,
                  scan_mode=True):
    """Обрабатывает изображение (BGR, numpy). Возвращает (результат, словарь отчёта).

    Параметры:
        do_perspective — искать страницу и исправлять перспективу/деформацию;
        do_skew        — выравнивать наклон по горизонтали и вертикали;
        do_enhance     — шумоподавление, свет, белизна, резкость;
        scan_mode      — «однотонный фон»: бумага к чистому белому
                         (эффект сканера, обучено на фото пользователя).
    """
    if image is None:
        raise ValueError("Пустое изображение")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result = image.copy()
    info = {"perspective": False, "skew_angle": 0.0, "enhanced": False,
            "scan_mode": bool(scan_mode), "steps": []}

    # 1. Перспектива и деформация
    if do_perspective:
        quad = find_document_contour(result)
        if quad is not None:
            quad = _expand_quad(quad, 0.015)
            result = four_point_transform(result, quad)
            info["perspective"] = True
            info["steps"].append("перспектива и деформация исправлены")

    # 2. Наклон: выравнивание по горизонтали и вертикали.
    # Если страница уже выровнена warp-ом по своим границам, лёгкий «наклон»
    # рукописных строк — не ошибка кадра: доворачиваем только заметный крен.
    rotated = False
    if do_skew:
        angle = detect_skew_angle(result)
        if info["perspective"] and abs(angle) < 1.5:
            angle = 0.0
        if abs(angle) >= 0.4:
            result = rotate_bound(result, angle)
            rotated = True
            info["skew_angle"] = round(float(angle), 2)
            info["steps"].append(
                "наклон %.1f град. исправлен — выровнено по горизонтали и вертикали" % angle
            )

    # 3. Чёткость и качество: шумоподавление + баланс белого + выравнивание света
    if do_enhance:
        result, den_ok = denoise(result, 5.0)
        result = balance_white(result)
        result = balance_light(result)
        info["enhanced"] = True

    # 4. Заливка фона белым от краёв: остатки стола, виньетка, клинья
    if do_perspective or rotated or scan_mode:
        result = whiten_background(result)
        if do_perspective:
            # мягкая чистка кромки: белым делаем только светлые пиксели
            # (остатки фона), тёмные (возможный текст у края) не трогаем
            h, w = result.shape[:2]
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            border = np.zeros((h, w), bool)
            border[:12, :] = True
            border[-12:, :] = True
            border[:, :12] = True
            border[:, -12:] = True
            result[border & (gray > 170)] = 255

    # 5. Обрезка артефактов: тёмные полосы (переплёт, рука, тень) и белые клинья
    if do_perspective or rotated:
        result = trim_dark_edges(result)
    if rotated:
        result = trim_white_margins(result)

    # 6. Однотонный фон (режим сканера): бумага к белому, клетка бледнеет
    if scan_mode and do_enhance:
        result = scan_clean(result)
        # повторная заливка: после осветления фона заливка от краёв
        # пробивается дальше — добивает остатки скатерти и теней
        result = whiten_background(result, tolerance=70)
        info["steps"].append("фон приведён к однотонному белому (режим сканера)")

    # 7. Резкость — последней, чтобы не усиливать границы артефактов
    if do_enhance:
        result = sharpen(result, 0.7)
        info["steps"].append("чёткость и качество: шум убран, свет выровнен, "
                             "оттенок исправлен, резкость повышена")

    info["size"] = {"width": int(result.shape[1]), "height": int(result.shape[0])}
    return result, info


def process_file(in_path, out_path, **kwargs):
    """Обрабатывает файл и сохраняет результат. Возвращает словарь отчёта."""
    image = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Не удалось прочитать файл: %s" % in_path)
    result, info = process_image(image, **kwargs)
    if not cv2.imwrite(out_path, result):
        raise ValueError("Не удалось сохранить файл: %s" % out_path)
    return info


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python bot_processor.py вход.jpg результат.jpg")
        sys.exit(1)
    try:
        report = process_file(sys.argv[1], sys.argv[2])
    except Exception as exc:
        print("Ошибка:", exc)
        sys.exit(1)
    print("Готово:", sys.argv[2])
    for step in report["steps"]:
        print(" -", step)
    print("Размер результата: %(width)d x %(height)d" % report["size"])
