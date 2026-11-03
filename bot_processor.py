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

def _quads_from_mask(mask, img_shape, min_area_ratio):
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
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")
    return None


def find_document_contour(image, min_area_ratio=0.15):
    """Находит главный четырёхугольник (страницу/объект) на фото.

    Путь 1 — контуры рёбер Canny (обычные фото).
    Путь 2 — порог Отсу после выравнивания освещённости (CLAHE):
             светлый документ на тёмном фоне, фото с тенями/виньеткой.
    Путь 3 — инвертированный порог Отсу: тёмный объект на светлом фоне.
    Возвращает 4x2 точки или None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blur, 60, 160)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=2)
    edged = cv2.erode(edged, np.ones((3, 3), np.uint8), iterations=1)
    quad = _quads_from_mask(edged, image.shape, min_area_ratio)
    if quad is not None:
        return quad

    # выравниваем освещённость, чтобы тени/виньетка не «съедали» края документа
    eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(blur)
    _, thr = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    quad = _quads_from_mask(thr, image.shape, min_area_ratio)
    if quad is not None:
        return quad

    thr_inv = cv2.morphologyEx(cv2.bitwise_not(thr), cv2.MORPH_CLOSE,
                               np.ones((9, 9), np.uint8))
    return _quads_from_mask(thr_inv, image.shape, min_area_ratio)


# ----------------------------------------------------------------------------
# Автоопределение и исправление наклона
# ----------------------------------------------------------------------------

def detect_skew_angle(image, max_angle=20.0):
    """Оценивает угол наклона по линиям текста/объекта (преобразование Хафа).

    Вертикальные линии приводятся к горизонтальному эквиваленту, поэтому
    выравнивание идёт сразу и по горизонтали, и по вертикали.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(60, w // 15),
        minLineLength=max(40, w // 10),
        maxLineGap=20,
    )
    if lines is None or len(lines) < 4:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        a = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
        if a > 45.0:
            a -= 90.0
        elif a < -45.0:
            a += 90.0
        if abs(a) <= max_angle:
            angles.append(a)
    if len(angles) < 4:
        return 0.0
    return float(np.median(angles))


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


def whiten_background(image, tolerance=50):
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

def process_image(image, do_perspective=True, do_skew=True, do_enhance=True):
    """Обрабатывает изображение (BGR, numpy). Возвращает (результат, словарь отчёта)."""
    if image is None:
        raise ValueError("Пустое изображение")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result = image.copy()
    info = {"perspective": False, "skew_angle": 0.0, "enhanced": False, "steps": []}

    # 1. Перспектива и деформация
    if do_perspective:
        quad = find_document_contour(result)
        if quad is not None:
            quad = _expand_quad(quad, 0.008)
            result = four_point_transform(result, quad)
            info["perspective"] = True
            info["steps"].append("перспектива и деформация исправлены")

    # 2. Наклон: выравнивание по горизонтали и вертикали
    rotated = False
    if do_skew:
        angle = detect_skew_angle(result)
        if abs(angle) >= 0.25:
            result = rotate_bound(result, angle)
            rotated = True
            info["skew_angle"] = round(float(angle), 2)
            info["steps"].append(
                "наклон %.1f град. исправлен — выровнено по горизонтали и вертикали" % angle
            )

    # 3. Чёткость и качество: шумоподавление + выравнивание света и белизны
    if do_enhance:
        result, den_ok = denoise(result, 5.0)
        result = balance_light(result)
        info["enhanced"] = True

    # 4. Заливка фона белым от краёв: остатки стола, виньетка, клинья
    if do_perspective or rotated:
        result = whiten_background(result)
        if do_perspective:
            # мягкая чистка кромки: белым делаем только светлые пиксели
            # (остатки фона), тёмные (возможный текст у края) не трогаем
            h, w = result.shape[:2]
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            border = np.zeros((h, w), bool)
            border[:8, :] = True
            border[-8:, :] = True
            border[:, :8] = True
            border[:, -8:] = True
            result[border & (gray > 180)] = 255

    # 5. Обрезка артефактов поворота (белые клинья по углам)
    if rotated:
        result = trim_white_margins(result)

    # 6. Резкость — последней, чтобы не усиливать границы артефактов
    if do_enhance:
        result = sharpen(result, 0.7)
        info["steps"].append("чёткость и качество: шум убран, свет выровнен, фон белее, резкость повышена")

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
