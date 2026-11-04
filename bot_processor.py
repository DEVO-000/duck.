# -*- coding: utf-8 -*-
"""
Ядро бота обработки изображений (движок v2).

Полный конвейер:
1) автоматический поиск границ документа/объекта и коррекция перспективы
   (4-точечное преобразование; несколько путей поиска: контуры, пороги
   Отсу/адаптивный, линии Хафа);
2) выравнивание по горизонтали И вертикали:
   - глобальный наклон — проекционным методом по строкам текста
     (дисперсия горизонтальной проекции), страховка — медиана линий Хафа;
   - построчное выпрямление (dewarp): локальный наклон строк измеряется
     в вертикальных полосах и компенсируется сдвигами столбцов — убирает
     трапецию и изгиб страницы, когда четырёхугольник не найден;
   - выравнивание вертикальных линий (клетка, поля, края) аналогичным
     методом по транспонированному изображению;
3) повышение чёткости и качества: шумоподавление, выравнивание
   освещённости и ОДНОТОННЫЙ фон (flat-field нормализация для любых
   кадров — светлых и тёмных), заливка фона белым, обрезка полей,
   усиление резкости.

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


def _expand_quad(pts, margin=0.008):
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
# Поиск границ документа / объекта (v2: несколько путей + валидация)
# ----------------------------------------------------------------------------

def _quad_angles_ok(pts):
    """Внутренние углы четырёхугольника в разумных пределах (55..125 градусов)."""
    p = order_points(pts.astype(np.float32))
    for i in range(4):
        a = p[i]
        b = p[(i + 1) % 4]
        c = p[(i + 2) % 4]
        v1 = b - a
        v2 = c - b
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        if ang < 55.0 or ang > 125.0:
            return False
    return True


def _quads_from_mask(mask, img_shape, min_area_ratio):
    """Ищет самый большой выпуклый 4-угольник в бинарной маске (с перебором eps)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    img_area = float(img_shape[0] * img_shape[1])
    for c in cnts[:12]:
        area = cv2.contourArea(c)
        if area < min_area_ratio * img_area or area > 0.995 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.035, 0.05, 0.07):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2).astype("float32")
                if _quad_angles_ok(pts):
                    return pts
    return None


def _quad_from_hough_lines(image):
    """Строит четырёхугольник из длинных линий Хафа: верх/низ/лево/право.

    Выручает, когда контур документа не замыкается (тёмные фото, доски,
    пёстрый фон), но рёбра объекта дают сильные прямые линии.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(blur)
    edges = cv2.Canny(eq, 50, 150)
    h, w = edges.shape
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(80, w // 12),
        minLineLength=max(90, int(w * 0.22)),
        maxLineGap=30,
    )
    if lines is None:
        return None
    hl, vl = [], []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        ln = float(np.hypot(x2 - x1, y2 - y1))
        ang = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))) % 180.0
        if (ang <= 25.0 or ang >= 155.0) and ln >= w * 0.18:
            hl.append((ln, (x1, y1), (x2, y2)))
        elif 65.0 <= ang <= 115.0 and ln >= h * 0.18:
            vl.append((ln, (x1, y1), (x2, y2)))
    if len(hl) < 2 or len(vl) < 2:
        return None
    hl.sort(key=lambda t: -t[0])
    vl.sort(key=lambda t: -t[0])
    top = min(hl[:5], key=lambda t: (t[1][1] + t[2][1]) / 2.0)
    bot = max(hl[:5], key=lambda t: (t[1][1] + t[2][1]) / 2.0)
    left = min(vl[:5], key=lambda t: (t[1][0] + t[2][0]) / 2.0)
    right = max(vl[:5], key=lambda t: (t[1][0] + t[2][0]) / 2.0)
    if (top is bot) or (left is right):
        return None

    def line_params(p1, p2):
        (x1, y1), (x2, y2) = p1, p2
        a, b, c = float(y2 - y1), float(x1 - x2), float(x1 * y2 - x2 * y1)
        return a, b, c

    def inter(l1, l2):
        a1, b1, c1 = line_params(*l1)
        a2, b2, c2 = line_params(*l2)
        d = a1 * b2 - a2 * b1
        if abs(d) < 1e-9:
            return None
        return (float((b1 * c2 - b2 * c1) / d), float((a2 * c1 - a1 * c2) / d))

    corners = [
        inter((top[1], top[2]), (left[1], left[2])),
        inter((top[1], top[2]), (right[1], right[2])),
        inter((bot[1], bot[2]), (right[1], right[2])),
        inter((bot[1], bot[2]), (left[1], left[2])),
    ]
    if any(c is None for c in corners):
        return None
    quad = np.array(corners, dtype="float32")
    # углы за пределами кадра с запасом -> линия не от документа
    pad = 0.12
    if (quad < [-w * pad, -h * pad]).any() or (quad > [w * (1 + pad), h * (1 + pad)]).any():
        return None
    area = cv2.contourArea(quad)
    if area < 0.10 * w * h:
        return None
    if not _quad_angles_ok(quad):
        return None
    return quad


def _quad_from_hull(mask, img_shape):
    """Ориентированный прямоугольник вокруг самого большого контура (последний шанс)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.12 * img_shape[0] * img_shape[1]:
        return None
    rect = cv2.minAreaRect(c)
    pts = cv2.boxPoints(rect).astype("float32")
    if _quad_angles_ok(pts):
        return pts
    return None


def _near_frame(quad, shape, tol=0.035, area_frac=0.90):
    """True, если четырёхугольник практически совпадает с рамкой кадра."""
    h, w = shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    q = order_points(quad.astype(np.float32))
    if cv2.contourArea(q) < area_frac * w * h:
        return False
    for i in range(4):
        if np.linalg.norm(q[i] - corners[i]) > tol * max(w, h):
            return False
    return True


def find_document_contour(image, min_area_ratio=0.10):
    """Находит главный четырёхугольник (страницу/объект) на фото.

    Пути (до первого успеха):
    1) порог Отсу после CLAHE — светлый документ на любом фоне;
    2) контуры Canny после CLAHE — обычные и тёмные фото;
    3) адаптивный порог — пёстрый фон, слабый контраст;
    4) инвертированный Отсу — тёмный объект на светлом фоне;
    5) длинные линии Хафа — рёбра объекта без замкнутого контура;
    6) ориентированный прямоугольник вокруг крупнейшего контура.
    Возвращает 4x2 точки, либо None (в т.ч. когда объект и так занимает
    весь кадр — тогда выпрямление сделает блок dewarp).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(blur)

    _, thr = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    quad = _quads_from_mask(thr, image.shape, min_area_ratio)
    if quad is not None:
        return None if _near_frame(quad, image.shape) else quad

    edged = cv2.Canny(eq, 60, 160)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=2)
    edged = cv2.erode(edged, np.ones((3, 3), np.uint8), iterations=1)
    quad = _quads_from_mask(edged, image.shape, min_area_ratio)
    if quad is not None:
        return None if _near_frame(quad, image.shape) else quad

    bs = int(max(15, min(51, min(image.shape[:2]) // 20)))
    if bs % 2 == 0:
        bs += 1
    adap = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, bs, 10)
    adap = cv2.morphologyEx(adap, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    quad = _quads_from_mask(adap, image.shape, min_area_ratio)
    if quad is not None:
        return None if _near_frame(quad, image.shape) else quad

    thr_inv = cv2.morphologyEx(cv2.bitwise_not(thr), cv2.MORPH_CLOSE,
                               np.ones((9, 9), np.uint8))
    quad = _quads_from_mask(thr_inv, image.shape, min_area_ratio)
    if quad is not None:
        return None if _near_frame(quad, image.shape) else quad

    quad = _quad_from_hough_lines(image)
    if quad is not None and not _near_frame(quad, image.shape):
        return quad

    quad = _quad_from_hull(thr, image.shape)
    if quad is not None and not _near_frame(quad, image.shape):
        return quad
    return None


# ----------------------------------------------------------------------------
# Текстовая маска и проекционные методы выравнивания (v2)
# ----------------------------------------------------------------------------

def _text_mask(gray, max_side=1000):
    """Бинарная маска текста/линий (белым) в уменьшенном масштабе.

    Адаптивный порог ловит и тёмный текст на светлом, и светлый мел
    на тёмном. Возвращает (маска, уменьшенный серый, масштаб).
    """
    h, w = gray.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        g = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    else:
        g = gray
    hh, ww = g.shape[:2]
    bs = int(np.clip(round(min(hh, ww) / 22.0), 15, 51))
    if bs % 2 == 0:
        bs += 1
    c = max(10, bs // 3)

    def clean(bw):
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
        if n > 1:
            # убираем пыль и «гигантов»: рёбра страницы, клинья поворота,
            # границы стола — они не текст и ломают измерение углов
            big = (stats[1:, cv2.CC_STAT_WIDTH] > 0.55 * ww) | \
                  (stats[1:, cv2.CC_STAT_HEIGHT] > 0.55 * hh)
            keep = np.zeros(n, bool)
            keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 8) & \
                       (stats[1:, cv2.CC_STAT_WIDTH] >= 3) & \
                       (stats[1:, cv2.CC_STAT_HEIGHT] >= 3) & ~big
            bw = bw * keep[lab].astype(np.uint8)
        return bw

    inv = clean(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY_INV, bs, c))
    pos = clean(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, bs, c))

    def frac_ok(m):
        f = float((m > 0).mean())
        return 0.004 <= f <= 0.32

    if frac_ok(inv) and not frac_ok(pos):
        mask = inv
    elif frac_ok(pos) and not frac_ok(inv):
        mask = pos
    elif frac_ok(inv) and frac_ok(pos):
        mask = inv if (inv > 0).mean() >= (pos > 0).mean() else pos
    else:
        # обе полярности «мимо» — берём ту, что ближе к разумной плотности
        di = abs(float((inv > 0).mean()) - 0.05)
        dp = abs(float((pos > 0).mean()) - 0.05)
        mask = inv if di <= dp else pos
    return mask, g, scale


def _rot_score(mask, angle):
    """Дисперсионная оценка «горизонтальности» маски при повороте на angle."""
    h, w = mask.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rot = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rs = rot.sum(axis=1).astype(np.float64)
    if rs.size < 8:
        return 0.0
    d = rs[1:] - rs[:-1]
    return float(np.dot(d, d))


def _projection_angle(mask, max_angle=12.0, coarse=0.25, fine=0.05):
    """Угол наклона строк текста проекционным методом. Возвращает (угол, уверенность)."""
    if (mask > 0).mean() < 0.002:
        return 0.0, 0.0
    angs = np.arange(-max_angle, max_angle + 1e-9, coarse)
    scores = np.array([_rot_score(mask, float(a)) for a in angs])
    i = int(np.argmax(scores))
    best = float(angs[i])
    lo, hi = best - coarse, best + coarse
    angs2 = np.arange(lo, hi + 1e-9, fine)
    scores2 = np.array([_rot_score(mask, float(a)) for a in angs2])
    j = int(np.argmax(scores2))
    angle = float(angs2[j])
    base = float(np.median(scores))
    spread = float(np.std(scores) + 1e-9)
    conf = (float(scores2[j]) - base) / spread
    return angle, conf


def _hough_median_angle(image, max_angle=20.0):
    """Страховка: медианный угол длинных линий Хафа (старый метод)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0,
                            threshold=max(60, w // 15),
                            minLineLength=max(40, w // 10), maxLineGap=20)
    if lines is None or len(lines) < 4:
        return 0.0, 0.0
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
        return 0.0, 0.0
    return float(np.median(angles)), min(1.0, len(angles) / 20.0)


def _hv_energy(mask):
    """Энергии горизонтальной (H) и вертикальной (V) линейной структуры маски."""
    rs = mask.sum(axis=1).astype(np.float64)
    cs = mask.sum(axis=0).astype(np.float64)
    h = float(np.var(rs) / (np.mean(rs) ** 2 + 1e-9))
    v = float(np.var(cs) / (np.mean(cs) ** 2 + 1e-9))
    return h, v


def _text_asymmetry(mask):
    """Медианный сдвиг центра масс строки от центра её полосы.

    У нормального (не перевёрнутого) текста центр масс чуть ниже середины
    полосы (строчные буквы + нижние выносные) — знак подсказывает, не
    перевёрнут ли текст на 180 град. Возвращает (асимметрия, число полос).
    """
    rs = mask.sum(axis=1).astype(np.float64)
    if rs.sum() < 500:
        return 0.0, 0
    rs = np.convolve(rs, np.ones(5) / 5.0, mode="same")
    thr = 0.35 * float(rs.max())
    on = rs > thr
    offs = []
    y, n_total = 0, len(rs)
    while y < n_total:
        if on[y]:
            y0 = y
            while y < n_total and on[y]:
                y += 1
            if y - y0 >= 6:
                seg = rs[y0:y]
                wsum = seg.sum()
                if wsum > 0:
                    c = float((np.arange(y0, y) * seg).sum() / wsum)
                    offs.append(c - (y0 + y - 1) / 2.0)
        else:
            y += 1
    if not offs:
        return 0.0, 0
    return float(np.median(offs)), len(offs)


def _orient_upright(image):
    """Поворачивает кадр, если он снят боком (текст вертикально).

    Признак «боком»: вертикальная линейная структура сильнее горизонтальной
    (V/H >= 2). Направление поворота выбирается по асимметрии профиля строк
    (не вверх ногами). Возвращает (изображение, повёрнут ли).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask, _, _ = _text_mask(gray)
    h, v = _hv_energy(mask)
    if v < 2.0 * max(h, 1e-9):
        return image, False
    # снят боком: выбрать направление по асимметрии текста
    m_cw = np.rot90(mask, -1)    # CW: так выглядит маска после поворота кадра CW
    m_ccw = np.rot90(mask, 1)    # CCW
    a_cw, n_cw = _text_asymmetry(m_cw)
    a_ccw, n_ccw = _text_asymmetry(m_ccw)
    code = cv2.ROTATE_90_CLOCKWISE
    if n_ccw >= 8 and n_cw >= 8 and (a_ccw - a_cw) >= 0.015:
        code = cv2.ROTATE_90_COUNTERCLOCKWISE
    return cv2.rotate(image, code), True


def detect_skew_angle(image, max_angle=15.0):
    """Определяет наклон: проекционный метод по строкам, страховка — Хаф."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask, _, _ = _text_mask(gray)
    ang, conf = _projection_angle(mask, max_angle=max_angle)
    if conf < 2.0:
        ang2, conf2 = _hough_median_angle(image, max_angle=max_angle)
        if conf2 > 0.2:
            return ang2
        return 0.0
    return ang


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
# Построчное выпрямление (dewarp): локальный наклон строк по полосам
# ----------------------------------------------------------------------------

def _shear_score(band, s):
    """Оценка «горизонтальности» полосы при вертикальном сдвиге s (тангенс).

    Сдвиг (shear) в отличие от поворота не выносит контент за пределы
    узкой полосы, поэтому годится для локального измерения наклона.
    """
    h, w = band.shape
    m = np.float32([[1.0, 0.0, 0.0], [s, 1.0, 0.0]])
    out = cv2.warpAffine(band, m, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rs = out.sum(axis=1).astype(np.float64)
    if rs.size < 8:
        return 0.0
    d = rs[1:] - rs[:-1]
    return float(np.dot(d, d))


def _band_angles(mask, n_bands=14, max_angle=6.0):
    """Локальный угол строк в вертикальных полосах (метод сдвига). [(x01, угол, вес), ...]"""
    h, w = mask.shape
    bw = max(24, int(round(w / max(4, n_bands))))
    step = max(8, int(round(bw * 0.5)))
    out = []
    for x0 in range(0, max(1, w - bw + 1), step):
        band = mask[:, x0:x0 + bw]
        cov = float((band > 0).mean())
        if cov < 0.006:
            continue
        coarse = np.arange(-max_angle, max_angle + 1e-9, 0.25)
        scores = np.array([_shear_score(band, float(np.tan(np.radians(a))))
                           for a in coarse])
        i = int(np.argmax(scores))
        best = float(coarse[i])
        fine = np.arange(best - 0.25, best + 0.25 + 1e-9, 0.05)
        scores2 = np.array([_shear_score(band, float(np.tan(np.radians(a))))
                            for a in fine])
        j = int(np.argmax(scores2))
        # знак: сдвиг s* = -tan(наклона), т.к. warpAffine инвертирует матрицу
        angle = -float(fine[j])
        base = float(np.median(scores))
        spread = float(np.std(scores) + 1e-9)
        conf = (float(scores2[j]) - base) / spread
        if conf < 1.5:
            continue
        weight = conf * min(1.0, cov / 0.04)
        out.append(((x0 + bw / 2.0) / float(w), angle, weight))
    return out


def _strip_angles(mask, n_strips=14, max_angle=6.0):
    """Локальный угол строк в горизонтальных полосах (по высоте).

    Ловит трапецию/перспективу: наклон строк меняется от строки к строке
    (верх страницы наклонён в одну сторону, низ — в другую).
    """
    h, w = mask.shape
    sh = max(20, int(round(h / max(4, n_strips))))
    step = max(6, int(round(sh * 0.5)))
    out = []
    for y0 in range(0, max(1, h - sh + 1), step):
        strip = mask[y0:y0 + sh]
        cov = float((strip > 0).mean())
        if cov < 0.006:
            continue
        coarse = np.arange(-max_angle, max_angle + 1e-9, 0.25)
        scores = np.array([_strip_score(strip, float(np.tan(np.radians(a))))
                           for a in coarse])
        i = int(np.argmax(scores))
        best = float(coarse[i])
        fine = np.arange(best - 0.25, best + 0.25 + 1e-9, 0.05)
        scores2 = np.array([_strip_score(strip, float(np.tan(np.radians(a))))
                            for a in fine])
        j = int(np.argmax(scores2))
        angle = -float(fine[j])
        base = float(np.median(scores))
        spread = float(np.std(scores) + 1e-9)
        conf = (float(scores2[j]) - base) / spread
        if conf < 1.5:
            continue
        weight = conf * min(1.0, cov / 0.04)
        out.append(((y0 + sh / 2.0) / float(h), angle, weight))
    return out


def _strip_score(strip, s):
    """Сдвиговая оценка наклона строк в горизонтальной полосе (с запасом кадра)."""
    h, w = strip.shape
    pad = int(abs(s) * w * 0.5) + 2
    m = np.float32([[1.0, 0.0, 0.0], [s, 1.0, float(pad)]])
    out = cv2.warpAffine(strip, m, (w, h + 2 * pad), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rs = out.sum(axis=1).astype(np.float64)
    if rs.size < 8:
        return 0.0
    d = rs[1:] - rs[:-1]
    return float(np.dot(d, d))


def _build_maps(shape, ch_x=None, mv_y=None, cv_y=None):
    """Карта ремапа из компонент выпрямления.

    ch_x — сдвиг столбцов Ch(x) (изгиб страницы);
    mv_y — строчный наклон m(y) (трапеция): map_y -= m(y)*(x - w/2);
    cv_y — сдвиг строк Cv(y) (вертикали).
    """
    h, w = shape[:2]
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))
    if ch_x is not None:
        map_y += ch_x.astype(np.float32).reshape(1, w)
    if mv_y is not None:
        xs = np.arange(w, dtype=np.float32) - float(w) / 2.0
        map_y -= mv_y.astype(np.float32).reshape(h, 1) * xs.reshape(1, w)
    map_x = np.tile(np.arange(w, dtype=np.float32).reshape(1, w), (h, 1))
    if cv_y is not None:
        map_x += cv_y.astype(np.float32).reshape(h, 1)
    return map_x, map_y


def _fit_line(points, degree=1):
    """Робастная аппроксимация угла полос: угол = f(x). Возвращает coef или None."""
    if len(points) < 4:
        return None

    def _fit(pts, deg):
        x = np.array([p[0] for p in pts], np.float64)
        y = np.array([p[1] for p in pts], np.float64)
        w = np.array([max(0.1, p[2]) for p in pts], np.float64)
        for _ in range(3):
            if x.size < deg + 2:
                return None, None
            coef = np.polyfit(x, y, deg, w=np.sqrt(w))
            resid = y - np.polyval(coef, x)
            mad = float(np.median(np.abs(resid - np.median(resid)))) + 0.05
            keep = np.abs(resid) < 3.0 * mad
            if keep.all() or keep.sum() < deg + 2:
                break
            x, y, w = x[keep], y[keep], w[keep]
        if x.size < deg + 2 or coef is None:
            return None, None
        resid = np.abs(y - np.polyval(coef, x))
        return coef, float(np.median(resid))

    coef1, r1 = _fit(points, 1)
    if degree >= 2 and len(points) >= 7:
        coef2, r2 = _fit(points, 2)
        # квадратику берём только если она заметно лучше (не «гнётся» на выбросах)
        if coef2 is not None and (coef1 is None or r2 < 0.7 * r1):
            return coef2
    return coef1


def _displacement_from_angles(coef, n, size_px, cap_frac=0.10):
    """Сдвиги столбцов/строк из аппроксимации углов (кумулятивная сумма тангенсов)."""
    xs = np.linspace(0.0, 1.0, n)
    ang = np.polyval(coef, xs)
    s = np.tan(np.radians(ang))
    d = np.zeros(n, np.float64)
    d[1:] = np.cumsum((s[1:] + s[:-1]) * 0.5 / (n - 1))
    d -= d.mean()
    d *= float(size_px)
    cap = cap_frac * size_px
    return np.clip(d, -cap, cap), float(np.max(ang) - np.min(ang))


def _apply_maps(image, ch_x=None, mv_y=None, cv_y=None):
    """Применяет сдвиги выпрямления одним ремапом."""
    map_x, map_y = _build_maps(image.shape[:2], ch_x, mv_y, cv_y)
    return cv2.remap(image, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(255, 255, 255) if image.ndim == 3 else 255)


def _warp_mask(mask, ch_x=None, mv_y=None, cv_y=None, replicate=False):
    """То же для маски/серого (для проверки результата)."""
    map_x, map_y = _build_maps(mask.shape[:2], ch_x, mv_y, cv_y)
    return cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_REPLICATE if replicate
                     else cv2.BORDER_CONSTANT, borderValue=0)


def _spread_of(mask):
    """Разброс локальных углов строк (градусы) — мера «ровности».

    Усечённый (10–90 перцентиль), чтобы мусорные крайние полосы не раздували оценку.
    """
    pts = _band_angles(mask)
    if len(pts) < 3:
        return 0.0
    angs = np.array([p[1] for p in pts])
    return float(np.percentile(angs, 90) - np.percentile(angs, 10))


def _vertical_line_angle(gray):
    """Медианный наклон длинных вертикальных линий (клетка, поля, края).

    Возвращает (угол_от_вертикали_в_градусах, число_линий).
    """
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0,
                            threshold=max(50, h // 20),
                            minLineLength=max(60, h // 6), maxLineGap=25)
    if lines is None:
        return 0.0, 0
    angs = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if dy < 0.0:
            dx, dy = -dx, -dy
        if abs(dy) <= abs(dx):
            continue
        a = float(np.degrees(np.arctan2(dx, dy)))
        if abs(a) <= 10.0:
            angs.append(a)
    if len(angs) < 4:
        return 0.0, len(angs)
    return float(np.median(angs)), len(angs)


def dewarp_lines(image, gray=None):
    """Построчное выпрямление: трапеция, изгиб страницы, вертикали.

    Две независимые компоненты наклона строк:
    - изгиб страницы: наклон меняется вдоль СТРОКИ -> сдвиги столбцов Ch(x);
    - трапеция/перспектива: наклон меняется от строки к строке ->
      строчная компонента m(y): map_y -= m(y)*(x - w/2).
    Вертикали: медианный наклон длинных вертикальных линий (Хаф)
    компенсируется линейным сдвигом строк Cv(y).

    Возвращает (изображение, отчёт dict). Если выправлять нечего —
    изображение без изменений.
    """
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask, gsmall, scale = _text_mask(gray)
    h, w = image.shape[:2]
    mh, mw = mask.shape
    report = {"applied": False, "h": False, "v": False, "trap": False,
              "spread_before": 0.0, "spread_after": 0.0, "v_angle": 0.0}

    def _full_ch(ch_small):
        ch = np.interp(np.arange(w, dtype=np.float64),
                       np.linspace(0, w - 1, len(ch_small)), ch_small)
        return np.clip(ch * (float(h) / mh), -0.10 * h, 0.10 * h)

    for _it in range(2):
        # --- изгиб: полосы по x -> Ch(x) ---
        ch_small = None
        pts = _band_angles(mask)
        deg = 2 if len(pts) >= 7 else 1
        coef = _fit_line(pts, degree=deg)
        if coef is not None:
            d_small, spread = _displacement_from_angles(coef, mw, mh, 0.10)
            if spread >= 0.5 or abs(float(np.polyval(coef, 0.5))) >= 0.5:
                ch_small = d_small
                if report["spread_before"] == 0.0:
                    report["spread_before"] = round(spread, 2)

        # --- трапеция: полосы по y -> m(y) ---
        mv_small = None
        pts_y = _strip_angles(mask)
        deg_y = 2 if len(pts_y) >= 7 else 1
        coef_y = _fit_line(pts_y, degree=deg_y)
        if coef_y is not None:
            ys01 = np.linspace(0.0, 1.0, mh)
            ang_y = np.polyval(coef_y, ys01)
            if float(np.max(ang_y) - np.min(ang_y)) >= 0.5 or \
                    abs(float(np.polyval(coef_y, 0.5))) >= 0.5:
                mv_small = np.tan(np.radians(ang_y))
                mv_small = np.clip(mv_small, -0.12, 0.12)

        if ch_small is None and mv_small is None:
            break

        # --- проверка на маске: разброс углов не должен ухудшиться ---
        before = _spread_of(mask)
        after = _spread_of(_warp_mask(mask, ch_x=ch_small, mv_y=mv_small))
        if after > before + 0.25:
            # пробуем компоненты по отдельности
            after_ch = _spread_of(_warp_mask(mask, ch_x=ch_small)) \
                if ch_small is not None else before
            after_mv = _spread_of(_warp_mask(mask, mv_y=mv_small)) \
                if mv_small is not None else before
            if min(after_ch, after_mv) <= before + 0.25:
                if after_ch <= after_mv:
                    mv_small, after = None, after_ch
                else:
                    ch_small, after = None, after_mv
            else:
                break

        ch_full = _full_ch(ch_small) if ch_small is not None else None
        mv_full = None
        if mv_small is not None:
            mv_full = np.interp(np.arange(h, dtype=np.float64),
                                np.linspace(0, h - 1, len(mv_small)), mv_small)
            mv_full = np.clip(mv_full, -0.12, 0.12)
        image = _apply_maps(image, ch_x=ch_full, mv_y=mv_full)
        mask = _warp_mask(mask, ch_x=ch_small, mv_y=mv_small)
        report["h"] = report["h"] or ch_small is not None
        report["trap"] = report["trap"] or mv_small is not None
        report["spread_after"] = round(after, 2)
        if after < 0.8:
            break

    # --- вертикали: линейный сдвиг строк Cv(y) по медиане Хафа ---
    cv_small = None
    av, cnt = _vertical_line_angle(gsmall)
    report["v_angle"] = round(av, 2)
    if cnt >= 4 and 0.4 <= abs(av) <= 8.0:
        t = float(np.tan(np.radians(av)))
        ys = np.arange(mh, dtype=np.float64)
        cand = t * (ys - mh / 2.0)
        if float(np.abs(cand).max()) <= 0.06 * mw:
            g2 = _warp_mask(gsmall, cv_y=cand.astype(np.float32), replicate=True)
            av2, _ = _vertical_line_angle(g2)
            if abs(av2) <= abs(av) + 0.25:
                cv_small = cand
                report["v"] = True
                cv_full = np.interp(np.arange(h, dtype=np.float64),
                                    np.linspace(0, h - 1, len(cv_small)),
                                    cv_small) * (float(w) / mw)
                cv_full = np.clip(cv_full, -0.06 * w, 0.06 * w)
                image = _apply_maps(image, cv_y=cv_full)

    report["applied"] = report["h"] or report["v"] or report["trap"]
    return image, report


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
    """Выравнивает освещённость и делает фон ОДНОТОННЫМ (flat-field).

    Яркость делится на сглаженную оценку фона — уходят виньетка, тени,
    разводы, градиенты. Для «белых листов» (медиана тона высокая) фон
    подтягивается к белому (цель 252) + мягкий CLAHE для контраста текста.
    Для досок и тёмной бумаги тон сохраняется и только выравнивается
    (мягкое усиление, без CLAHE — он раздувает шум до белёсых разводов).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    h, w = l.shape
    # класс кадра: «белый лист» (медиана тона высокая) или доска/тёмная бумага.
    # Медиана устойчивее доли ярких пикселей: фото доски со светлой стеной
    # не должно пытаться выбелиться до 252
    med = float(np.median(l))
    p90 = float(np.percentile(l, 90))
    white_page = med >= 170.0 or p90 >= 240.0
    if white_page:
        k = int(np.clip(min(h, w) / 18.0, 21, 101))
    else:
        # доска/тёмная бумага: ядро больше, чтобы широкие рисунки мелом
        # не давали ореолов
        k = int(np.clip(min(h, w) / 10.0, 31, 151))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3.0, borderType=cv2.BORDER_REPLICATE)
    bgf = np.maximum(bg.astype(np.float32), 30.0)
    if white_page:
        gain = np.clip(252.0 / bgf, 0.55, 2.0)
    else:
        # тон сохраняем, только выравниваем: без выбеливания и без CLAHE
        # (CLAHE на тёмных тайлах раздувает шум до белёсых разводов)
        target = float(np.clip(np.percentile(bg, 55), 50.0, 190.0))
        gain = np.clip(target / bgf, 0.75, 1.30)
    l = np.clip(l.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if white_page:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
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
    """Заливает белым фоновые области, соединённые с краями кадра."""
    h, w = image.shape[:2]
    total = float(h * w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_frac = float((gray > 180).mean())
    # для светлых кадров фону разрешено до 55% (остатки стола/клинья),
    # но не больше: иначе заливка протекает сквозь клетку тетради на текст
    max_area = 0.55 if bright_frac > 0.5 else 0.30
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
    info = {"perspective": False, "skew_angle": 0.0, "dewarp": None,
            "enhanced": False, "steps": []}

    # 1. Ориентация: если кадр снят боком (текст вертикально) — повернуть
    if do_skew:
        result, oriented = _orient_upright(result)
        if oriented:
            info["steps"].append("кадр снят боком — повёрнут на 90 град.")

    # 2. Перспектива и деформация: ищем границы документа
    if do_perspective:
        quad = find_document_contour(result)
        if quad is not None:
            quad = _expand_quad(quad, 0.008)
            result = four_point_transform(result, quad)
            info["perspective"] = True
            info["steps"].append("границы найдены — перспектива и трапеция исправлены")

    # 3. Выравнивание по горизонтали и вертикали
    rotated = False
    if do_skew:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        mask, _, _ = _text_mask(gray)
        angle, conf = _projection_angle(mask)
        if conf < 2.0:
            angle, conf2 = _hough_median_angle(result)
            if conf2 < 0.2:
                angle = 0.0
        if abs(angle) >= 0.1:
            result = rotate_bound(result, angle)
            rotated = True
            info["skew_angle"] = round(float(angle), 2)
            info["steps"].append(
                "наклон %.2f град. исправлен — строки горизонтальны" % angle
            )

        # 3b. Построчное выпрямление: изгиб страницы, трапеция, вертикали
        result, drep = dewarp_lines(result)
        info["dewarp"] = drep
        if drep["h"]:
            rotated = True
            info["steps"].append(
                "изгиб строк выпрямлен (разброс %.1f -> %.1f град.)"
                % (drep["spread_before"], drep["spread_after"])
            )
        if drep["trap"]:
            rotated = True
            info["steps"].append("трапеция/перспектива строк выпрямлена")
        if drep["v"]:
            rotated = True
            info["steps"].append("вертикальные линии выровнены")

        # 3c. контрольный остаточный наклон
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        mask2, _, _ = _text_mask(gray)
        ang2, conf2 = _projection_angle(mask2, max_angle=4.0)
        if conf2 >= 2.0 and abs(ang2) >= 0.4:
            result = rotate_bound(result, ang2)
            rotated = True
            info["skew_angle"] = round(float(ang2), 2)
            info["steps"].append("остаточный наклон %.2f град. исправлен" % ang2)

    # 4. Чёткость и качество: шумоподавление + однотонный фон
    if do_enhance:
        result, den_ok = denoise(result, 5.0)
        result = balance_light(result)
        info["enhanced"] = True

    # 5. Заливка фона белым от краёв: остатки стола, виньетка, клинья
    if do_perspective or rotated:
        result = whiten_background(result)
        if do_perspective or info.get("dewarp", {}).get("h"):
            # мягкая чистка кромки: белым делаем только светлые пиксели
            h, w = result.shape[:2]
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            border = np.zeros((h, w), bool)
            border[:8, :] = True
            border[-8:, :] = True
            border[:, :8] = True
            border[:, -8:] = True
            result[border & (gray > 180)] = 255

    # 6. Обрезка артефактов поворота (белые клинья по углам)
    if rotated:
        result = trim_white_margins(result)

    # 7. Резкость — последней, чтобы не усиливать границы артефактов
    if do_enhance:
        result = sharpen(result, 0.7)
        info["steps"].append(
            "качество: шум убран, фон однотонный, свет выровнен, резкость повышена"
        )

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
