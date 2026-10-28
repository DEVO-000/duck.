"""
Сканер документов v2: находит лист (в том числе тетрадный, с почерком и клеткой)
на фото, выравнивает перспективу и делает «скан».

Что изменилось по сравнению с v1:
  • Поиск листа: рамка теперь «примагничивается» к реальному краю бумаги —
    каждая сторона четырёхугольника уточняется по градиенту яркости, поэтому
    рамка больше не улетает в стол и не отрезает края листа.
  • Ч/б режим: вместо одного жёсткого порога — локальный адаптивный порог
    (Sauvola) + автоподавление тетрадной клетки + деликатная чистка шума.
    Бледные чернила и тонкие штрихи больше не пропадают.

Установка:
    pip install opencv-python numpy

Запуск:
    python scanner.py photo.jpg                 -> photo_scan.png
    python scanner.py photo.jpg -o result.png
    python scanner.py photo.jpg --debug         (сохранит photo_debug.png с найденным контуром)
    python scanner.py photo.jpg --mode gray     (bw — чёрно-белый, gray — серый, color — цветной)
"""

import argparse
import os
import sys

import cv2
import numpy as np

MAX_SIDE = 2400  # длинная сторона готового скана, больше не нужно
DET_SIDE = 1000  # рабочая сторона для поиска листа


# ───────────────────────── геометрия ─────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    """Упорядочивает 4 точки: левый-верх, правый-верх, правый-низ, левый-низ."""
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Растягивает четырёхугольник в ровный прямоугольник."""
    rect = order_points(pts)
    tl, tr, br, bl = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    width, height = max(width, 2), max(height, 2)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (width, height))


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


# ───────────────────────── вспомогательное для поиска листа ─────────────────────────

def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    ff = mask.copy()
    frame = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, frame, (0, 0), 255)
    return mask | cv2.bitwise_not(ff)


def _main_component(mask: np.ndarray) -> np.ndarray:
    """Самый большой связный компонент (маска бумаги)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.zeros_like(mask)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8) * 255


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(21))
    mask = _main_component(mask)
    if cv2.countNonZero(mask) == 0:
        return mask
    return _fill_holes(mask)


def _quad_ok(quad: np.ndarray, img_area: float) -> bool:
    """Проверка: выпуклость, разумные углы, достаточная площадь."""
    q = order_points(quad)
    area = cv2.contourArea(q)
    if area < img_area * 0.10 or area > img_area * 1.01:
        return False
    # углы
    for i in range(4):
        p0, p1, p2 = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        v1, v2 = p1 - p0, p2 - p1
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1 or n2 < 1:
            return False
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        ang = np.degrees(np.arccos(cosang))
        if ang < 45 or ang > 135:
            return False
    return True


def _quads_from_mask(mask: np.ndarray, min_area: float):
    """Из бинарной маски достаёт четырёхугольники-кандидаты."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            break
        hull = cv2.convexHull(c)
        quads = []
        for eps in (0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
            approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
            if len(approx) == 4:
                quads.append(approx.reshape(4, 2).astype("float32"))
                break
            if len(approx) < 4:
                break
        # запасной вариант — повёрнутый прямоугольник
        quads.append(cv2.boxPoints(cv2.minAreaRect(hull)).astype("float32"))
        for quad in quads:
            if _quad_ok(quad, mask.shape[0] * mask.shape[1]) and cv2.contourArea(order_points(quad)) >= min_area:
                out.append(quad)
    return out


def _edge_map(gray: np.ndarray) -> np.ndarray:
    """Карта силы краёв (градиент яркости), сглаженная для устойчивости."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-6)
    return cv2.GaussianBlur(mag, (5, 5), 0)


def _side_support(mag: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> tuple:
    """
    Сила края вдоль стороны: в каждой из ~30 точек стороны ищем максимум
    градиента в полосе вдоль нормали. Возвращает (оценка, точки края).
    """
    h, w = mag.shape[:2]
    L = float(np.linalg.norm(p1 - p0))
    if L < 8:
        return 0.0, np.array([p0, p1], dtype="float32")
    d = (p1 - p0) / L
    nrm = np.array([d[1], -d[0]], dtype="float32")  # нормаль (наружу для CW-контура)

    n_samples = 30
    T = float(np.clip(0.05 * L, 6.0, 60.0))
    offs = np.linspace(-T, T, 21).astype("float32")
    ts = np.linspace(0.10, 0.90, n_samples).astype("float32")

    base = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]          # (S,2)
    pts = base[:, None, :] + offs[None, :, None] * nrm[None, None, :]  # (S,K,2)

    xs = np.clip(pts[..., 0], 0, w - 1)
    ys = np.clip(pts[..., 1], 0, h - 1)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx, fy = xs - x0, ys - y0
    vals = (mag[y0, x0] * (1 - fx) * (1 - fy) + mag[y0, x1] * fx * (1 - fy)
            + mag[y1, x0] * (1 - fx) * fy + mag[y1, x1] * fx * fy)

    best_idx = np.argmax(vals, axis=1)
    best = vals[np.arange(n_samples), best_idx]
    best_pts = pts[np.arange(n_samples), best_idx]

    # внутрь кадра попали не все точки — учитываем только валидные
    valid = ((base[:, 0] + offs[best_idx] * nrm[0]) >= -1) & \
            ((base[:, 0] + offs[best_idx] * nrm[0]) <= w) & \
            ((base[:, 1] + offs[best_idx] * nrm[1]) >= -1) & \
            ((base[:, 1] + offs[best_idx] * nrm[1]) <= h)
    if valid.sum() < 8:
        return 0.0, np.array([p0, p1], dtype="float32")
    return float(best[valid].mean()), best_pts[valid].astype("float32")


def _quad_support(mag: np.ndarray, quad: np.ndarray) -> float:
    q = order_points(quad)
    total = 0.0
    for i in range(4):
        s, _ = _side_support(mag, q[i], q[(i + 1) % 4])
        total += s
    return total / 4.0


def _fit_line(pts: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> tuple:
    """Прямая по точкам края (устойчиво, с отбраковкой выбросов).
    Возвращает (точка, направление) или None, если точек мало."""
    if len(pts) < 8:
        return None
    cur = pts.copy()
    for _ in range(3):
        mean = cur.mean(axis=0)
        uu, _, vv = np.linalg.svd(cur - mean, full_matrices=False)
        d = vv[0] / (np.linalg.norm(vv[0]) + 1e-9)
        dist = np.abs((cur - mean) @ np.array([-d[1], d[0]]))
        keep = dist <= max(2.5, 2.0 * np.median(dist))
        if keep.sum() < 8:
            break
        if keep.all():
            break
        cur = cur[keep]
    return mean, d


def _intersect(l1, l2, fallback):
    if l1 is None or l2 is None:
        return fallback
    (m1, d1), (m2, d2) = l1, l2
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype="float32")
    det = np.linalg.det(a)
    if abs(det) < 1e-4:
        return fallback
    t = np.linalg.solve(a, (m2 - m1).astype("float32"))
    return m1 + t[0] * d1


def _refine_quad(gray: np.ndarray, mag: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Примагничивает каждую сторону квадра к реальному краю бумаги."""
    q = order_points(quad)
    h, w = gray.shape[:2]
    diag = np.hypot(w, h)

    sides = []
    for i in range(4):
        p0, p1 = q[i], q[(i + 1) % 4]
        s, pts = _side_support(mag, p0, p1)
        sides.append((p0, p1, s, pts))

    lines = []
    for (p0, p1, s, pts) in sides:
        lines.append(_fit_line(pts, p0, p1) if len(pts) >= 10 else None)

    new_q = np.zeros((4, 2), dtype="float32")
    for i in range(4):
        # угол i = пересечение стороны (i-1) и стороны i
        new_q[i] = _intersect(lines[i - 1], lines[i], q[i])

    # защита от бреда: угол не дальше 12% диагонали от исходного
    max_jump = 0.12 * diag
    for i in range(4):
        if np.linalg.norm(new_q[i] - q[i]) > max_jump:
            new_q[i] = q[i]
    new_q[:, 0] = np.clip(new_q[:, 0], -0.02 * w, 1.02 * w)
    new_q[:, 1] = np.clip(new_q[:, 1], -0.02 * h, 1.02 * h)

    if not _quad_ok(new_q, h * w):
        return q
    return order_points(new_q)


def _quad_key(quad: np.ndarray) -> tuple:
    return tuple(np.round(order_points(quad).ravel() / 8).astype(int))


def find_document_contour(image: np.ndarray):
    """
    Ищет лист. Устойчив к почерку, клетке и предметам на столе: сначала строит
    маску бумаги, из неё — кандидатов-четырёхугольники, затем каждый кандидат
    уточняет по градиентам и выбирает тот, чьи стороны лучше всего совпадают
    с реальными краями. Возвращает 4 точки в координатах исходного изображения.
    """
    h, w = image.shape[:2]
    scale = DET_SIDE / max(h, w)
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    img_area = float(sh * sw)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # «стираем» почерк и клетку, чтобы они не мешали границам
    k = max(9, (max(sh, sw) // 40) | 1)
    closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, _ellipse(k))
    closed = cv2.GaussianBlur(closed, (7, 7), 0)

    masks = []

    # 1) светлая бумага на более тёмном фоне
    _, m = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(_clean_mask(m))

    # 2) бумага почти бесцветная, стол/ткань — цветные
    sat = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1], (7, 7), 0)
    _, m2 = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(_clean_mask(cv2.bitwise_and(m, m2)))

    # 3) границы на «очищенной» картинке
    e = cv2.Canny(closed, 30, 90)
    e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=2)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, _ellipse(9))
    masks.append(e)

    mag = _edge_map(closed)

    candidates = []
    seen = set()
    min_area = img_area * 0.12
    for mask in masks:
        for quad in _quads_from_mask(mask, min_area):
            key = _quad_key(quad)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(quad)

    if not candidates:
        return None

    best_quad, best_score = None, 0.0
    for quad in candidates:
        refined = _refine_quad(gray, mag, quad)
        for q in (quad, refined):
            area = cv2.contourArea(order_points(q))
            support = _quad_support(mag, q)
            # чуть-чуть предпочитаем покрупнее при равной силе края
            score = support * (0.95 + 0.05 * np.sqrt(area / img_area))
            corners = order_points(q)
            near_frame = all(
                np.linalg.norm(corners[i] - np.array([(0, 0), (sw, 0), (sw, sh), (0, sh)][i], dtype="float32"))
                < 0.02 * max(sh, sw) for i in range(4)
            )
            if near_frame:
                score *= 0.55  # «весь кадр» — подозрительно
            if score > best_score:
                best_quad, best_score = q, score

    if best_quad is None:
        return None
    return (order_points(best_quad) / scale).astype("float32")


# ───────────────────────── обработка «под скан» ─────────────────────────

def _limit_size(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    k = MAX_SIDE / max(h, w)
    if k < 1:
        return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return img


def _flatten_light(img: np.ndarray) -> np.ndarray:
    """Убирает тени и неравномерный свет: делит картинку на оценку фона бумаги."""
    h, w = img.shape[:2]
    k = 600.0 / max(h, w)
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
    ks = max(7, (max(small.shape[:2]) // 20) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, _ellipse(ks))  # стираем штрихи ручки
    bg = cv2.GaussianBlur(bg, (0, 0), ks / 3)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    bg = np.maximum(bg, 1)
    return cv2.divide(img, bg, scale=255)


def _remove_grid(ink: np.ndarray, gray: np.ndarray, flat_bgr: np.ndarray, white: float,
                 strict: np.ndarray = None) -> np.ndarray:
    """Убирает тетрадную клетку, НЕ трогая почерк.

    Клетка — это ПЕРИОДИЧЕСКАЯ решётка прямых линий, поэтому ищем её как
    решётку: оцениваем угол поворота страницы, шаг и положение линий по
    проекционным профилям светлых пикселей. Удаляем только то, что целиком
    лежит в полосах решётки, тонкое, светлое и без тёмного ядра.
    У бледной ручки и клетки может быть одинаковая яркость, но слова не
    укладываются в тонкие полосы решётки — поэтому почерк выживает.
    Если решётка не распознана уверенно, сетка НЕ удаляется вовсе
    (клетка на скане лучше, чем потерянные буквы).
    """
    H, W = ink.shape[:2]
    S = max(H, W)
    line_thr = max(118.0, white - 0.55 * (white - 30.0))  # граница «светлого»
    core_thr = line_thr - 18.0                            # «тёмное ядро» штриха

    light = ((gray >= line_thr) & (ink > 0)).astype(np.uint8) * 255

    # ── 1) угол решётки: перебираем углы, ищем самый «пиковый» профиль ──
    def _rot_profiles(mask, angle_deg, down):
        h, w = mask.shape[:2]
        c = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        rot = cv2.warpAffine(mask, c, (w, h), flags=cv2.INTER_NEAREST)
        Px = rot.sum(axis=0).astype(np.float64)
        Py = rot.sum(axis=1).astype(np.float64)
        return Px, Py

    def _peakiness(prof, lo, hi):
        p = prof - np.convolve(prof, np.ones(31) / 31, mode="same")
        r = np.correlate(p, p, mode="full")[len(p) - 1:]
        r = r / (r[0] + 1e-9)
        if hi >= len(r):
            hi = len(r) - 1
        seg = r[lo:hi]
        if len(seg) == 0:
            return 0.0
        return float(seg.max())

    small4 = cv2.resize(light, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    best_theta, best_score = 0.0, 0.0
    lo4, hi4 = 8, 26  # шаг клетки 40..110 px в масштабе /4
    for theta in np.arange(-8.0, 8.01, 0.5):
        Px, Py = _rot_profiles(small4, theta, 4)
        sc = _peakiness(Px, lo4, hi4) + _peakiness(Py, lo4, hi4)
        if sc > best_score:
            best_theta, best_score = float(theta), sc
    for theta in np.arange(best_theta - 0.5, best_theta + 0.51, 0.1):  # уточнение
        Px, Py = _rot_profiles(small4, theta, 4)
        sc = _peakiness(Px, lo4, hi4) + _peakiness(Py, lo4, hi4)
        if sc > best_score:
            best_theta, best_score = float(theta), sc
    if best_score < 0.9:
        return ink  # периодической решётки нет (клетка слабая или её нет) —
                    # ничего не трогаем: клетка на скане лучше, чем потерянный текст

    # ── 2) profiles в масштабе /2 при найденном угле ──
    light2 = cv2.resize(light, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    h2, w2 = light2.shape[:2]
    c_fwd = cv2.getRotationMatrix2D((w2 / 2, h2 / 2), best_theta, 1.0)
    rot2 = cv2.warpAffine(light2, c_fwd, (w2, h2), flags=cv2.INTER_LINEAR)

    bands = np.zeros(light2.shape, np.uint8)
    _strip_lattice(rot2 > 0, vertical=True, bands=bands)
    _strip_lattice(rot2 > 0, vertical=False, bands=bands)
    if cv2.countNonZero(bands) == 0:
        return ink

    # возвращаем полосы в исходную систему координат (ОБРАТНЫЙ поворот!)
    c_inv = cv2.getRotationMatrix2D((w2 / 2, h2 / 2), -best_theta, 1.0)
    bands_full = cv2.warpAffine(bands, c_inv, (w2, h2), flags=cv2.INTER_NEAREST)
    band_mask = cv2.resize(bands_full, (W, H), interpolation=cv2.INTER_LINEAR) > 100

    # ── 3) удаление ──
    # Phase 1: изолированные обрывки, ЦЕЛИКОМ лежащие в полосах решётки
    # (всё векторизовано через bincount — компонентов могут быть сотни тысяч)
    out = ink.copy()
    frag_cap = max(150, int(S * S * 0.0007))
    for _ in range(2):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        if n <= 1:
            break
        idx = labels.ravel()
        cnt = np.bincount(idx, minlength=n).astype(np.float64)
        cnt[cnt == 0] = 1
        sum_band = np.bincount(idx, weights=band_mask.ravel().astype(np.float64), minlength=n)
        sum_dark = np.bincount(idx, weights=(gray < core_thr).ravel().astype(np.float64), minlength=n)
        sum_gray = np.bincount(idx, weights=gray.ravel().astype(np.float64), minlength=n)
        mean_gray = sum_gray / cnt
        inband = sum_band / cnt
        kill = (np.arange(n) > 0) \
            & (stats[:, cv2.CC_STAT_AREA] <= frag_cap) \
            & (np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) <= 0.06 * S) \
            & (inband >= 0.85) \
            & (mean_gray >= line_thr + 6) \
            & (sum_dark / cnt < 0.03)
        if not kill.any():
            break
        out[kill[labels]] = 0

    # Phase 2: попиксельно — тонкие (в перпендикуляре) светлые структуры в полосах.
    # Штрих буквы, пересекающий полосу, продолжается за её пределами, т.е. «толстый»
    # в перпендикулярном направлении — и потому не удаляется. Линия клетки тонкая.
    band_u8 = band_mask.astype(np.uint8) * 255
    band_e = cv2.dilate(band_u8, _ellipse(3)) > 0
    # направленное замыкание склеивает пунктир клетки в сплошную линию,
    # после чего opening длиной 51px находит её; текст после замыкания
    # тоже удлиняется, но его защищает проверка «тулова буквы»
    Hcont = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    Vcont = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    Hopen = cv2.morphologyEx(Hcont, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 51)))
    Vopen = cv2.morphologyEx(Vcont, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (51, 1)))
    if cv2.countNonZero(Hopen) or cv2.countNonZero(Vopen):
        Hline = cv2.dilate(Hopen, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))) > 0
        Vline = cv2.dilate(Vopen, cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))) > 0
        solid = out > 0
        solidf = solid.astype(np.float32)
        vrun = _run_lengths(solid, axis=0)  # вертикальные пробеги
        hrun = _run_lengths(solid, axis=1)  # горизонтальные пробеги
        near_dark = cv2.dilate((gray < core_thr).astype(np.uint8), _ellipse(7)) > 0
        lightpx = gray >= line_thr

        # «тело буквы»: чернила за пределами полосы в перпендикулярном окне.
        # У линии клетки сверху/снизу пусто, у буквы, лежащей на линии, — нет.
        # Сглаживание вдоль линии гасит всплески от пересечений с другой сеткой.
        # окно меньше полшага клетки, чтобы не задевать соседние строки текста
        M45v = cv2.boxFilter(solidf, -1, (1, 45), normalize=False)
        M19v = cv2.boxFilter(solidf, -1, (1, 19), normalize=False)
        beyondV = cv2.boxFilter(np.maximum(M45v - M19v, 0), -1, (13, 1))
        bodyV = beyondV > 12  # рядом вертикальное «туловище» буквы
        M45h = cv2.boxFilter(solidf, -1, (45, 1), normalize=False)
        M19h = cv2.boxFilter(solidf, -1, (19, 1), normalize=False)
        beyondH = cv2.boxFilter(np.maximum(M45h - M19h, 0), -1, (1, 13))
        bodyH = beyondH > 12

        remove2 = ((Hline & band_e & (vrun <= 11) & ~bodyV)
                   | (Vline & band_e & (hrun <= 11) & ~bodyH))
        remove2 &= lightpx & ~near_dark & solid
        out[remove2] = 0

    # Phase 2.5: в полосах решётки стираем всё светлое, что не выжило при
    # СТРОГОМ пороге (жирная ручка). «Туловище» буквы и темнота защищают:
    # у текста есть тело в перпендикуляре, у линии клетки — нет.
    if strict is not None:
        strict_ink = strict > 0
        remove3 = band_e & solid & lightpx & ~near_dark & ~bodyV & ~bodyH & ~strict_ink
        out[remove3] = 0

    # Phase 3: мелкие тёмные обрывки (шумовые точки решётки), целиком лежащие
    # в полосах. Точки и запятые ручки спасает цвет: они насыщенно-синие/красные.
    blue = flat_bgr[:, :, 0].astype(np.float32)
    red = flat_bgr[:, :, 2].astype(np.float32)
    protect_dark = cv2.dilate(cv2.erode(out, _ellipse(9)), _ellipse(13))
    speck_cap = max(80, int(S * S * 0.0002))
    for _ in range(2):  # после удаления цепочки распадаются — второй заход
        n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        if n <= 1:
            break
        idx = labels.ravel()
        cnt = np.bincount(idx, minlength=n).astype(np.float64)
        cnt_safe = np.maximum(cnt, 1)
        sum_band = np.bincount(idx, weights=band_e.ravel().astype(np.float64), minlength=n)
        sum_prot = np.bincount(idx, weights=(protect_dark > 0).ravel().astype(np.float64), minlength=n)
        sum_B = np.bincount(idx, weights=blue.ravel().astype(np.float64), minlength=n)
        sum_R = np.bincount(idx, weights=red.ravel().astype(np.float64), minlength=n)
        # быстрый векторный пре-фильтр с запасом, точные медианы — только для кандидатов
        cand = (np.arange(n) > 0) \
            & (stats[:, cv2.CC_STAT_AREA] <= speck_cap) \
            & (np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) <= 0.045 * S) \
            & (sum_band / cnt_safe >= 0.9) \
            & (sum_prot / cnt_safe <= 0.3) \
            & (sum_B / cnt_safe - sum_R / cnt_safe < 26) & (sum_R / cnt_safe - sum_B / cnt_safe < 20)
        cand_ids = np.where(cand)[0]
        if len(cand_ids) == 0:
            break
        removed = False
        for i in cand_ids:
            comp = labels == i
            mB = float(np.median(blue[comp])); mR = float(np.median(red[comp]))
            if mB - mR >= 22 or mR - mB >= 18:
                continue  # насыщенная ручка (синяя или красная)
            out[comp] = 0
            removed = True
        if not removed:
            break
    return out


def _run_lengths(mask: np.ndarray, axis: int) -> np.ndarray:
    """Карта длин пробегов True-пикселей вдоль axis (0 — по столбцам, 1 — по строкам)."""
    res = np.zeros(mask.shape, np.int32)
    if axis == 0:
        for c in range(mask.shape[1]):
            res[:, c] = _runs1d(mask[:, c])
    else:
        for r in range(mask.shape[0]):
            res[r, :] = _runs1d(mask[r, :])
    return res


def _runs1d(v: np.ndarray) -> np.ndarray:
    """Длины пробегов единиц в одномерном булевом массиве."""
    d = np.diff(np.concatenate(([0], v.view(np.int8), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    out = np.zeros(len(v), np.int32)
    for s, e in zip(starts, ends):
        out[s:e] = e - s
    return out


def _strip_lattice(rot: np.ndarray, vertical: bool, bands: np.ndarray) -> None:
    """Полосы решётки через ТРЕКИНГ линий между сегментами.

    В каждом сегменте перпендикулярного направления — пики профиля с проверкой
    покрытия (линия клетки тянется через весь сегмент, у строки текста между
    словами промежутки). Затем пики сшиваются в треки: линия клетки живёт
    через большинство сегментов и дрейфует плавно (лист изогнут — позиции
    между сегментами немного плывут, трек это отслеживает). Строка текста
    короче страницы и не проходит ни по длине, ни по плавности."""
    Hb, Wb = bands.shape[:2]
    step, over = 120, 60
    lo, hi = 17, 38     # шаг клетки 34..76 px в масштабе /2
    match_tol = 14      # допустимый скачок трека между сегментами, px /2
    min_alive = 0.55    # трек должен жить >=55% сегментов

    if vertical:
        spans = [(y0, min(Hb, y0 + step + over)) for y0 in range(0, Hb, step)]
        cross = Wb
    else:
        spans = [(x0, min(Wb, x0 + step + over)) for x0 in range(0, Wb, step)]
        cross = Hb
    n_spans = len(spans)

    strip_peaks = []
    for a0, a1 in spans:
        if a1 - a0 < 40:
            strip_peaks.append([])
            continue
        seg = rot[a0:a1, :] if vertical else rot[:, a0:a1]
        axis = 0 if vertical else 1
        prof = seg.sum(axis=axis).astype(np.float64)
        n = len(prof)
        ps = np.convolve(prof, np.ones(5) / 5, mode="same")
        pmax = float(ps.max())
        if pmax <= 0:
            strip_peaks.append([])
            continue
        anymask = seg > 0
        peaks = []
        for i in range(2, n - 2):
            if ps[i] < ps[i - 1] or ps[i] <= ps[i + 1] or ps[i] < 0.15 * pmax:
                continue
            w0, w1 = max(0, i - 3), min(n, i + 4)
            # покрытие по ПОПЕРЕЧНИКУ сегмента: линия клетки видна почти везде,
            # у строки текста между словами промежутки
            if vertical:   # i — позиция по x, покрытие считаем по y сегмента
                cov = float(anymask[:, w0:w1].any(axis=1).mean())
            else:          # i — позиция по y, покрытие считаем по x сегмента
                cov = float(anymask[w0:w1, :].any(axis=0).mean())
            if cov >= 0.50:
                peaks.append(i)
        # разредить близкие пики (оставить сильнейший в окне 12px)
        peaks.sort(key=lambda i: -ps[i])
        thinned = []
        for i in peaks:
            if all(abs(i - j) >= 12 for j in thinned):
                thinned.append(i)
        strip_peaks.append(sorted(thinned))

    # трекинг: тянем треки слева направо с предсказанием по скорости
    # track: [pred, last, count, points, last_delta]
    tracks = []
    for si, peaks in enumerate(strip_peaks):
        used = set()
        for tr in tracks:
            if tr[4] == si:
                continue
            pred = tr[0]
            best, best_d = None, match_tol + 1
            for p in peaks:
                if p in used:
                    continue
                d = abs(p - pred)
                if d < best_d:
                    best, best_d = p, d
            if best is not None:
                used.add(best)
                delta = best - tr[1]
                tr[4] = si
                tr[0] = best + int(np.clip(delta, -6, 6))  # предсказание на след. сегмент
                tr[1] = best
                tr[2] += 1
                tr[3].append((si, best))
        for p in peaks:
            if p not in used:
                tracks.append([p, p, 1, [(si, p)], si])

    good_tracks = []
    for tr in tracks:
        cnt = tr[2]
        if cnt < max(4, int(min_alive * n_spans)):
            continue
        ys = [p for _, p in tr[3]]
        if max(ys) - min(ys) > 70:  # слишком прыгучая — не линия клетки
            continue
        good_tracks.append(tr[3])

    for points in good_tracks:
        for si, pos in points:
            a0, a1 = spans[si]
            if vertical:
                cv2.line(bands, (pos, a0), (pos, a1), 255, 11)
            else:
                cv2.line(bands, (a0, pos), (a1, pos), 255, 11)




def _sauvola_ink(gray32: np.ndarray, win: int, k: float) -> np.ndarray:
    """Локальный адаптивный порог Sauvola: чернилом считается то, что заметно
    темнее СВОЕГО окружения — бледные чернила и тонкие штрихи выживают."""
    mean = cv2.boxFilter(gray32, -1, (win, win), borderType=cv2.BORDER_REPLICATE)
    mean2 = cv2.boxFilter(gray32 * gray32, -1, (win, win), borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(mean2 - mean * mean, 0))
    thr = mean * (1.0 + k * (std / 128.0 - 1.0))
    return (gray32 < thr).astype(np.uint8) * 255


def _despeckle(ink: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return ink
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels].astype(np.uint8) * 255


def to_black_white(image: np.ndarray, mode: str = "bw", trim: float = 0.005) -> np.ndarray:
    """
    mode:
      bw    — чистый чёрно-белый: клетка и шум убраны, текст остаётся (даже бледный)
      gray  — серый: бумага белая, почерк читаемый (лучше всего для рукописи)
      color — цветной: бумага белая, ручка сохраняет цвет
    """
    # срезаем узкую полоску по краю — там часто остаётся кусок стола
    h, w = image.shape[:2]
    dy, dx = int(h * trim), int(w * trim)
    if h - 2 * dy > 20 and w - 2 * dx > 20:
        image = image[dy:h - dy, dx:w - dx]

    image = _limit_size(image)
    flat = _flatten_light(image)
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)

    sample = gray[::4, ::4]
    white = float(max(170.0, np.percentile(sample, 90)))   # уровень бумаги

    if mode == "color":
        out = np.empty_like(flat)
        for c in range(3):
            ch = flat[:, :, c].astype(np.float32)
            wc = max(float(np.percentile(ch[::4, ::4], 90)), 1.0)
            out[:, :, c] = np.clip(ch * (255.0 / wc), 0, 255)
        return out.astype(np.uint8)

    if mode == "gray":
        black = float(np.percentile(sample, 0.2))
        scale = 255.0 / max(white - black, 1.0)
        out = cv2.convertScaleAbs(gray, alpha=scale, beta=-black * scale)
        out[out > 252] = 255
        return out

    # ── bw: адаптивный порог Sauvola ──
    g = gray.astype(np.float32)
    S = max(g.shape[:2])
    win = int(np.clip(S // 48, 25, 81))
    win += (win + 1) % 2

    ink = None
    # страховка от «совсем бледных» чернил: если чернил почти не нашлось,
    # повторяем с большей чувствительностью
    for k in (0.06, 0.09, 0.13):
        ink = _sauvola_ink(g, win, k)
        frac = float(cv2.countNonZero(ink)) / ink.size
        if frac >= 0.004:
            break

    strict = _sauvola_ink(g, win, 0.20)  # только жирные штрихи — как «контрольный»
    ink = _remove_grid(ink, gray, flat, white, strict)
    ink = _despeckle(ink, min_area=max(3, int(S * S * 2.5e-7)))
    # слегка «набираем» штрих, чтобы скан был читабельным
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)

    return np.where(ink > 0, 0, 255).astype(np.uint8)


# ───────────────────────── командная строка ─────────────────────────

def scan(path: str, out: str, mode: str = "bw", debug: bool = False) -> None:
    image = cv2.imread(path)
    if image is None:
        sys.exit(f"Не удалось открыть изображение: {path}")

    pts = find_document_contour(image)

    if pts is None:
        print("Лист не найден — обрабатываю всё изображение целиком.")
        warped = image
    else:
        warped = four_point_transform(image, pts)
        if debug:
            dbg = image.copy()
            cv2.polylines(dbg, [pts.astype(int)], True, (0, 255, 0), max(3, image.shape[1] // 300))
            base, _ = os.path.splitext(path)
            cv2.imwrite(base + "_debug.png", dbg)

    result = to_black_white(warped, mode)
    cv2.imwrite(out, result)
    print(f"Готово: {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Сканер документов по фото")
    p.add_argument("input", help="путь к фото")
    p.add_argument("-o", "--output", help="куда сохранить результат")
    p.add_argument("--mode", choices=["bw", "gray", "color"], default="bw")
    p.add_argument("--debug", action="store_true", help="сохранить картинку с найденным контуром")
    args = p.parse_args()

    out = args.output
    if not out:
        base, _ = os.path.splitext(args.input)
        out = base + "_scan.png"
    scan(args.input, out, args.mode, args.debug)


if __name__ == "__main__":
    main()
