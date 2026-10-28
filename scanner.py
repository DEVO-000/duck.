"""
Сканер документов v2: находит лист (в том числе тетрадный, с почерком и клеткой)
на фото, ТОЧНО выставляет рамку по углам листа и делает «скан».

Что добавлено к v1 (по вашим фото-примерам):
  * детекция бумаги по 4 маскам: яркость, низкая насыщенность, цветовое
    расстояние до «цвета бумаги», границы Canny — лист находится даже на
    светлой скатерти/столе;
  * отбраковка «рамки всего кадра» и тонких полос (стена, планки);
  * проверка опоры краёв на градиенте: рамка подтверждается реальными
    границами листа, а не текстурой фона;
  * точная докрутка каждого края по градиенту (RANSAC-прямые + пересечение):
    углы встают ровно по листу, даже если угол закрыт ручкой или пальцем;
  * соседние листы больше не приклеиваются к основному (меньшая морфология).

Совместим с bot.py: экспортирует find_document_contour, four_point_transform,
order_points, to_black_white.

Установка:
    pip install opencv-python numpy

Запуск:
    python scanner.py photo.jpg                 -> photo_scan.png
    python scanner.py photo.jpg -o result.png
    python scanner.py photo.jpg --debug         (photo_debug.png с рамкой)
    python scanner.py photo.jpg --mode gray     (bw/gray/color)
"""

import argparse
import os
import sys

import cv2
import numpy as np

MAX_SIDE = 2400  # длинная сторона готового скана, больше не нужно

_COARSE = 900.0   # размер для грубого поиска листа
_FINE = 1600.0    # размер для точной докрутки краёв


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


# ───────────────────────── вспомогательные ─────────────────────────

def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _resize_to(image: np.ndarray, long_side: float) -> np.ndarray:
    h, w = image.shape[:2]
    k = long_side / max(h, w)
    if k >= 1:
        return image
    return cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _profile(img: np.ndarray, xs, ys) -> np.ndarray:
    """Билинейная выборка значений img в точках (xs, ys)."""
    h, w = img.shape[:2]
    xm = np.clip(xs, 0, w - 1).astype(np.float32)
    ym = np.clip(ys, 0, h - 1).astype(np.float32)
    return cv2.remap(img, xm.reshape(1, -1), ym.reshape(1, -1), cv2.INTER_LINEAR).ravel()


def _line_intersection(l1, l2):
    p1, d1 = l1
    p2, d2 = l2
    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    d1 = d1 / n1
    d2 = d2 / n2
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-6:
        return None
    dp = p2 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
    return p1 + t * d1


def _angle_at(a, b, c) -> float:
    v1 = a - b
    v2 = c - b
    cosv = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))


def _valid_quad(q: np.ndarray, img_area: float, diag: float) -> bool:
    q = np.asarray(q, dtype=np.float64)
    area = cv2.contourArea(q.astype(np.float32))
    if area < img_area * 0.08:
        return False
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    if min(sides) < diag * 0.12 or min(sides) / max(sides) < 0.22:
        return False
    for i in range(4):
        ang = _angle_at(q[i - 1], q[i], q[(i + 1) % 4])
        if not (48.0 <= ang <= 132.0):
            return False
    # выпуклость (все попарные векторные произведения одного знака)
    signs = []
    for i in range(4):
        v1 = q[(i + 1) % 4] - q[i]
        v2 = q[(i + 2) % 4] - q[(i + 1) % 4]
        signs.append(v1[0] * v2[1] - v1[1] * v2[0])
    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        return False
    return True


# ───────────────────────── поиск листа ─────────────────────────

def _quad_from_contour(c: np.ndarray):
    """Квад из контура: approxPolyDP с лестницей eps, резерв — minAreaRect."""
    peri = cv2.arcLength(c, True)
    best = None
    for eps in (0.008, 0.012, 0.018, 0.025, 0.035, 0.05, 0.07, 0.09):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            best = approx.reshape(4, 2)
            break
        if len(approx) < 4:
            break
    if best is None:
        hull = cv2.convexHull(c)
        hperi = cv2.arcLength(hull, True)
        for eps in (0.01, 0.018, 0.028, 0.045, 0.07):
            approx = cv2.approxPolyDP(hull, eps * hperi, True)
            if len(approx) == 4:
                best = approx.reshape(4, 2)
                break
            if len(approx) < 4:
                break
    if best is None:
        rect = cv2.minAreaRect(c)
        ra = rect[1][0] * rect[1][1]
        if ra > 0 and cv2.contourArea(c) / ra > 0.80:
            best = cv2.boxPoints(rect)
    return None if best is None else best.astype(np.float32)


def _paper_masks(small: np.ndarray):
    """Несколько бинарных масок «что похоже на бумагу»."""
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2].astype(np.float32)
    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)

    masks = []

    # 1) светлая бумага на более тёмном фоне
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(m)

    # 2) бумага почти бесцветная, стол/ткань — цветные
    _, m = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(m)

    # 3) расстояние до «цвета бумаги» (медиана по светлым малонасыщенным пикселям)
    sel = (S < 70) & (V > np.percentile(V, 60))
    if sel.sum() > 50:
        lr = float(np.median(L[sel]))
        ar = float(np.median(A[sel]))
        br = float(np.median(B[sel]))
        dist = np.sqrt(0.35 * (L - lr) ** 2 + (A - ar) ** 2 + (B - br) ** 2).astype(np.float32)
        dist = cv2.GaussianBlur(dist, (5, 5), 0)
        dmax = float(dist.max())
        if dmax > 1e-3:
            dist8 = np.clip(dist * (255.0 / dmax), 0, 255).astype(np.uint8)
            _, m = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            masks.append(m)

    # 4) границы на «очищенной» картинке
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, _ellipse(13))
    e = cv2.Canny(closed, 30, 90)
    e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=2)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, _ellipse(9))
    masks.append(e)

    # 5-6) гладкая + светлая/бесцветная область: бумага гладкая, а кружево,
    # вышивка и ткань — «шероховатые». Отделяет лист от светлой скатерти.
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    tex = cv2.GaussianBlur(lap, (0, 0), 4.0)
    tex8 = np.clip(tex * 2.0, 0, 255).astype(np.uint8)
    t_thr, _ = cv2.threshold(tex8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    smooth = tex8 <= t_thr
    b_thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    s_thr, _ = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(((smooth & (blur >= b_thr)) * 255).astype(np.uint8))
    masks.append(((smooth & (S <= s_thr)) * 255).astype(np.uint8))

    return masks


def _quad_mean_gray(gray: np.ndarray, q: np.ndarray) -> float:
    """Средняя яркость внутри квада (бумага светлая — отбраковывает доску/темноту)."""
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [np.asarray(q, np.float32).astype(np.int32)], 255)
    val = cv2.mean(gray, mask)[0]
    return float(val)


def _shrink_quad(small: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    «Усадка» рамки: эталон бумаги берём из ядра квада (центральная область —
    там точно лист), затем внутри рамки оставляем пиксели, похожие на эталон
    в Lab с допуском (яркость, теплота/холодность). Отсекает светлую скатерть,
    серое дерево, край соседнего листа и предметы, попавшие в рамку.
    """
    h, w = small.shape[:2]
    x, y, bw, bh = cv2.boundingRect(np.asarray(q, np.float32))
    pad = int(0.08 * max(bw, bh))  # запас наружу: усадка может и ВЕРНУТЬ отрезанное
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    roi = small[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    if roi.size == 0:
        return q

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab).astype(np.float32)
    qm = np.zeros((rh, rw), np.uint8)
    cv2.fillPoly(qm, [(np.asarray(q, np.float32) - [x0, y0]).astype(np.int32)], 255)
    sel = qm > 0  # ядро-квад: из него берём эталон и точку затопления

    # ядро = центральная часть квада, сжатая к центроиду: там гарантированно бумага
    qc = np.asarray(q, np.float32) - [x0, y0]
    centroid = qc.mean(axis=0)
    core = centroid + (qc - centroid) * 0.45
    cm = np.zeros((rh, rw), np.uint8)
    cv2.fillPoly(cm, [core.astype(np.int32)], 255)
    core_sel = cm > 0
    if np.count_nonzero(core_sel) < 300:
        return q

    Lc = lab[:, :, 0][core_sel]
    Ac = lab[:, :, 1][core_sel]
    Bc = lab[:, :, 2][core_sel]
    lr, ar, br = (float(np.median(v)) for v in (Lc, Ac, Bc))
    sl, sa, sb = (float(np.std(v)) for v in (Lc, Ac, Bc))

    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    dist = np.sqrt(
        ((L - lr) / (2.8 * sl + 20.0)) ** 2 * 0.7
        + ((A - ar) / (1.2 * sa + 4.0)) ** 2
        + ((B - br) / (1.2 * sb + 4.0)) ** 2
    ).astype(np.float32)

    if np.count_nonzero(sel) < 500:
        return q
    d100 = dist * 100.0

    # Геодезическая маска: затопляем из ядра по «похоже на бумагу» (рыхлый
    # порог — пропускает тени на листе). Барьер — только края, где сама
    # «бумажность» скачет (граница лист-фон): тень на бумаге меняет dist
    # плавно и затопление проходит, а переход на фон/кружево — резко.
    loose = d100 <= 190.0
    canny = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0), 25, 70)
    canny = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1)

    # Шаг 1: затопление из ядра ВНУТРИ квада; края Canny блокируют переход
    # на фон (тень на бумаге отличается плавно и остаётся внутри).
    ok1 = (loose & (canny == 0) & sel).astype(np.uint8)
    n1, lab1 = cv2.connectedComponents(ok1, connectivity=8)
    core_ids = np.unique(lab1[core_sel])
    core_ids = core_ids[core_ids != 0]
    if core_ids.size == 0:
        m = np.zeros((rh, rw), np.uint8)
        m[sel] = np.where(d100[sel] <= 105.0, 255, 0).astype(np.uint8)
    else:
        m = np.isin(lab1, core_ids).astype(np.uint8) * 255
        m[~sel] = 0

    # Шаг 2: возврат отрезанного снаружи (например, тень у края листа):
    # приклеиваем компоненты «почти бумаги» (dist <= 1.25), касающиеся маски.
    ok2 = (d100 <= 125.0).astype(np.uint8)
    n2, lab2 = cv2.connectedComponents(ok2, connectivity=8)
    touch = cv2.dilate(m, _ellipse(9)) > 0
    attach_ids = np.unique(lab2[touch & (lab2 > 0)])
    if attach_ids.size:
        m2 = np.isin(lab2, attach_ids).astype(np.uint8) * 255
        m = cv2.bitwise_or(m, m2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(9))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return q
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.40 * max(np.count_nonzero(sel), 1):
        return q  # маска развалилась — не трогаем рамку
    q2 = _quad_from_contour(c)
    if q2 is None:
        return q
    q2 = q2 + np.array([x0, y0], dtype=np.float32)
    if not _valid_quad(q2, float(h * w), float(np.hypot(h, w))):
        return q
    return q2


def _candidates(small: np.ndarray):
    """Кандидаты-квады из всех масок: (quad, area_frac, fit, tag)."""
    h, w = small.shape[:2]
    img_area = float(h * w)
    diag = float(np.hypot(h, w))
    gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    out = []
    for tag, mask in enumerate(_paper_masks(small)):
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(5))
        # без close: закрытие склеивает соседние листы через тёмный зазор;
        # дырки от клетки/почерка внешнему контуру не мешают
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        for c in contours:
            ca = cv2.contourArea(c)
            if ca < img_area * 0.10:
                continue
            q = _quad_from_contour(c)
            if q is None:
                continue
            qa = cv2.contourArea(q)
            if qa < img_area * 0.10:
                continue
            sides = sorted(float(np.linalg.norm(q[i] - q[(i + 1) % 4])) for i in range(4))
            if sides[0] < diag * 0.10 or sides[0] / sides[-1] < 0.25:
                continue  # слишком тонкая полоса — не лист
            if _quad_mean_gray(gray_s, q) < 118.0:
                continue  # внутри темно (доска, темный стол) — не бумага
            hull_area = cv2.contourArea(cv2.convexHull(c))
            fit = min(qa, hull_area) / max(qa, hull_area) if hull_area > 0 else 0.0
            out.append((q, qa / img_area, fit, tag))
    return out


def _side_support(gray: np.ndarray, mag: np.ndarray, p, r) -> float:
    """Доля точек одной грани, где рядом есть реальный перепад яркости."""
    h, w = gray.shape
    d = r - p
    ln = float(np.linalg.norm(d))
    if ln < 1:
        return 0.0
    u = d / ln
    n = np.array([u[1], -u[0]], dtype=np.float64)
    total = 0
    good = 0
    for t in np.linspace(0.08, 0.92, max(6, int(ln / 8))):
        pt = p + t * d
        total += 1
        bestv = 0.0
        for s in range(-3, 4):
            x = pt[0] + s * n[0]
            y = pt[1] + s * n[1]
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                v = mag[yi, xi]
                if v > bestv:
                    bestv = float(v)
        if bestv > 30.0:
            good += 1
    return good / max(total, 1)


def _support(gray: np.ndarray, mag: np.ndarray, q: np.ndarray) -> float:
    """Доля точек периметра квада, где рядом с гранью есть реальный перепад яркости."""
    vals = [_side_support(gray, mag, q[i], q[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(vals))


def _hough_edge(gray: np.ndarray, p, r, n_in, band_w):
    """
    Резерв для слабой грани: ищем в полосе шириной band_w внутрь длинную
    прямую, параллельную грани, у которой внутрь светлее (бумага).
    Возвращает (point, dir) или None.
    """
    h, w = gray.shape
    d = r - p
    ln = float(np.linalg.norm(d))
    if ln < 30 or band_w < 8:
        return None
    u = d / ln

    canny = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0).astype(np.uint8), 25, 70)
    band = np.zeros((h, w), np.uint8)
    poly = np.array([p, r, r + n_in * band_w, p + n_in * band_w], np.float32)
    cv2.fillPoly(band, [poly.astype(np.int32)], 255)
    band_c = cv2.bitwise_and(canny, band)

    min_len = max(40.0, 0.30 * ln)
    segs = cv2.HoughLinesP(band_c, 1, np.pi / 360, threshold=int(min_len * 0.6),
                           minLineLength=int(min_len), maxLineGap=10)
    if segs is None:
        return None

    cos_lim = np.cos(np.radians(8.0))
    best, best_len = None, 0.0
    for s in segs.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, s)
        sv = np.array([x2 - x1, y2 - y1])
        sl = float(np.linalg.norm(sv))
        if sl < min_len:
            continue
        su = sv / sl
        if abs(float(np.dot(su, u))) < cos_lim:  # почти параллельна грани
            continue
        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)
        off = (mx - p[0]) * n_in[0] + (my - p[1]) * n_in[1]
        if not (2.0 <= off <= band_w):  # строго внутрь от текущей грани
            continue
        g1 = _bilinear_gray(gray, x1 - n_in[0] * 12, y1 - n_in[1] * 12)
        g2 = _bilinear_gray(gray, x2 - n_in[0] * 12, y2 - n_in[1] * 12)
        g1i = _bilinear_gray(gray, x1 + n_in[0] * 12, y1 + n_in[1] * 12)
        g2i = _bilinear_gray(gray, x2 + n_in[0] * 12, y2 + n_in[1] * 12)
        if (g1i + g2i) / 2 - (g1 + g2) / 2 < 12.0:  # внутрь — заметно светлее
            continue
        if sl > best_len:
            best, best_len = ((np.array([x1, y1]), su), sl)
    return best


def _bilinear_gray(gray: np.ndarray, x: float, y: float) -> float:
    h, w = gray.shape
    if not (0 <= x < w - 1 and 0 <= y < h - 1):
        return 0.0
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    v = (gray[y0, x0] * (1 - fx) * (1 - fy) + gray[y0, x0 + 1] * fx * (1 - fy)
         + gray[y0 + 1, x0] * (1 - fx) * fy + gray[y0 + 1, x0 + 1] * fx * fy)
    return float(v)


def _refine_quad(small: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Точная докрутка: каждый край заменяем прямой, проведённой по градиенту."""
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = _grad_mag(gray.astype(np.uint8))
    h, w = gray.shape
    diag = float(np.hypot(h, w))
    R = max(6.0, diag * 0.028)
    lim = R * 2.5

    q = order_points(quad).astype(np.float64)
    centroid = q.mean(axis=0)

    lines = []
    for i in range(4):
        p, r = q[i], q[(i + 1) % 4]
        d = r - p
        ln = float(np.linalg.norm(d))
        u = d / ln
        n = np.array([-u[1], u[0]])
        mid = (p + r) / 2.0
        if np.dot(n, centroid - mid) < 0:
            n = -n  # нормаль внутрь листа

        pts = []
        for t in np.linspace(0.07, 0.93, 25):
            base = p + t * d
            ss = np.arange(-R, R + 0.01, 0.75)
            xs = base[0] + ss * n[0]
            ys = base[1] + ss * n[1]
            vals = _profile(mag, xs, ys)
            grays = _profile(gray, xs, ys)
            peaks = [
                k for k in range(7, len(ss) - 8)
                if vals[k] >= vals[k - 1] and vals[k] >= vals[k + 1] and vals[k] > 35.0
            ]
            if not peaks:
                continue
            peaks.sort(key=lambda k: -vals[k])
            for k in peaks[:3]:
                inside = float(np.mean(grays[k + 2:k + 7]))
                outside = float(np.mean(grays[k - 6:k - 1]))
                if inside - outside > 3.0:  # внутрь — светлее (бумага)
                    pts.append(base + ss[k] * n)
                    break

        sup = _side_support(gray, mag.astype(np.uint8), p, r)
        if len(pts) >= 12:
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
            vl = cv2.fitLine(arr, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            line = (np.array([vl[2], vl[3]]), np.array([vl[0], vl[1]]))
        else:
            line = (mid.copy(), u.copy())
        lines.append(line)

    newq = []
    for i in range(4):
        x = _line_intersection(lines[i - 1], lines[i])
        newq.append(x if x is not None else q[i])
    newq = np.array(newq, dtype=np.float64)

    shift = np.linalg.norm(newq - q, axis=1)
    far = shift > lim
    if far.any():
        newq[far] = q[far]  # докрутка не должна уводить край далеко
    return newq.astype(np.float32)


def find_document_contour(image: np.ndarray):
    """
    Ищет лист. Устойчив к почерку, клетке, наклону, ручкам/пальцам поверх
    листа и пёстрому фону. Возвращает 4 точки в координатах исходного
    изображения или None, если листа нет.
    """
    h, w = image.shape[:2]
    img_area = float(h * w)
    diag_full = float(np.hypot(h, w))

    # --- этап 1: грубый поиск на уменьшенной копии ---
    small = _resize_to(image, _COARSE)
    sh, sw = small.shape[:2]
    s_area = float(sh * sw)
    gray1 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mag1 = _grad_mag(gray1)

    best, best_score = None, 0.0
    for q, qa_frac, fit, _tag in _candidates(small):
        sup = _support(gray1, mag1, q)
        score = qa_frac * fit * (0.25 + 0.75 * sup)
        if qa_frac > 0.93:
            score *= 0.35  # «весь кадр» — почти наверняка фон, а не лист
        if sup < 0.30:
            score *= 0.5
        if score > best_score:
            best, best_score = q, score

    if best is None:
        return None

    # --- этап 2: усадка рамки по цвету (отрезаем светлый фон/соседние листы) ---
    best = _shrink_quad(small, best)

    # --- этап 3: точная докрутка краёв на увеличенной копии ---
    small2 = _resize_to(image, _FINE)
    s2 = _FINE / max(h, w)
    full = best / (max(sh, sw) / max(h, w))  # координаты полного кадра
    q2 = (full * s2).astype(np.float32)

    refined = _refine_quad(small2, q2)
    full_refined = refined / s2

    if _valid_quad(full_refined, img_area, diag_full):
        return full_refined.astype(np.float32)
    if _valid_quad(full, img_area, diag_full):
        return full.astype(np.float32)
    return None


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
    ks = max(7, (max(small.shape[:2]) // 25) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, _ellipse(ks))  # стираем штрихи ручки
    bg = cv2.GaussianBlur(bg, (0, 0), ks / 2)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    bg = np.maximum(bg, 1)
    return cv2.divide(img, bg, scale=255)


def _border_trim(image: np.ndarray, max_frac: float = 0.10, run: int = 10) -> np.ndarray:
    """
    Срезает с краёв выпрямленного скана полосы явной «не-бумаги» (кусок стола,
    скатерти, спираль тетради), если рамка прошла чуть мимо края листа.
    Консервативно: полоса «не-бумага», только если она сильно темнее бумаги
    ИЛИ заметно другой теплоты (канал b* Lab); требуется серия из `run`
    подряд таких строк/столбцов, поэтому текст и тени не срезаются.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab_b = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 2].astype(np.float32)
    h, w = gray.shape
    # эталон бумаги — из центральной части скана (там точно лист)
    cy0, cy1 = int(h * 0.2), int(h * 0.8)
    cx0, cx1 = int(w * 0.2), int(w * 0.8)
    core = gray[cy0:cy1, cx0:cx1]
    core_b = lab_b[cy0:cy1, cx0:cx1]
    white = float(np.percentile(core[::4, ::4], 85))
    thr = white - 35.0
    bright = core_b[::4, ::4][core[::4, ::4] > white - 25]
    ref_b = float(np.median(bright)) if bright.size > 100 else 128.0
    lim_y, lim_x = int(h * max_frac), int(w * max_frac)

    def flags(lines_l, lines_b):
        # True = «не-бумага»
        return (lines_l.mean(axis=1) < thr) | (np.abs(lines_b.mean(axis=1) - ref_b) > 12.0)

    fy = flags(gray, lab_b)
    fx = flags(gray.T, lab_b.T)

    def run_start(f, limit):
        c = 0
        for i in range(limit):
            c = c + 1 if f[i] else 0
            if c >= run:
                return i - run + 2  # чуть отступить от края листа
        return 0

    def run_end(f, limit):
        c = 0
        for i in range(limit):
            c = c + 1 if f[len(f) - 1 - i] else 0
            if c >= run:
                return len(f) - (i - run + 2)
        return len(f)

    top = run_start(fy, lim_y)
    bottom = run_end(fy, lim_y)
    left = run_start(fx, lim_x)
    right = run_end(fx, lim_x)
    top, bottom = min(top, lim_y), max(bottom, h - lim_y)
    left, right = min(left, lim_x), max(right, w - lim_x)

    if top < 3 and (h - bottom) < 3 and left < 3 and (w - right) < 3:
        return image
    return image[max(top, 0):bottom, max(left, 0):right]


def to_black_white(image: np.ndarray, mode: str = "bw", trim: float = 0.002) -> np.ndarray:
    """
    mode:
      bw    — чистый чёрно-белый: клетка и шум убраны, текст остаётся
      gray  — серый: бумага белая, почерк читаемый (лучше всего для рукописи)
      color — цветной: бумага белая, ручка сохраняет цвет
    """
    # автообрезка «не-бумаги» по краям, затем узкая страховочная полоска
    image = _border_trim(image)
    h, w = image.shape[:2]
    dy, dx = int(h * trim), int(w * trim)
    if h - 2 * dy > 20 and w - 2 * dx > 20:
        image = image[dy:h - dy, dx:w - dx]

    image = _limit_size(image)
    flat = _flatten_light(image)
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)

    sample = gray[::4, ::4]
    white = float(max(180.0, np.percentile(sample, 85)))   # уровень бумаги
    ink = float(min(np.percentile(sample, 1), 130.0))      # уровень чернил

    if mode == "color":
        return cv2.convertScaleAbs(flat, alpha=255.0 / white)

    if mode == "gray":
        black = float(np.percentile(sample, 0.5))
        scale = 255.0 / max(white - black, 1.0)
        return cv2.convertScaleAbs(gray, alpha=scale, beta=-black * scale)

    # bw: порог между бумагой и чернилами; светлая клетка выше порога и пропадает
    soft = cv2.GaussianBlur(gray, (0, 0), 0.8)
    thr = white - 0.5 * (white - ink)
    ink_mask = (soft < thr).astype(np.uint8)

    # выкидываем мелкие «точки» (шум JPEG, фактура бумаги)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    min_area = max(12, int(ink_mask.shape[0] * ink_mask.shape[1] * 1.2e-5))
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    ink_mask = keep[labels]

    return np.where(ink_mask, 0, 255).astype(np.uint8)


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
            for x, y in pts.astype(int):
                cv2.circle(dbg, (x, y), max(6, image.shape[1] // 350), (0, 0, 255), -1)
            base, _ = os.path.splitext(path)
            cv2.imwrite(base + "_debug.png", dbg)

    result = to_black_white(warped, mode)
    cv2.imwrite(out, result)
    print(f"Готово: {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Сканер документов по фото (v2)")
    p.add_argument("input", help="путь к фото")
    p.add_argument("-o", "--output", help="куда сохранить результат")
    p.add_argument("--mode", choices=["bw", "gray", "color"], default="bw")
    p.add_argument("--debug", action="store_true", help="сохранить картинку с найденной рамкой")
    args = p.parse_args()

    out = args.output
    if not out:
        base, _ = os.path.splitext(args.input)
        out = base + "_scan.png"
    scan(args.input, out, args.mode, args.debug)


if __name__ == "__main__":
    main()
