"""
Сканер документов: находит лист (в том числе тетрадный, с почерком и клеткой)
на фото, выравнивает перспективу и делает «скан».

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


def _quads_from_mask(mask: np.ndarray, min_area: float):
    """Из бинарной маски достаёт четырёхугольники-кандидаты с оценкой."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(15))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            break
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        peri = cv2.arcLength(hull, True)

        quad = None
        for eps in (0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype("float32")
                break
            if len(approx) < 4:
                break
        if quad is None:
            quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype("float32")

        qa = cv2.contourArea(quad)
        if qa < min_area or hull_area <= 0:
            continue
        fit = min(qa, hull_area) / max(qa, hull_area)  # насколько выпуклая форма похожа на 4-угольник
        out.append((quad, qa, qa * fit * fit))
    return out


def find_document_contour(image: np.ndarray):
    """
    Ищет лист. Устойчиво к почерку и клетке: перед поиском границ «стираем»
    тёмные тонкие линии морфологическим закрытием, поэтому внутренности листа
    не мешают. Возвращает 4 точки в координатах исходного изображения.
    """
    h, w = image.shape[:2]
    scale = 800.0 / max(h, w)
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    img_area = float(sh * sw)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    k = max(9, (max(sh, sw) // 40) | 1)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, _ellipse(k))  # убирает почерк и клетку
    closed = cv2.GaussianBlur(closed, (7, 7), 0)

    masks = []

    # 1) светлая бумага на более тёмном фоне
    _, m = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(m)

    # 2) бумага почти бесцветная, стол/ткань — цветные
    sat = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1], (7, 7), 0)
    _, m = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(m)

    # 3) границы на «очищенной» картинке
    e = cv2.Canny(closed, 30, 90)
    e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=2)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, _ellipse(9))
    masks.append(e)

    best, best_score = None, 0.0
    for mask in masks:
        for quad, qa, score in _quads_from_mask(mask, img_area * 0.15):
            if qa > img_area * 0.95:
                score *= 0.85  # «весь кадр» — подозрительно, но возможно
            if score > best_score:
                best, best_score = quad, score

    if best is None:
        return None
    return (best / scale).astype("float32")


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


def to_black_white(image: np.ndarray, mode: str = "bw", trim: float = 0.006) -> np.ndarray:
    """
    mode:
      bw    — чистый чёрно-белый: клетка и шум убраны, текст остаётся
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
