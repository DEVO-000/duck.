"""
Сканер документов: находит лист на фото, выравнивает перспективу
и делает чёрно-белый «скан».

Установка:
    pip install opencv-python numpy

Запуск:
    python scanner.py photo.jpg                 -> photo_scan.png
    python scanner.py photo.jpg -o result.png
    python scanner.py photo.jpg --debug         (сохранит photo_debug.png с найденным контуром)
    python scanner.py photo.jpg --mode gray     (серый скан вместо ч/б)
"""

import argparse
import os
import sys

import cv2
import numpy as np


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
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (width, height))


def find_document_contour(image: np.ndarray):
    """Ищет контур листа. Возвращает 4 точки в координатах исходного изображения."""
    h, w = image.shape[:2]
    scale = 800.0 / max(h, w)
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    img_area = small.shape[0] * small.shape[1]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates = []

    # Вариант 1: границы (Canny)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    candidates.append(edges)

    # Вариант 2: светлый лист на более тёмном фоне (Otsu + закрытие дыр)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(
        otsu, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    )
    candidates.append(otsu)

    best = None
    best_area = 0
    for mask in candidates:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
        for c in contours:
            area = cv2.contourArea(c)
            if area < img_area * 0.15:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
                best, best_area = approx, area

    # Запасной вариант: минимальный повёрнутый прямоугольник вокруг самого большого контура
    if best is None:
        for mask in candidates:
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) >= img_area * 0.15:
                best = cv2.boxPoints(cv2.minAreaRect(c))
                break

    if best is None:
        return None
    return (best.reshape(4, 2) / scale).astype("float32")


def to_black_white(image: np.ndarray, mode: str = "bw") -> np.ndarray:
    """Превращает выровненное фото в скан."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Выравниваем неравномерное освещение: делим на сильно размытый фон
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 21)
    norm = cv2.divide(gray, bg, scale=255)

    if mode == "gray":
        return cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)

    block = max(15, (min(norm.shape) // 25) | 1)
    bw = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 15
    )
    # убираем мелкий шум
    bw = cv2.medianBlur(bw, 3)
    return bw


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
    p.add_argument("--mode", choices=["bw", "gray"], default="bw", help="bw — чёрно-белый, gray — серый")
    p.add_argument("--debug", action="store_true", help="сохранить картинку с найденным контуром")
    args = p.parse_args()

    out = args.output
    if not out:
        base, _ = os.path.splitext(args.input)
        out = base + "_scan.png"
    scan(args.input, out, args.mode, args.debug)


if __name__ == "__main__":
    main()
