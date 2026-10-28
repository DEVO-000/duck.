"""Самопроверка DocScan.

Генерирует синтетические «фото документов» (лист на тёмном столе, с наклоном
и лёгкой перспективой), прогоняет через конвейер и проверяет:
  * углы листа найдены;
  * после обработки остаточный наклон строк близок к нулю;
  * итоговый файл создан.

Запуск:  python selftest.py
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docscan.deskew import estimate_skew, rotate_bound  # noqa: E402
from docscan.pipeline import process_image               # noqa: E402

WORDS = ['INVOICE', 'ORDER', 'DATE', 'TOTAL', 'NAME', 'ADDRESS', 'ITEM', 'QTY',
         'PRICE', 'SUM', 'PAGE', 'SIGN', 'ACCOUNT', 'CODE', 'NOTE', 'REPORT',
         'NUMBER', 'TAX', 'SHIPPING', 'PAYMENT', 'CUSTOMER', 'COMPANY']


def make_page(w=1000, h=1400, seed=0):
    """Рисует «документ»: белый лист со строками текста."""
    rng = np.random.default_rng(seed)
    page = np.full((h, w), 248, dtype=np.uint8)
    y = 90
    while y < h - 80:
        size = float(rng.uniform(0.8, 1.2))
        x = 70
        for _ in range(int(rng.integers(4, 9))):
            word = str(rng.choice(WORDS))
            cv2.putText(page, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size,
                        35, 2, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, size, 2)
            x += tw + 24
            if x > w - 180:
                break
        y += int(52 * size) + 14
    return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


def make_photo(page, out_path, angle=7.0, canvas_scale=1.4, seed=0):
    """Помещает лист на шумный тёмный «стол» с перспективой и наклоном."""
    rng = np.random.default_rng(seed)
    ph, pw = page.shape[:2]

    m = 60
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    jitter = lambda: m + float(rng.uniform(-12, 12))  # noqa: E731
    dst = np.float32([[jitter(), jitter()],
                      [pw + jitter() - m, jitter()],
                      [pw + jitter() - m, ph + jitter() - m],
                      [jitter(), ph + jitter() - m]])
    warped = cv2.warpPerspective(page, cv2.getPerspectiveTransform(src, dst),
                                 (pw + 2 * m, ph + 2 * m),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(70, 70, 70))
    warped = rotate_bound(warped, angle)

    wh, ww = warped.shape[:2]
    cw, ch = max(int(pw * canvas_scale), ww), max(int(ph * canvas_scale), wh)
    bg = np.full((ch, cw, 3), 70, dtype=np.uint8)
    bg = np.clip(bg.astype(np.int16) + rng.integers(-12, 12, bg.shape), 0, 255).astype(np.uint8)
    x0, y0 = (cw - ww) // 2, (ch - wh) // 2
    region = bg[y0:y0 + wh, x0:x0 + ww]
    mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 100
    region[mask] = warped[mask]
    bg[y0:y0 + wh, x0:x0 + ww] = region

    photo = np.clip(bg.astype(np.int16) + rng.integers(-8, 8, bg.shape), 0, 255).astype(np.uint8)
    photo = cv2.GaussianBlur(photo, (3, 3), 0)
    cv2.imwrite(out_path, photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    in_dir = os.path.join(base, 'examples', 'in')
    out_dir = os.path.join(base, 'examples', 'out')
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    cases = [('sample_tilt7.jpg', 7.0, 1.4),
             ('sample_tilt-5.jpg', -5.0, 1.4),
             ('sample_straight.jpg', 0.0, 1.4),
             ('sample_fullframe.jpg', 2.0, 1.02)]

    failed = 0
    for i, (name, angle, cs) in enumerate(cases):
        path = os.path.join(in_dir, name)
        if not os.path.exists(path):
            make_photo(make_page(seed=i * 7), path, angle=angle,
                       canvas_scale=cs, seed=i + 1)
        r = process_image(path, out_dir, debug=False, mode='soft')
        out_img = cv2.imread(r['output'], cv2.IMREAD_GRAYSCALE)
        residual, method = estimate_skew(out_img)
        boundary_ok = r['steps'].get('границы', '').split(',')[0] != 'fullframe'
        ok = abs(residual) < 1.5 and boundary_ok
        failed += not ok
        print(f'{"OK  " if ok else "FAIL"} {name}:')
        print(f'     границы: {r["steps"].get("границы", "—")}')
        print(f'     исправлен наклон: {r["steps"].get("наклон", "—")}, '
              f'остаточный: {residual:+.2f}° ({method})')
        print(f'     результат: {os.path.basename(r["output"])} ({r["seconds"]} c)')

    print('\nИтог самопроверки:', 'УСПЕХ — все тесты пройдены'
          if failed == 0 else f'ПРОВАЛЕНО тестов: {failed}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
