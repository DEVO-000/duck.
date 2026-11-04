# -*- coding: utf-8 -*-
"""
Ядро бота обработки изображений документов.

Полный конвейер:
1) автоматический поиск границ листа и коррекция перспективы/деформации
   (4-точечное преобразование). Кандидаты-границы собираются несколькими
   методами: маска листа (геодезическая реконструкция фона), заливка фона
   от кромки кадра, яркостные/цветовые маски, рёбра Canny. Выбранный квад
   проверяется «по результату»: после выпрямления рамка должна быть светлой
   (иначе квад захвата стол или врезался в текст);
2) финальная доводка кадра «только лист»: обрезка по профилям яркости —
   отрезаются только ряды/столбцы ФОНА (тёмные/цветные), светлые ряды
   листа и текст не трогаются, контент потерять невозможно;
3) автоматическое выравнивание по горизонтали И вертикали: угол ищется
   проекционными профилями строк текста (знак надёжный, диапазон ±12°),
   Хаф — только как подстраховка на малых углах;
4) повышение чёткости: шумоподавление, выравнивание освещения (flat-field,
   тени и виньетка), однотонный белый фон страницы, резкость.

Запуск из командной строки:
    python bot_processor.py вход.jpg результат.jpg
"""
import io
import sys

import cv2
import numpy as np

try:
    from PIL import Image, ImageOps
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


# ----------------------------------------------------------------------------
# Загрузка с учётом EXIF-ориентации (телефонные фото)
# ----------------------------------------------------------------------------

def load_image_bgr(source):
    """Читает изображение (путь к файлу ИЛИ bytes) с учётом EXIF-ориентации."""
    if _HAS_PIL:
        im = Image.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray))
                        else source)
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        return np.asarray(im)[:, :, ::-1].copy()  # RGB -> BGR
    if isinstance(source, (bytes, bytearray)):
        return cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
    return cv2.imread(source, cv2.IMREAD_COLOR)


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


def _expand_quad(pts, margin=0.014):
    """Расширяет четырёхугольник наружу, чтобы не срезать край листа."""
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
# Оценка освещённости (тени/виньетка убираются делением на гладкий фон)
# ----------------------------------------------------------------------------

def _flat_field(gray, target=250.0, kfrac=0.12):
    """Делит яркость на крупномасштабную оценку фона: тени/виньетка исчезают.

    Оценка фона строится на уменьшенной копии (быстро и гладко): морфологическое
    замыкание убирает текст/мелкие детали, блюр сглаживает оценку.
    kfrac — доля меньшей стороны под ядро замыкания.
    """
    h, w = gray.shape
    scale = max(1, int(round(min(h, w) / 240.0)))
    sw, sh = max(16, int(round(w / scale))), max(16, int(round(h / scale)))
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    k = int(max(15, min(sw, sh) * kfrac))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg_s = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    bg_s = cv2.GaussianBlur(bg_s, (0, 0), max(3.0, k / 5.0))
    bg = cv2.resize(bg_s, (w, h), interpolation=cv2.INTER_LINEAR)
    bgf = np.maximum(bg.astype(np.float32), 25.0)
    return np.clip(gray.astype(np.float32) * (target / bgf), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# «Похоже на лист»: дистанция от цвета листа (яркость + цвет)
# ----------------------------------------------------------------------------

def _page_reference(lab):
    """Опорный цвет листа: самый светлый из нескольких центральных зон.

    Центр кадра почти всегда лист; тёмные рисунки/таблицы во весь лист
    мешают, поэтому при подозрительно тёмных зонах берём самый светлый
    квартил всего кадра (бумага — самый светлый крупный класс).
    """
    h, w = lab.shape[:2]
    zones = [
        (0.40, 0.40, 0.60, 0.60),   # центр
        (0.30, 0.15, 0.50, 0.45),   # слева-выше
        (0.50, 0.15, 0.70, 0.45),   # справа-выше
        (0.30, 0.55, 0.50, 0.85),   # слева-ниже
        (0.50, 0.55, 0.70, 0.85),   # справа-ниже
    ]
    best = None
    best_l = -1.0
    for (y0, x0, y1, x1) in zones:
        p = lab[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
        if p.size == 0:
            continue
        ml = float(np.median(p[:, :, 0]))
        if ml > best_l:
            best_l = ml
            best = (ml, float(np.median(p[:, :, 1])), float(np.median(p[:, :, 2])))
    if best is None or best_l < 170.0:
        # зоны тёмные (таблицы/рисунки во весь лист): берём светлый квартил
        lv = lab[:, :, 0].astype(np.float32)
        t75 = float(np.percentile(lv, 75))
        bright_sel = lab[:, :, 0] >= t75
        if bright_sel.any():
            best = (float(np.percentile(lv[bright_sel], 90)),
                    float(np.median(lab[:, :, 1][bright_sel])),
                    float(np.median(lab[:, :, 2][bright_sel])))
        else:
            best = (200.0, 128.0, 128.0)
    return best


def _page_distance(image, ref=None):
    """Карта «похожести на лист»: 255 = цвет листа, меньше = фон/стол.

    Лист отличается от стола не только яркостью, но и ЦВЕТОМ (дерево теплее,
    ткань синее). Тени предварительно убираются плоским полем по L, опорный
    цвет листа оценивается по тому ЖЕ выровненному каналу (согласованность!),
    затем считается дистанция в Lab. Такой образ различает даже светлый
    стол, почти совпадающий с бумагой по яркости.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l = _flat_field(lab[:, :, 0], target=250.0, kfrac=0.08)
    a = cv2.GaussianBlur(lab[:, :, 1], (7, 7), 0)
    b = cv2.GaussianBlur(lab[:, :, 2], (7, 7), 0)
    stack = cv2.merge((l, a, b))
    if ref is None:
        ref = _page_reference(stack)
    rl, ra, rb = ref
    lf = l.astype(np.float32)
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    dist = np.sqrt((lf - rl) ** 2
                   + ((af - ra) ** 2 + (bf - rb) ** 2) * 1.6)
    return np.clip(255.0 - dist * 2.0, 0, 255).astype(np.uint8), ref


# ----------------------------------------------------------------------------
# Маска листа: геодезическая реконструкция фона от кромки кадра
# ----------------------------------------------------------------------------

def _dist_stats(dist):
    """Статистики карты похожести: центр (лист), кромка кадра (фон), порог.

    Порог делит лист и стол ПОПОЛАМ между их типичными уровнями —
    адаптивно: светлый стол близок к бумаге, и фиксированный порог
    его не отделяет, а середина между кромкой и центром — отделяет.
    """
    h, w = dist.shape[:2]
    ch0, ch1 = int(h * 0.30), int(h * 0.70)
    cw0, cw1 = int(w * 0.30), int(w * 0.70)
    center = float(np.median(dist[ch0:ch1, cw0:cw1]))
    bh, bw = max(4, int(h * 0.08)), max(4, int(w * 0.08))
    strips = np.concatenate([dist[:bh].ravel(), dist[-bh:].ravel(),
                             dist[:, :bw].ravel(), dist[:, -bw:].ravel()])
    border = float(np.median(strips))
    t_split = float(np.clip(0.5 * (center + border), 200.0, 236.0))
    return center, border, t_split


def _geodesic_reconstruct(seed, allowed, iters=220):
    """Растит маску от seed только по разрешённым пикселям (allowed).

    Фон не может «просочиться» на лист: рост блокируется яркой полосой
    самой страницы. Заменяет ненадёжную заливку с допуском по цвету.
    """
    bg = seed.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for _ in range(iters):
        nxt = cv2.dilate(bg, kernel)
        np.bitwise_and(nxt, allowed, out=nxt)
        if np.array_equal(nxt, bg):
            break
        bg = nxt
    return bg


def page_mask(image):
    """Маска листа в кадре. Возвращает (маска uint8*255|None, доля площади).

    1) карта похожести на лист (цвет листа vs всё остальное); 2) фон =
    геодезическое расширение кромки кадра по НЕ-листу — на лист не
    просачивается; 3) лист = дополнение фона, дырки залиты, оставлена
    крупнейшая часть.
    """
    h, w = image.shape[:2]
    if h < 200 or w < 200:
        return None, 0.0

    # уменьшенная копия для скорости
    scale = max(1, int(round(max(h, w) / 640.0)))
    sw, sh = max(64, w // scale), max(64, h // scale)
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    dist, ref = _page_distance(small)

    # адаптивный порог: середина между кромкой (стол) и центром (лист)
    center, border, t_split = _dist_stats(dist)
    page_like = (dist > t_split).astype(np.uint8)
    # закрытие ТОЛЬКО мелких дырок (штрихи текста); ядро маленькое,
    # чтобы не «зашить» ажурные/текстурные фоны (кружево, ткань с узором)
    close_k = 9
    page_like = cv2.morphologyEx(
        page_like, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    non_page = (page_like == 0).astype(np.uint8)

    # старт: кромка кадра, не похожая на лист
    ring = np.zeros((sh, sw), np.uint8)
    rb = max(3, int(0.05 * min(sh, sw)))
    ring[:rb, :] = 1
    ring[-rb:, :] = 1
    ring[:, :rb] = 1
    ring[:, -rb:] = 1
    seed = (ring & non_page).astype(np.uint8)
    if seed.sum() == 0:
        # вся кромка похожа на лист — лист занимает весь кадр
        return None, 1.0

    bg = _geodesic_reconstruct(seed, non_page)
    # фон не должен занять весь кадр (иначе порог «листа» никуда не годится)
    bg_frac = float((bg > 0).mean())
    if bg_frac > 0.96 or bg_frac < 0.02:
        return None, 0.0

    page = (bg == 0).astype(np.uint8) * 255
    page = cv2.morphologyEx(page, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    page = cv2.morphologyEx(page, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    # дырки внутри листа (тёмные рисунки во всю ширину) заливаются:
    # это компоненты НЕ-листа, не касающиеся границы кадра
    inv = (page == 0).astype(np.uint8)
    if inv.any():
        n_cc, cc_lab = cv2.connectedComponents(inv, 8)
        border_ids = set(np.unique(np.concatenate([
            cc_lab[0, :], cc_lab[-1, :], cc_lab[:, 0], cc_lab[:, -1]])))
        hole_ids = [i for i in range(1, n_cc) if i not in border_ids]
        if hole_ids:
            holes = np.isin(cc_lab, hole_ids)
            page[holes] = 255

    n, lab, stats, _ = cv2.connectedComponentsWithStats(page, 8)
    if n <= 1:
        return None, 0.0
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 0.05 * sh * sw:
        return None, 0.0
    m_small = (lab == i).astype(np.uint8) * 255
    m = cv2.resize(m_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return m, float((m > 0).mean())


# ----------------------------------------------------------------------------
# Поиск границ листа
# ----------------------------------------------------------------------------

def _mask_to_quads(mask, img_shape, min_area_ratio=0.08, max_quads=3):
    """Из бинарной маски — кандидаты-четырёхугольники (страница может быть изогнута)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    img_area = float(img_shape[0] * img_shape[1])
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    out = []
    for c in cnts[:6]:
        area = cv2.contourArea(c)
        if area < min_area_ratio * img_area or area > 0.995 * img_area:
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        if peri < 32:
            continue
        quad = None
        for eps_rel in (0.02, 0.04, 0.07, 0.11):
            approx = cv2.approxPolyDP(hull, eps_rel * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype("float32")
                break
        if quad is None:
            # изогнутый/рваный край: вписанный прямоугольник по выпуклой оболочке
            quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype("float32")
        out.append(quad)
        if len(out) >= max_quads:
            break
    return out


def _edge_quads(gray_norm, img_shape, min_area_ratio=0.08):
    """Кандидаты по рёбрам Canny (нормализованная копия — границы контрастнее)."""
    blur = cv2.GaussianBlur(gray_norm, (5, 5), 0)
    edged = cv2.Canny(blur, 40, 120)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    img_area = float(img_shape[0] * img_shape[1])
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    out = []
    for c in cnts[:14]:
        area = cv2.contourArea(c)
        if area < min_area_ratio * img_area or area > 0.995 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        for eps_rel in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(c, eps_rel * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                out.append(approx.reshape(4, 2).astype("float32"))
                break
        if len(out) >= 5:
            break
    return out


def _quad_score(gray, quad, img_shape):
    """Оценивает, насколько четырёхугольник похож на ГРАНИЦУ ЛИСТА.

    Главный признак — перепад яркости через край: сразу внутри листа светло,
    сразу снаружи (стол/клеёнка) темнее. Внутренние рамки таблиц и блоки
    текста такого перепада не дают. Стороны, ушедшие за рамку кадра (лист
    вплотную к краю), не штрафуются — внешней полосы там просто нет.
    """
    h, w = img_shape[:2]
    pts = quad.astype(np.float32)
    area = cv2.contourArea(pts.astype(np.int32))
    img_area = float(h * w)
    cover = area / img_area
    if cover < 0.06 or cover > 1.20:
        return -1e9

    # прямоугольность: отклонение углов от 90°
    d = np.roll(pts, -1, axis=0) - pts
    ang_dev = 0.0
    for i in range(4):
        v1, v2 = d[i - 1], d[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return -1e9
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        ang_dev += abs(np.degrees(np.arccos(cosang)) - 90.0)
    ang_dev /= 4.0

    # полосы вдоль сторон: внутри и снаружи (по одной паре на сторону)
    strip = max(6.0, 0.015 * min(h, w))
    c = pts.mean(axis=0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    drops, grads, far_page = [], [], []
    t = np.linspace(0.08, 0.92, 48)  # вдоль стороны, не задевая углы
    for i in range(4):
        p1, p2 = pts[i], pts[(i + 1) % 4]
        mid = 0.5 * (p1 + p2)
        direction = p2 - p1
        nrm = np.array([-direction[1], direction[0]], dtype=np.float32)
        nn = np.linalg.norm(nrm)
        if nn < 1e-6:
            return -1e9
        nrm /= nn
        if np.dot(mid - c, nrm) < 0:  # нормаль наружу от центра
            nrm = -nrm
        base = p1 + (p2 - p1) * t[:, None]
        inner_px, outer_px, far_px = [], [], []
        for k in range(4):  # полоса из 4 линий с каждой стороны
            off_in = -strip * (1.6 - 0.35 * k)
            off_out = strip * (0.4 + 0.35 * k)
            off_far = strip * (3.2 + 0.6 * k)
            for off, store in ((off_in, inner_px), (off_out, outer_px),
                               (off_far, far_px)):
                xy = base + nrm * off
                xi = np.round(xy[:, 0]).astype(int)
                yi = np.round(xy[:, 1]).astype(int)
                ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                if ok.sum() < 8:
                    continue
                store.append(gray[yi[ok], xi[ok]])
        if inner_px and outer_px:
            drops.append(float(np.concatenate(inner_px).mean()
                               - np.concatenate(outer_px).mean()))
        # далеко снаружи снова «лист»? значит квад врезан в страницу:
        # за границей настоящего листа лежит стол, а не бумага
        if far_px:
            far_page.append(float(np.concatenate(far_px).mean()))
        # градиент вдоль самой стороны
        xi = np.round(base[:, 0]).astype(int)
        yi = np.round(base[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if ok.sum() > 15:
            grads.append(float(mag[yi[ok], xi[ok]].mean()))

    drop = float(np.mean(drops)) if drops else 0.0
    grad = float(np.mean(grads)) if grads else 0.0
    # штраф за «страницоподобную» дальнюю полосу снаружи
    far_pen = 0.0
    if far_page:
        far_frac = float(np.mean([f > 218.0 for f in far_page]))
        far_pen = 2.0 * far_frac

    cover_score = 1.0 if 0.25 <= cover <= 0.96 else max(0.0, 1.0 - abs(cover - 0.6))
    rect_score = max(0.0, 1.0 - ang_dev / 40.0)
    score = (3.0 * np.clip(drop / 22.0, -1.0, 1.0)
             + 1.2 * cover_score
             + 1.0 * rect_score
             + 0.8 * np.clip(grad / 25.0, 0.0, 1.0)
             - far_pen)
    return float(score)


def _warp_verify(image, quad):
    """Проверка кандидата «по результату»: выпрямляем и смотрим рамку.

    Правильный квад листа после выпрямления даёт ПОХОЖУЮ НА ЛИСТ полосу по
    внутренним кромкам (поля страницы). Чужая полоса = в квад попал стол.
    Стороны, прижатые к границе кадра, не проверяются. Допуск мягкий:
    тетради со спиралями и линейками имеют тёмные поля у края.
    """
    h, w = image.shape[:2]
    rect = order_points(quad.astype("float32"))
    scale = 420.0 / max(h, w)
    small = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1 else image
    q = rect * (scale if scale < 1 else 1.0)
    prev = four_point_transform(small, q)
    if prev.shape[0] < 60 or prev.shape[1] < 60:
        return False
    g, _ = _page_distance(prev)
    ph, pw = g.shape
    ph_in, pw_in = max(3, int(ph * 0.03)), max(3, int(pw * 0.03))
    bands = {
        'top': g[ph_in:ph_in * 2, pw // 4: 3 * pw // 4],
        'bottom': g[-2 * ph_in:-ph_in, pw // 4: 3 * pw // 4],
        'left': g[ph // 4: 3 * ph // 4, pw_in:2 * pw_in],
        'right': g[ph // 4: 3 * ph // 4, -2 * pw_in:-pw_in],
    }
    # стороны квада, прижатые к кромке кадра, пропускаем
    near = 0.04
    sides_open = []
    order = [(0, 'top'), (1, 'right'), (2, 'bottom'), (3, 'left')]
    for idx, name in order:
        p1, p2 = rect[idx], rect[(idx + 1) % 4]
        if name == 'top':
            d = min(p1[1], p2[1]) / h
        elif name == 'bottom':
            d = 1.0 - max(p1[1], p2[1]) / h
        elif name == 'left':
            d = min(p1[0], p2[0]) / w
        else:
            d = 1.0 - max(p1[0], p2[0]) / w
        sides_open.append((name, d < near))
    for name, open_side in sides_open:
        if open_side:
            continue
        band = bands[name]
        if band.size == 0:
            continue
        if float(band.mean()) < 150.0 or float((band < 140).mean()) > 0.30:
            return False
    return True


def _bg_complement_quads(image, img_shape):
    """Кандидаты через заливку ФОНА от кромки кадра (цветовая связность).

    Дополнительный метод: полезен, когда лист и стол различаются в первую
    очередь цветом, а не яркостью.
    """
    h, w = img_shape[:2]
    scale = max(1, int(round(max(h, w) / 700.0)))
    sw, sh = max(64, w // scale), max(64, h // scale)
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    bg = np.zeros((sh, sw), np.uint8)
    total = float(sh * sw)
    lo = (26, 11, 11)
    up = (26, 11, 11)
    step = max(2, int(min(sh, sw) / 60))
    border_pts = []
    for x in range(0, sw, step):
        border_pts += [(x, 0), (x, sh - 1)]
    for y in range(0, sh, step):
        border_pts += [(0, y), (sw - 1, y)]
    for (x, y) in border_pts:
        if bg[y, x]:
            continue
        mask = np.zeros((sh + 2, sw + 2), np.uint8)
        flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
        cv2.floodFill(small.copy(), mask, (x, y), (255, 255, 255),
                      loDiff=lo, upDiff=up, flags=flags)
        area = float((mask == 255).sum())
        if area <= 0 or area > 0.34 * total:
            continue  # заливка ушла на лист — не фон
        bg |= mask[1:-1, 1:-1]
    bg_frac = float((bg > 0).mean())
    if bg_frac < 0.04 or bg_frac > 0.92:
        return []  # фон не найден или «фон» — весь кадр
    bg = cv2.dilate(bg, np.ones((5, 5), np.uint8), iterations=2)
    page = cv2.bitwise_not(bg)
    page = cv2.morphologyEx(page, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return _mask_to_quads(page, img_shape, 0.08)


def find_document_contour(image, min_area_ratio=0.08):
    """Находит четырёхугольник листа. Возвращает (quad, метод) или (None, причина).

    Главный путь — квад из МАСКИ листа (геодезическая реконструкция по
    цвету): она находит границу листа даже на светлом столе. Масковый квад
    проверяется выпрямлением и принимается сразу. Прочие кандидаты
    (заливка фона, яркостные маски, рёбра) — запасные, с высокой планкой
    оценки: ошибка перспективы хуже её отсутствия.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (7, 7), 0), cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]

    m, frac = page_mask(image)
    if m is not None and frac > 0.985:
        return None, "лист занимает весь кадр"

    # --- главный путь: квад из маски листа ---
    if m is not None and frac >= 0.20:
        mask_quads = _mask_to_quads(m, image.shape, min_area_ratio, max_quads=2)
        if mask_quads:
            q = max(mask_quads, key=lambda qq: cv2.contourArea(qq.astype(np.int32)))
            if _warp_verify(image, q):
                return q, "маска листа (доля %.2f)" % frac

    candidates = []
    methods = []
    if m is not None:
        qs = _mask_to_quads(m, image.shape, min_area_ratio)
        candidates += qs
        methods += ["маска"] * len(qs)

    candidates += _bg_complement_quads(image, image.shape)
    methods += ["заливка фона"] * 10

    gray_norm = _flat_field(gray)

    # маски яркости/насыщенности (светлый лист на тёмном или цветном фоне)
    otsu_t, bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v_lo = max(60, int(otsu_t * 0.55))
    bw_shadowfree = ((sat < 70) & (val > v_lo)).astype(np.uint8) * 255
    bw_dark = cv2.bitwise_not(bw_otsu)
    for mask in (bw_otsu, bw_shadowfree, bw_dark):
        cover = float((mask > 0).mean())
        if cover < 0.10 or cover > 0.88:
            continue  # маска ни о чём: почти всё или почти ничего
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        qs = _mask_to_quads(mask, image.shape, min_area_ratio)
        candidates += qs
        methods += ["яркость"] * len(qs)

    # рёбра Canny
    qs = _edge_quads(gray_norm, image.shape, min_area_ratio)
    candidates += qs
    methods += ["рёбра"] * len(qs)

    if not candidates:
        return None, "контур не найден"

    # карта похожести на лист: внутри квада — лист, снаружи — стол
    dist_img, _ = _page_distance(image)
    scored = []
    for i, quad in enumerate(candidates):
        sc = _quad_score(dist_img, quad, image.shape)
        if sc > -1e8:
            scored.append((sc, float(cv2.contourArea(quad.astype(np.int32))), i))
    if not scored:
        return None, "надёжный контур не найден"
    best_sc = max(s[0] for s in scored)
    if best_sc < 4.2:
        # слабые кандидаты не принимаем: ошибка перспективы хуже её отсутствия
        return None, "надёжный контур не найден (оценка %.2f)" % best_sc

    # среди почти таких же по оценке — самый большой квад, затем проверка
    top = sorted([s for s in scored if s[0] >= best_sc - 0.55], key=lambda s: -s[1])
    for sc, area, i in top[:4]:
        if m is not None and area < (frac - 0.12) * h * w:
            continue  # квад меньше известной маски листа — врезка
        if _warp_verify(image, candidates[i]):
            return candidates[i], "кандидат %d (метод: %s, оценка %.2f)" % (
                i + 1, methods[i] if i < len(methods) else "?", sc)
    return None, "кандидаты не прошли проверку выпрямлением"


def _edge_content_frac(image, inset=0.004, band=0.018, dark=130):
    """Доля тёмных пикселей (текста) в кромочных полосах каждой стороны."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    bi, bb = max(2, int(h * band)), max(2, int(w * band))
    yi0, yi1 = int(h * inset), int(h * inset) + bb
    xi0, xi1 = int(w * inset), int(w * inset) + bb
    bands = {
        'top': gray[yi0:yi1, int(w * 0.2):int(w * 0.8)],
        'bottom': gray[h - yi1:h - yi0 or h, int(w * 0.2):int(w * 0.8)],
        'left': gray[int(h * 0.2):int(h * 0.8), xi0:xi1],
        'right': gray[int(h * 0.2):int(h * 0.8), w - xi1:w - xi0 or w],
    }
    out = {}
    for k, v in bands.items():
        out[k] = float((v < dark).mean()) if v.size else 0.0
    return out


def crop_by_mask(image, min_frac=0.20, max_frac=0.975, axis_only=False):
    """Кроп «только лист» по маске: повернутый прямоугольник маски -> рамка.

    Гарантирует, что в кадре остаётся только лист, даже если перспективу
    найти не удалось: маска листа строится по цвету (геодезически),
    вписанный прямоугольник выпрямляет и обрезает одним махом. Если у
    кромки результата обнаружен текст (маска села близко к контенту),
    рамка расширяется — контент важнее чистоты кромки.
    axis_only=True — резать БЕЗ поворота (осевой габарит): для контрольного
    прохода ПОСЛЕ выравнивания, когда любой «угол» прямоугольника — шум
    маски, а не реальный наклон.
    Возвращает (изображение, применено ли, доля маски).
    """
    h, w = image.shape[:2]
    m, frac = page_mask(image)
    if m is None or not (min_frac <= frac <= max_frac):
        return image, False, frac
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image, False, frac
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    quad = None
    peri = cv2.arcLength(hull, True)
    for eps_rel in (0.02, 0.04, 0.07):
        approx = cv2.approxPolyDP(hull, eps_rel * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype("float32")
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype("float32")

    if axis_only:
        # осевой габарит маски — без какого-либо поворота
        x0, y0, ww, hh = cv2.boundingRect(hull)
        out = None
        for margin in (0.006, 0.016):
            mx = max(4, int(margin * w))
            my = max(4, int(margin * h))
            xa = max(0, x0 - mx)
            ya = max(0, y0 - my)
            xb = min(w, x0 + ww + mx)
            yb = min(h, y0 + hh + my)
            cand = image[ya:yb, xa:xb].copy()
            if cand.shape[0] < 0.3 * h or cand.shape[1] < 0.3 * w:
                break
            out = cand
            edges = _edge_content_frac(cand)
            if max(edges.values()) <= 0.025:
                break
        if out is not None:
            return out, True, frac
        return image, False, frac

    out = None
    for margin in (0.006, 0.016):
        q = _expand_quad(quad, margin)
        cand = four_point_transform(image, q)
        # защита от вырождения и разворотов на 90°
        if cand.shape[0] < 0.3 * h or cand.shape[1] < 0.3 * w:
            break
        if cand.shape[0] * cand.shape[1] > 1.6 * h * w:
            break
        out = cand
        # у кромки явно есть текст? — чуть расширяем рамку (контент важнее)
        edges = _edge_content_frac(cand)
        if max(edges.values()) <= 0.025:
            break
    if out is None:
        return image, False, frac
    return out, True, frac


# ----------------------------------------------------------------------------
# Автоопределение и исправление наклона
# ----------------------------------------------------------------------------

def _hough_skew(gray, max_angle=12.0):
    """Угол по линиям (Хаф). Быстро, но знак иногда ошибается — только подстраховка."""
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0,
        threshold=max(60, w // 15),
        minLineLength=max(40, w // 10),
        maxLineGap=20,
    )
    if lines is None or len(lines) < 4:
        return None
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
        return None
    return float(np.median(angles))


def _rotate_mask(sub, angle):
    """Поворот маски на малый угол вокруг центра (для скана углов)."""
    h, w = sub.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(sub, m, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _projection_skew(gray, max_angle=12.0):
    """Угол по проекционным профилям строк текста: знак надёжный.

    Текст бинаризуем адаптивным порогом (не боится теней), поворачиваем маску
    на пробные углы и ищем максимум дисперсии суммы по строкам: строки текста
    дают резкие пики, когда текст горизонтален.
    """
    h, w = gray.shape
    scale = max(1, max(h, w) // 900)
    small = gray[::scale, ::scale]
    mask = cv2.adaptiveThreshold(
        small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    frac = float((mask > 0).mean())
    if frac < 0.004 or frac > 0.45:
        return None
    ys, xs = np.where(mask > 0)
    y0, y1 = max(0, ys.min() - 12), min(small.shape[0], ys.max() + 13)
    x0, x1 = max(0, xs.min() - 12), min(small.shape[1], xs.max() + 13)
    sub = mask[y0:y1, x0:x1]
    if sub.shape[0] < 60 or sub.shape[1] < 60:
        return None

    def score_angle(a):
        m = _rotate_mask(sub, a).astype(np.float32)
        rows = m.sum(axis=1)
        return float(((rows - rows.mean()) ** 2).sum())

    # грубый скан, затем уточнение
    angles = np.arange(-max_angle, max_angle + 1e-6, 0.5)
    scores = [score_angle(a) for a in angles]
    i = int(np.argmax(scores))
    a0 = angles[i]
    fine = np.arange(a0 - 0.45, a0 + 0.45 + 1e-6, 0.1)
    fs = [score_angle(a) for a in fine]
    best = fine[int(np.argmax(fs))]

    # если даже лучший профиль слабый (текста мало) — не доверяем:
    # остаточный наклон в 1–2° лучше, чем поворот на неверный угол
    m = _rotate_mask(sub, best).astype(np.float32)
    rows = m.sum(axis=1)
    sharp = float(rows.std() / max(rows.mean(), 1.0))
    if sharp < 0.55:
        return None
    # большие углы — только при очень уверенном пике (защита от ложных
    # срабатываний на мелких кропах и рисунках)
    if abs(best) > 8.0 and sharp < 0.80:
        return None
    return float(best)


def detect_skew_angle(image, max_angle=12.0):
    """Итоговый угол наклона: проекции строк (надёжный знак) + Хаф (подстраховка).

    Хафу доверяем только на малых углах (до 3°): у Хафа на перспективно
    искажённом тексте знак иногда ошибается, а на малом угле ошибка мала.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    a_proj = _projection_skew(gray, max_angle)
    a_hough = _hough_skew(gray, max_angle)
    if a_proj is None and a_hough is None:
        return 0.0
    if a_proj is None:
        # без проекций поворачиваем только чуть-чуть и только по Хафу
        if abs(a_hough) <= 3.0:
            return a_hough
        return 0.0
    if a_hough is None:
        return a_proj
    if abs(a_hough - a_proj) <= 1.0:
        return 0.5 * (a_hough + a_proj)
    return a_proj  # разошлись — верим проекциям: знак проверен профилями


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
# Обрезка «только лист»: по профилям яркости
# ----------------------------------------------------------------------------

def _side_cut(frac_line, page_level, K):
    """Ищет индекс, начиная с которого линия «страничная».

    frac_line[i] — доля «страничных» пикселей в строке/столбце i.
    Возвращаем первый индекс, откуда начинается устойчиво страничный
    участок (окно длины K подряд >= page_level).
    """
    n = len(frac_line)
    if n < 2 * K:
        return 0
    f = np.asarray(frac_line, dtype=np.float32)
    # интегральное окно
    csum = np.concatenate([[0.0], np.cumsum(f)])
    win = (csum[K:] - csum[:-K]) / float(K)  # win[i] = среднее по [i, i+K)
    good = win >= page_level
    if not good.any():
        return 0
    i = int(np.argmax(good))
    return min(i, n - K)


def profile_crop(image, max_side_cut=0.48):
    """Обрезает ФОН вокруг листа: в кадре остаётся только лист.

    Работает по профилям карты «похоже на лист» (цвет листа vs стол):
    ряд/столбец считается «листом», если достаточная доля его пикселей
    похожа на лист. Чужие ряды (стол, ткань) отрезаются. Контент потерять
    нельзя: строка с текстом всё равно в основном цвета листа.
    Возвращает (изображение, словарь с размеров срезов).
    """
    h, w = image.shape[:2]
    info_cuts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    if h < 240 or w < 240:
        return image, info_cuts

    dist, ref = _page_distance(image)

    # адаптивный порог «лист/стол» и уровень активации обрезки
    center, border, t_split = _dist_stats(dist)
    if center - border < 8.0:
        # контраста «лист-фон» нет: лист занимает весь кадр
        return image, info_cuts

    pageish = (dist > t_split - 6.0).astype(np.float32)
    level = 0.58

    Kh = max(6, int(round(h * 0.025)))
    Kw = max(6, int(round(w * 0.025)))
    rows = pageish.mean(axis=1)
    cols = pageish.mean(axis=0)

    lim_h = int(h * max_side_cut)
    lim_w = int(w * max_side_cut)

    top = _side_cut(rows, level, Kh)
    bottom = h - _side_cut(rows[::-1], level, Kh)
    left = _side_cut(cols, level, Kw)
    right = w - _side_cut(cols[::-1], level, Kw)

    top = min(top, lim_h)
    bottom = max(bottom, h - lim_h)
    left = min(left, lim_w)
    right = max(right, w - lim_w)

    # совсем мелкую обрезку не делаем (шум профилей)
    if top < h * 0.01:
        top = 0
    if h - bottom < h * 0.01:
        bottom = h
    if left < w * 0.01:
        left = 0
    if w - right < w * 0.01:
        right = w

    if top == 0 and bottom == h and left == 0 and right == w:
        return image, info_cuts
    if bottom - top < h * 0.30 or right - left < w * 0.30:
        # подозрительно много — лучше не трогать (защита от ошибки)
        return image, info_cuts

    info_cuts = {"top": int(top), "bottom": int(h - bottom),
                 "left": int(left), "right": int(w - right)}
    return image[top:bottom, left:right].copy(), info_cuts


# ----------------------------------------------------------------------------
# Повышение чёткости и качества
# ----------------------------------------------------------------------------

def denoise(image, h=4.0):
    """Лёгкое шумоподавление (0 = выключено). Возвращает (изображение, применено ли)."""
    hh, ww = image.shape[:2]
    if h > 0 and hh * ww <= 4_500_000:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21), True
    return image, False


def balance_light(image, clahe_clip=1.35):
    """Тени/виньетка убираются, фон страницы приводится к однотонному белому.

    1) flat-field по L-каналу (ядро мельче — ловит и средние тени): делим
       яркость на гладкую оценку фона — пропадают тени, разводы, виньетка;
    2) мягкий CLAHE — контраст текста;
    3) двухступенчатое мягкое «отбеливание»: всё, что светлее ~196, плавно
       тянется к белому, светлее ~234 — становится чистым белым. Фон
       страницы получается однотонным, текст и карандаш не затрагиваются.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _flat_field(l, target=252.0, kfrac=0.08)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lf = l.astype(np.float32)
    t1 = np.clip((lf - 196.0) / 50.0, 0.0, 1.0)   # 196 -> 0, 246 -> 1
    lf = lf * (1.0 - t1) + 255.0 * t1
    t2 = np.clip((lf - 232.0) / 12.0, 0.0, 1.0)   # финальный чистый белый
    l = np.round(lf * (1.0 - t2) + 255.0 * t2).astype(np.uint8)

    near_white = l >= 240
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a[near_white] = 128.0 + (a[near_white] - 128.0) * 0.25
    b[near_white] = 128.0 + (b[near_white] - 128.0) * 0.25
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def whiten_background(image, tolerance=50, grad_tolerance=8):
    """Заливает белым фоновые области, соединённые с краями кадра.

    Две заливки: с фиксированным диапазоном (грубо, широким допуском) и
    «плавающим» диапазоном с малым допуском — она спускается по плавным
    градиентам тени у края листа и добивает их до чистого белого. Текст
    не затрагивается: у штрихов резкие перепады, заливка останавливается.
    """
    h, w = image.shape[:2]
    total = float(h * w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_frac = float((gray > 180).mean())
    max_area = 0.96 if bright_frac > 0.5 else 0.30
    out = image.copy()
    seeds = []
    for k in range(5):
        seeds.append((int(w * k / 4.0), 0))
        seeds.append((int(w * k / 4.0), h - 1))
    for k in range(5):
        seeds.append((0, int(h * k / 4.0)))
        seeds.append((w - 1, int(h * k / 4.0)))
    for seed in seeds:
        sx = min(max(seed[0], 0), w - 1)
        sy = min(max(seed[1], 0), h - 1)
        for fixed, tol in ((True, tolerance), (False, grad_tolerance)):
            mask = np.zeros((h + 2, w + 2), np.uint8)
            flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
            if fixed:
                flags |= cv2.FLOODFILL_FIXED_RANGE
            cv2.floodFill(out, mask, (sx, sy), (255, 255, 255),
                          loDiff=(tol,) * 3, upDiff=(tol,) * 3,
                          flags=flags)
            area = float((mask == 255).sum())
            if area == 0 or area > max_area * total:
                continue
            sel = mask[1:-1, 1:-1] == 255
            out[sel] = (255, 255, 255)
    return out


def trim_white_margins(image, thresh=242, pad=10, max_cut=0.08):
    """Обрезает только ПОЛНОСТЬЮ пустые белые кромки (клинья после поворота).

    Строка/столбец отрезается, если в нём нет ни одного пикселя темнее thresh —
    контент (текст, линии, тени) физически не может быть потерян.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    content = gray < thresh
    if not content.any():
        return image
    h, w = gray.shape
    rows = content.any(axis=1)
    cols = content.any(axis=0)
    if rows.all() and cols.all():
        return image
    top = int(np.argmax(rows)) if rows.any() else 0
    bottom = h - int(np.argmax(rows[::-1])) if rows.any() else h
    left = int(np.argmax(cols)) if cols.any() else 0
    right = w - int(np.argmax(cols[::-1])) if cols.any() else w
    # режем не более max_cut с каждой стороны и только с запасом pad:
    # клинья после поворота маленькие, а поля страницы не трогаем
    top_cut = min(top, int(h * max_cut))
    bot_cut = min(h - bottom, int(h * max_cut))
    left_cut = min(left, int(w * max_cut))
    right_cut = min(w - right, int(w * max_cut))
    top = max(0, top_cut - pad)
    bottom = h - max(0, bot_cut - pad)
    left = max(0, left_cut - pad)
    right = w - max(0, right_cut - pad)
    if bottom - top < 200 or right - left < 200:
        return image
    if (bottom - top) * (right - left) < 0.5 * h * w:
        return image
    return image[top:bottom, left:right]


def sharpen(image, amount=0.6, sigma=2.6):
    """Повышение резкости (маска нерезкости)."""
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)


def enhance_quality(image, denoise_h=4.0, clahe_clip=1.6, sharpen_amt=0.6):
    """Полное улучшение качества одним вызовом (для совместимости)."""
    notes = []
    image, den_ok = denoise(image, denoise_h)
    notes.append("шум убран" if den_ok else "шумоподавление пропущено (большой размер)")
    image = balance_light(image, clahe_clip)
    image = sharpen(image, sharpen_amt)
    notes.append("освещённость выровнена, фон однотонный, резкость повышена")
    return image, notes


# ----------------------------------------------------------------------------
# Главный конвейер
# ----------------------------------------------------------------------------

def process_image(image, do_perspective=True, do_skew=True, do_enhance=True,
                  rotate_deg=0):
    """Обрабатывает изображение (BGR, numpy). Возвращает (результат, словарь отчёта).

    rotate_deg — предварительный поворот всего кадра на 90/180/270°
    (для снимков «на боку» или «вверх ногами»): делается ДО всех
    остальных этапов, чтобы выравнивание и поиск границ работали
    с правильно ориентированным текстом.
    """
    if image is None:
        raise ValueError("Пустое изображение")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if rotate_deg in (90, 180, 270):
        code = {90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_CLOCKWISE}[int(rotate_deg)]
        image = cv2.rotate(image, code)
    result = image.copy()
    info = {"perspective": False, "skew_angle": 0.0, "enhanced": False, "steps": []}

    # 1. Перспектива и деформация: в кадре остаётся только лист
    if do_perspective:
        quad, why = find_document_contour(result)
        if quad is not None:
            quad = _expand_quad(quad, 0.014)
            result = four_point_transform(result, quad)
            info["perspective"] = True
            info["steps"].append("границы листа найдены, перспектива и деформация исправлены")
        else:
            info["steps"].append("перспектива: %s" % why)

    # 2. Обрезка «только лист»: по маске (повёрнутый прямоугольник),
    #    при неудаче — по профилям яркости. Светлые ряды листа не режутся,
    #    поэтому контент не теряется. Делается ДО шумоподавления (денойз
    #    сглаживает фактуру фона до «похожести на лист») и ДО выравнивания
    #    наклона (кроп мог бы внести свой мелкий угол — строки текста
    #    надёжнее доведут ориентацию последними).
    before = result.shape[:2]
    result, cropped, mfrac = crop_by_mask(result)
    if cropped:
        info["steps"].append("кадр обрезан по границам листа (маска, доля %.2f)" % mfrac)
    else:
        result, cuts = profile_crop(result)
        if cuts["top"] or cuts["bottom"] or cuts["left"] or cuts["right"]:
            info["steps"].append(
                "лишний фон срезан (низ %d, верх %d, лево %d, право %d пикс.)"
                % (cuts["bottom"], cuts["top"], cuts["left"], cuts["right"])
            )

    # 3. Наклон: выравнивание по горизонтали и вертикали (после кропа —
    #    исправляет и собственный мелкий угол вписанного прямоугольника)
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
            # поворот размазывает кромочные полосы вдоль краёв — контрольный
            # проход осевого кропа (без поворота!) снимает клинья и размазку
            result, cropped2, _frac2 = crop_by_mask(result, axis_only=True)
            if not cropped2:
                result, _cuts2 = profile_crop(result)

    # 4. Шумоподавление (после обрезки — меньше площадь, быстрее и чище)
    if do_enhance:
        result, den_ok = denoise(result, 4.0)

    # 5. Чёткость и качество: выравнивание света + однотонный фон
    if do_enhance:
        result = balance_light(result)
        info["enhanced"] = True

    # 6. Заливка фона белым от краёв: остатки стола, виньетка, клинья
    if do_perspective or rotated or do_enhance:
        result = whiten_background(result)
        h, w = result.shape[:2]
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        border = np.zeros((h, w), bool)
        border[:8, :] = True
        border[-8:, :] = True
        border[:, :8] = True
        border[:, -8:] = True
        result[border & (gray > 185)] = 255

    # 7. Обрезка полностью пустых белых кромок (клинья поворота)
    result = trim_white_margins(result)

    # 8. Резкость — последней, чтобы не усиливать артефакты
    if do_enhance:
        result = sharpen(result, 0.5)
        info["steps"].append("чёткость и качество: шум убран, свет выровнен, фон однотонно белый, резкость повышена")

    info["size"] = {"width": int(result.shape[1]), "height": int(result.shape[0])}
    return result, info


def process_file(in_path, out_path, **kwargs):
    """Обрабатывает файл и сохраняет результат. Возвращает словарь отчёта."""
    image = load_image_bgr(in_path)
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
