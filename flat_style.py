import cv2
import numpy as np
from poster_pipeline import get_masks, quantize_palette, apply_palette, make_paper_texture
from landmark_face import draw_landmark_face, draw_landmark_face_profile, detect_faces


def merge_small_fragments(quant, keep_mask, min_area=250):
    """同じ色の小さすぎる断片を、隣接する一番優勢な色に統合する
    （ジャケットの一部だけ別の色になる『まだら』を減らすため）"""
    result = quant.copy()
    unique_colors = np.unique(quant.reshape(-1, 3), axis=0)
    for color in unique_colors:
        region = (np.all(quant == color, axis=-1) & (keep_mask == 1)).astype(np.uint8)
        num_labels, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
        for i in range(1, num_labels):
            if stats_cc[i, cv2.CC_STAT_AREA] < min_area:
                comp_mask = (labels_cc == i).astype(np.uint8)
                dilated = cv2.dilate(comp_mask, np.ones((9, 9), np.uint8))
                neighbor_ring = (dilated == 1) & (comp_mask == 0) & (keep_mask == 1)
                if neighbor_ring.sum() == 0:
                    continue
                neighbor_colors = result[neighbor_ring]
                colors_u, counts = np.unique(neighbor_colors.reshape(-1, 3), axis=0, return_counts=True)
                dominant = colors_u[np.argmax(counts)]
                result[comp_mask == 1] = dominant
    return result


def boost_saturation(bgr_img, factor=1.35):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def draw_flat_outlines(canvas, keep_mask, quant, ink_color, thickness=2, epsilon=2.5, min_len=20):
    """写真テクスチャ(Canny)を追わず、色領域そのものの境界だけを
    太く・大きく単純化した1本の線として描く（イラストレーターの線に近づける）"""
    unique_colors = np.unique(quant.reshape(-1, 3), axis=0)
    for color in unique_colors:
        region = np.all(quant == color, axis=-1).astype(np.uint8) & keep_mask
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 80:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.polylines(canvas, [approx], True, ink_color, thickness, cv2.LINE_AA)
    return canvas


def flatten_busy_patterns(img_s, keep_mask):
    """柄物の服など、局所的にテクスチャが激しい部分だけ強めにぼかして
    そもそもの量子化前に柄自体を弱めておく"""
    gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_mean = cv2.boxFilter(gray, -1, (9, 9))
    local_sq_mean = cv2.boxFilter(gray ** 2, -1, (9, 9))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    busy = (local_std > 18).astype(np.uint8) & keep_mask.astype(np.uint8)
    busy = cv2.morphologyEx(busy, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    heavy_blur = cv2.GaussianBlur(img_s, (0, 0), sigmaX=14)
    out = img_s.copy()
    out[busy == 1] = heavy_blur[busy == 1]
    return out


def generate_illustration_content(path, mode="objects", k=7):
    """写真から『イラスト部分』(content_crop, alpha_crop, img_s)を生成する共通処理"""
    img = cv2.imread(path)
    h, w = img.shape[:2]
    scale = 1000 / max(h, w)
    img_s = cv2.resize(img, (int(w * scale), int(h * scale)))
    hs, ws = img_s.shape[:2]

    subject_mask, keep_mask = get_masks(img_s, top_n_objects=3, mode=mode)
    img_s_flat = flatten_busy_patterns(img_s, keep_mask)
    shifted = cv2.pyrMeanShiftFiltering(img_s_flat, sp=28, sr=55, maxLevel=2)
    palette, names = quantize_palette(shifted, keep_mask, subject_mask=subject_mask, k=k, snap_to_tombow=True)
    print("palette:", names)

    quant = apply_palette(shifted, palette)
    quant = cv2.medianBlur(quant, 9)
    quant = merge_small_fragments(quant, keep_mask, min_area=400)
    quant = boost_saturation(quant, factor=1.4)

    ys, xs = np.where(keep_mask == 1)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    pad = 22
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(hs, y1 + pad), min(ws, x1 + pad)

    content = np.ones((hs, ws, 3), dtype=np.uint8) * 238
    content[keep_mask == 1] = quant[keep_mask == 1]

    keep_clean = cv2.morphologyEx(keep_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    num_labels, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(keep_clean, connectivity=8)
    small_noise = np.zeros_like(keep_mask)
    for i in range(1, num_labels):
        if stats_cc[i, cv2.CC_STAT_AREA] < 150:
            small_noise[labels_cc == i] = 1
    content[small_noise == 1] = 238
    keep_mask_for_draw = np.maximum(keep_mask - small_noise, 0).astype(np.uint8)

    ink_color = (30, 30, 30)

    gray_full = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
    # 正面顔が見つからない場合は横顔(haarcascade_profileface)をフォールバックで試す
    faces = detect_faces(gray_full, keep_mask, frontal_min_neighbors=3, frontal_min_size=(30, 30))
    for (fx, fy, fw, fh, facing) in faces:
        if facing == 'front':
            draw_landmark_face(content, img_s, fx, fy, fw, fh, ink_color)
        else:
            draw_landmark_face_profile(content, img_s, fx, fy, fw, fh, ink_color, facing=facing)

    content = draw_flat_outlines(content, keep_mask_for_draw, quant, ink_color, thickness=1, epsilon=2.5)

    contours, _ = cv2.findContours(keep_mask_for_draw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 200:
            continue
        approx = cv2.approxPolyDP(cnt, 2.0, True)
        cv2.polylines(content, [approx], True, ink_color, 3, cv2.LINE_AA)

    content_crop = content[y0:y1, x0:x1]
    alpha_crop = keep_mask_for_draw[y0:y1, x0:x1].astype(np.float32)
    alpha_crop = cv2.GaussianBlur(alpha_crop, (5, 5), 0)

    fh_, fw_ = alpha_crop.shape
    feather_px = 20
    ramp_y = np.ones(fh_, dtype=np.float32)
    ramp_y[:feather_px] = np.linspace(0, 1, feather_px)
    ramp_y[-feather_px:] = np.linspace(1, 0, feather_px)
    ramp_x = np.ones(fw_, dtype=np.float32)
    ramp_x[:feather_px] = np.linspace(0, 1, feather_px)
    ramp_x[-feather_px:] = np.linspace(1, 0, feather_px)
    border_fade = ramp_y[:, None] * ramp_x[None, :]
    alpha_crop = alpha_crop * border_fade
    return content_crop, alpha_crop, img


def place_content_on_paper(canvas, content_crop, alpha_crop, place_x, place_y, target_h):
    ch, cw = content_crop.shape[:2]
    target_w = int(cw * (target_h / ch))
    content_resized = cv2.resize(content_crop, (target_w, target_h))
    alpha_resized = cv2.resize(alpha_crop, (target_w, target_h))[:, :, None]
    region = canvas[place_y:place_y + target_h, place_x:place_x + target_w].astype(np.float32)
    blended = alpha_resized * content_resized.astype(np.float32) + (1 - alpha_resized) * region
    canvas[place_y:place_y + target_h, place_x:place_x + target_w] = blended.astype(np.uint8)


def draw_title(canvas, title_lines, text_x, text_y, line_gap, font_scale, underline_w):
    for i, line in enumerate(title_lines):
        cv2.putText(canvas, line, (text_x, text_y + i * line_gap),
                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX, font_scale, (70, 65, 60), 1, cv2.LINE_AA)
    cv2.line(canvas, (text_x, text_y + len(title_lines) * line_gap + 6),
             (text_x + underline_w, text_y + len(title_lines) * line_gap + 6), (70, 65, 60), 1, cv2.LINE_AA)


def build_flat_style(path, out_path, title_lines=("Good", "Days"), mode="objects", k=7):
    content_crop, alpha_crop, _ = generate_illustration_content(path, mode, k)

    CANVAS_W = 900
    HALF_H = 600
    canvas = make_paper_texture(HALF_H, CANVAS_W, base=250)
    target_h = int(HALF_H * 0.42)
    place_x = int(CANVAS_W * 0.08)
    place_y = HALF_H - target_h - int(HALF_H * 0.10)
    place_content_on_paper(canvas, content_crop, alpha_crop, place_x, place_y, target_h)
    draw_title(canvas, title_lines, int(CANVAS_W * 0.62), int(HALF_H * 0.45), 34, 0.9, 90)

    cv2.imwrite(out_path, canvas)


def color_grade_photo(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    l = np.clip((l - 128) * 1.05 + 128 + 4, 0, 255)
    lab = cv2.merge([l, a, b]).astype(np.uint8)
    graded = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(graded, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.92
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def crop_to_ratio(img, target_w, target_h):
    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    cur_ratio = w / h
    if cur_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        img = img[:, x0:x0 + new_w]
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        img = img[y0:y0 + new_h, :]
    return cv2.resize(img, (target_w, target_h))


def build_flat_style_a4(path, out_path, title_lines=("Good", "Days"), mode="objects", k=7):
    """A4比率(1:1.414)のフルポスター：上半分=写真、下半分=紙+小さいイラスト。
    イラスト自体のサイズは3:4版と同じ小ささのまま、紙の余白だけが増える。"""
    content_crop, alpha_crop, img = generate_illustration_content(path, mode, k)

    A4_W = 900
    A4_HALF_H = int(A4_W * (297 / 210) / 2)  # A4全体の半分の高さ

    top = crop_to_ratio(img, A4_W, A4_HALF_H)
    top = color_grade_photo(top)

    bottom = make_paper_texture(A4_HALF_H, A4_W, base=250)
    target_h = int(A4_HALF_H * 0.30)  # 3:4版よりさらに小さめ（紙が広がった分、余白重視）
    place_x = int(A4_W * 0.08)
    place_y = A4_HALF_H - target_h - int(A4_HALF_H * 0.14)
    place_content_on_paper(bottom, content_crop, alpha_crop, place_x, place_y, target_h)
    draw_title(bottom, title_lines, int(A4_W * 0.62), int(A4_HALF_H * 0.32), 34, 0.9, 90)

    canvas = np.vstack([top, bottom])
    cv2.imwrite(out_path, canvas)


if __name__ == "__main__":
    build_flat_style('/mnt/user-data/uploads/IMG_1141.jpeg', '/home/claude/flat_br.png', ('Sweet', 'Treat'))
    build_flat_style('/mnt/user-data/uploads/IMG_1042.jpeg', '/home/claude/flat_platform.png', ('Train', 'Days'))
    build_flat_style('/mnt/user-data/uploads/IMG_0943.jpeg', '/home/claude/flat_dog.png', ('Good', 'Boy'))
