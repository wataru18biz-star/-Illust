import cv2
import numpy as np
from poster_pipeline import (
    get_masks, quantize_palette, apply_palette, make_paper_texture,
    draw_handdrawn_lines, apply_brush_texture
)
from tombow_palette import snap_palette_to_master, MASTER_BGR, MASTER_NAMES

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# 肌色として使ってよい候補だけに絞る（グレー・青・緑・黒は肌には使わない）
SKIN_NAMES = ["020 Peach", "850 Flesh", "879 Brown", "899 Redwood", "912 Pale Cherry",
              "942 Tan", "947 Burnt Sienna", "977 Saddle Brown", "990 Light Sand", "992 Sand"]
_skin_idx = [MASTER_NAMES.index(n) for n in SKIN_NAMES]
SKIN_BGR = MASTER_BGR[_skin_idx]


def snap_one_color(bgr, candidates_bgr=None, candidates_names=None):
    if candidates_bgr is None:
        candidates_bgr, candidates_names = SKIN_BGR, SKIN_NAMES
    lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
    master_lab = cv2.cvtColor(candidates_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    idx = np.argmin(np.linalg.norm(master_lab - lab[None, :], axis=1))
    return tuple(int(c) for c in candidates_bgr[idx]), candidates_names[idx]


def draw_landmark_face(content, img_s, fx, fy, fw, fh, ink_color):
    """顔ボックス内を『意味のある形』として直接描く：
    - 顔全体は肌の平均色を1色でフラット塗り
    - 目はHaar cascadeで検出した位置に楕円
    - 口・頬は顔の比率から決め打ちで配置
    """
    # 肌色の平均を取る範囲は、前髪・帽子の影・輪郭の外側を避けて
    # 頬〜鼻筋あたり（顔の中央やや下、横幅も少し絞る）だけに限定する
    sx0, sx1 = int(fw * 0.28), int(fw * 0.72)
    sy0, sy1 = int(fh * 0.54), int(fh * 0.66)
    sample_region = img_s[fy + sy0:fy + sy1, fx + sx0:fx + sx1]
    if sample_region.size == 0:
        sample_region = img_s[fy:fy + fh, fx:fx + fw]
    # 明るさが極端に低い/高いピクセル（影・ハイライト）を除いた中央値を使う
    pixels = sample_region.reshape(-1, 3).astype(np.float32)
    brightness = pixels.mean(axis=1)
    lo, hi = np.percentile(brightness, 25), np.percentile(brightness, 75)
    mask_mid = (brightness >= lo) & (brightness <= hi)
    if mask_mid.sum() > 0:
        avg_color = pixels[mask_mid].mean(axis=0)
    else:
        avg_color = pixels.mean(axis=0)
    skin_color, skin_name = snap_one_color(avg_color)
    face_region = img_s[fy:fy + fh, fx:fx + fw]

    # 顔を楕円マスクでフラット塗り（矩形だと角が不自然なので楕円に）
    mask = np.zeros((fh, fw), np.uint8)
    cv2.ellipse(mask, (fw // 2, fh // 2), (fw // 2, int(fh * 0.52)), 0, 0, 360, 255, -1)
    roi = content[fy:fy + fh, fx:fx + fw]
    roi[mask > 0] = skin_color
    content[fy:fy + fh, fx:fx + fw] = roi

    # 目：Haar cascadeで検出、無ければ比率から仮置き
    face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(face_gray, minNeighbors=6)
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        for (ex, ey, ew, eh) in eyes:
            cx, cy = fx + ex + ew // 2, fy + ey + eh // 2
            cv2.ellipse(content, (cx, cy), (max(3, ew // 5), max(2, eh // 7)), 0, 0, 360, ink_color, -1)
    else:
        for frac_x in (0.30, 0.70):
            cx, cy = fx + int(fw * frac_x), fy + int(fh * 0.42)
            cv2.ellipse(content, (cx, cy), (max(3, fw // 12), max(2, fh // 20)), 0, 0, 360, ink_color, -1)

    # 口：比率から少し弧を描いた線（笑顔らしさを出す）＋太めにして視認性を上げる
    mx0, mx1 = fx + int(fw * 0.32), fx + int(fw * 0.68)
    my = fy + int(fh * 0.74)
    mouth_pts = np.array([
        [mx0, my],
        [(mx0 + mx1) // 2, my + int(fh * 0.035)],
        [mx1, my]
    ], dtype=np.int32)
    cv2.polylines(content, [mouth_pts], False, ink_color, 2, cv2.LINE_AA)

    # 眉：目の少し上に軽いストローク
    for frac_x in (0.30, 0.70):
        bx0 = fx + int(fw * (frac_x - 0.10))
        bx1 = fx + int(fw * (frac_x + 0.10))
        by = fy + int(fh * 0.33)
        cv2.line(content, (bx0, by), (bx1, by - 2), ink_color, 2, cv2.LINE_AA)
    # 頬の陰影：肌色より少し暗いトーンを頬骨あたりに薄く
    shade = tuple(int(c * 0.85) for c in skin_color)
    for frac_x in (0.22, 0.78):
        cx, cy = fx + int(fw * frac_x), fy + int(fh * 0.58)
        overlay = content.copy()
        cv2.ellipse(overlay, (cx, cy), (int(fw * 0.14), int(fh * 0.10)), 0, 0, 360, shade, -1)
        content[:] = cv2.addWeighted(content, 0.72, overlay, 0.28, 0)

    # 顔の輪郭線
    cv2.ellipse(content, (fx + fw // 2, fy + fh // 2), (fw // 2, int(fh * 0.52)), 0, 0, 360, ink_color, 1, cv2.LINE_AA)
    return skin_name


def draw_landmark_face_profile(content, img_s, fx, fy, fw, fh, ink_color, facing='right'):
    """横顔用：正面顔と違い目・鼻・口は片側（顔が向いている側）にだけ描く。
    facing='right'は鼻が検出ボックスの右寄り、'left'は左寄りにあるケース。"""

    def fxr(frac):
        # frac=0(耳側/奥)〜1(鼻先側/手前)を、facingに応じて実座標に変換。
        # 1を超える値を渡すと鼻先のように輪郭の外側へ飛び出させられる。
        return fx + int(fw * frac) if facing == 'right' else fx + int(fw * (1 - frac))

    sx_a, sx_b = fxr(0.40), fxr(0.78)
    sx0, sx1 = min(sx_a, sx_b), max(sx_a, sx_b)
    sy0, sy1 = fy + int(fh * 0.46), fy + int(fh * 0.68)
    sample_region = img_s[sy0:sy1, sx0:sx1]
    if sample_region.size == 0:
        sample_region = img_s[fy:fy + fh, fx:fx + fw]
    pixels = sample_region.reshape(-1, 3).astype(np.float32)
    brightness = pixels.mean(axis=1)
    lo, hi = np.percentile(brightness, 25), np.percentile(brightness, 75)
    mask_mid = (brightness >= lo) & (brightness <= hi)
    avg_color = pixels[mask_mid].mean(axis=0) if mask_mid.sum() > 0 else pixels.mean(axis=0)
    skin_color, skin_name = snap_one_color(avg_color)

    mask = np.zeros((fh, fw), np.uint8)
    cv2.ellipse(mask, (fw // 2, fh // 2), (int(fw * 0.46), int(fh * 0.52)), 0, 0, 360, 255, -1)
    roi = content[fy:fy + fh, fx:fx + fw]
    roi[mask > 0] = skin_color
    content[fy:fy + fh, fx:fx + fw] = roi

    # 目：前方(鼻先側)寄りに1つだけ
    ex, ey = fxr(0.56), fy + int(fh * 0.42)
    cv2.ellipse(content, (ex, ey), (max(3, fw // 9), max(2, fh // 16)), 0, 0, 360, ink_color, -1)

    # 眉
    bx0, bx1 = fxr(0.42), fxr(0.68)
    by = fy + int(fh * 0.32)
    cv2.line(content, (bx0, by), (bx1, by - 2), ink_color, 2, cv2.LINE_AA)

    # 鼻：輪郭の少し外側に飛び出す鉤形の線
    nose_pts = np.array([
        [fxr(0.80), fy + int(fh * 0.36)],
        [fxr(1.06), fy + int(fh * 0.50)],
        [fxr(0.84), fy + int(fh * 0.58)],
    ], dtype=np.int32)
    cv2.polylines(content, [nose_pts], False, ink_color, 2, cv2.LINE_AA)

    # 口：前方寄りの短い線
    mx0, mx1 = fxr(0.58), fxr(0.82)
    my = fy + int(fh * 0.70)
    cv2.line(content, (mx0, my), (mx1, my), ink_color, 2, cv2.LINE_AA)

    # 頬の陰影
    shade = tuple(int(c * 0.85) for c in skin_color)
    cx, cy = fxr(0.58), fy + int(fh * 0.56)
    overlay = content.copy()
    cv2.ellipse(overlay, (cx, cy), (int(fw * 0.16), int(fh * 0.11)), 0, 0, 360, shade, -1)
    content[:] = cv2.addWeighted(content, 0.72, overlay, 0.28, 0)

    # 輪郭線
    cv2.ellipse(content, (fx + fw // 2, fy + fh // 2), (int(fw * 0.46), int(fh * 0.52)), 0, 0, 360, ink_color, 1, cv2.LINE_AA)
    return skin_name


def _box_overlap_ratio(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / min(aw * ah, bw * bh)


def detect_faces(gray_full, keep_mask, frontal_min_neighbors=3, frontal_min_size=(30, 30),
                  profile_min_neighbors=4, profile_min_size=(30, 30)):
    """正面顔を最優先で検出し、正面で見つからない箇所は横顔（左右とも）を
    フォールバックとして試す。横顔は`haarcascade_profileface.xml`が片方向
    （検出ボックスの右寄りに鼻がある向き）にしか反応しないため、画像を
    左右反転してもう一度検出することでもう片方の向きも拾う。

    戻り値: [(fx, fy, fw, fh, facing)]。facingは'front' / 'right' / 'left'。
    'right'/'left'はdraw_landmark_face_profileのfacing引数にそのまま渡せる。
    """
    h, w = gray_full.shape[:2]
    results = []

    frontal = face_cascade.detectMultiScale(gray_full, minNeighbors=frontal_min_neighbors, minSize=frontal_min_size)
    for (fx, fy, fw, fh) in frontal:
        if keep_mask[fy:fy + fh, fx:fx + fw].mean() < 0.5:
            continue
        results.append((fx, fy, fw, fh, 'front'))

    profile_right = profile_cascade.detectMultiScale(gray_full, minNeighbors=profile_min_neighbors, minSize=profile_min_size)
    for (fx, fy, fw, fh) in profile_right:
        if keep_mask[fy:fy + fh, fx:fx + fw].mean() < 0.5:
            continue
        if any(_box_overlap_ratio((fx, fy, fw, fh), r[:4]) > 0.3 for r in results):
            continue
        results.append((fx, fy, fw, fh, 'right'))

    gray_flipped = cv2.flip(gray_full, 1)
    keep_flipped = cv2.flip(keep_mask, 1)
    profile_left = profile_cascade.detectMultiScale(gray_flipped, minNeighbors=profile_min_neighbors, minSize=profile_min_size)
    for (fx, fy, fw, fh) in profile_left:
        if keep_flipped[fy:fy + fh, fx:fx + fw].mean() < 0.5:
            continue
        orig_fx = w - fx - fw
        if any(_box_overlap_ratio((orig_fx, fy, fw, fh), r[:4]) > 0.3 for r in results):
            continue
        results.append((orig_fx, fy, fw, fh, 'left'))

    return results


def build_with_face_landmarks(path, out_path, title_lines=("Good", "Days"), mode="objects", k=9):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    scale = 1000 / max(h, w)
    img_s = cv2.resize(img, (int(w * scale), int(h * scale)))
    hs, ws = img_s.shape[:2]

    subject_mask, keep_mask = get_masks(img_s, top_n_objects=3, mode=mode)
    shifted = cv2.pyrMeanShiftFiltering(img_s, sp=16, sr=32, maxLevel=1)
    palette, names = quantize_palette(shifted, keep_mask, subject_mask=subject_mask, k=k, snap_to_tombow=True)

    quant = apply_palette(shifted, palette)
    quant = cv2.medianBlur(quant, 7)
    quant = apply_brush_texture(quant, keep_mask, strength=0.30)

    gray_full = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_full, 40, 120)
    edges = cv2.bitwise_and(edges, edges, mask=keep_mask * 255)

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
        if stats_cc[i, cv2.CC_STAT_AREA] < 120:
            small_noise[labels_cc == i] = 1
    content[small_noise == 1] = 238
    keep_mask_for_draw = np.maximum(keep_mask - small_noise, 0).astype(np.uint8)

    # --- ここが今回の変更点：顔検出→意味のある形で直接描く ---
    # 正面顔が見つからない場合は横顔(haarcascade_profileface)をフォールバックで試す
    ink_color = (58, 70, 82)
    faces = detect_faces(gray_full, keep_mask, frontal_min_neighbors=6, frontal_min_size=(40, 40))
    face_names = []
    for (fx, fy, fw, fh, facing) in faces:
        if facing == 'front':
            name = draw_landmark_face(content, img_s, fx, fy, fw, fh, ink_color)
        else:
            name = draw_landmark_face_profile(content, img_s, fx, fy, fw, fh, ink_color, facing=facing)
        face_names.append(f"{name}({facing})")

    content = draw_handdrawn_lines(content, edges, ink_color)
    ink_layer = content.copy()
    content = cv2.addWeighted(content, 0.55, cv2.GaussianBlur(ink_layer, (3, 3), 0), 0.45, 0)
    content_crop = content[y0:y1, x0:x1]
    alpha_crop = keep_mask_for_draw[y0:y1, x0:x1].astype(np.float32)
    alpha_crop = cv2.GaussianBlur(alpha_crop, (9, 9), 0)

    fh_, fw_ = alpha_crop.shape
    feather_px = 24
    ramp_y = np.ones(fh_, dtype=np.float32)
    ramp_y[:feather_px] = np.linspace(0, 1, feather_px)
    ramp_y[-feather_px:] = np.linspace(1, 0, feather_px)
    ramp_x = np.ones(fw_, dtype=np.float32)
    ramp_x[:feather_px] = np.linspace(0, 1, feather_px)
    ramp_x[-feather_px:] = np.linspace(1, 0, feather_px)
    border_fade = ramp_y[:, None] * ramp_x[None, :]
    alpha_crop = alpha_crop * border_fade

    CANVAS_W = 900
    HALF_H = 600
    canvas = make_paper_texture(HALF_H, CANVAS_W)
    ch, cw = content_crop.shape[:2]
    target_h = int(HALF_H * 0.40)
    target_w = int(cw * (target_h / ch))
    content_resized = cv2.resize(content_crop, (target_w, target_h))
    alpha_resized = cv2.resize(alpha_crop, (target_w, target_h))[:, :, None]

    place_x = int(CANVAS_W * 0.08)
    place_y = HALF_H - target_h - int(HALF_H * 0.10)
    region = canvas[place_y:place_y + target_h, place_x:place_x + target_w].astype(np.float32)
    blended = alpha_resized * content_resized.astype(np.float32) + (1 - alpha_resized) * region
    canvas[place_y:place_y + target_h, place_x:place_x + target_w] = blended.astype(np.uint8)

    text_x = int(CANVAS_W * 0.62)
    text_y = int(HALF_H * 0.45)
    for i, line in enumerate(title_lines):
        cv2.putText(canvas, line, (text_x, text_y + i * 34),
                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (70, 65, 60), 1, cv2.LINE_AA)
    cv2.line(canvas, (text_x, text_y + len(title_lines) * 34 + 6),
             (text_x + 90, text_y + len(title_lines) * 34 + 6), (70, 65, 60), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)
    print("faces processed:", face_names)


if __name__ == "__main__":
    build_with_face_landmarks('/mnt/user-data/uploads/IMG_1141.jpeg', '/home/claude/landmark_br.png', ('Sweet', 'Treat'))
    build_with_face_landmarks('/mnt/user-data/uploads/IMG_1042.jpeg', '/home/claude/landmark_platform.png', ('Train', 'Days'))
    build_with_face_landmarks('/mnt/user-data/uploads/IMG_1249.jpeg', '/home/claude/landmark_pool.png', ('Pool', 'Days'))
