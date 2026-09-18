import cv2
import numpy as np
from rembg import remove, new_session

session = new_session('u2net')


def get_masks(img_s, top_n_objects=3, mode="objects", ellipse_margin=1.35, allow_landmark_exception=True):
    hs, ws = img_s.shape[:2]
    rgba = remove(img_s, session=session)
    alpha = rgba[:, :, 3]
    subject_mask = (alpha > 127).astype(np.uint8)

    inv = 1 - subject_mask
    dist = cv2.distanceTransform(inv.astype(np.uint8), cv2.DIST_L2, 5)
    dist_norm = dist / (dist.max() + 1e-6)

    gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_dil = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    edge_density = cv2.boxFilter(edges_dil.astype(np.float32) / 255, -1, (31, 31))
    edge_density_n = edge_density / (edge_density.max() + 1e-6)

    hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    sat_mean = cv2.boxFilter(sat, -1, (31, 31))
    sat_sq_mean = cv2.boxFilter(sat ** 2, -1, (31, 31))
    sat_var = np.sqrt(np.maximum(sat_sq_mean - sat_mean ** 2, 0))
    sat_var_n = sat_var / (sat_var.max() + 1e-6)

    img_f = img_s.astype(np.float32)
    small_mean = cv2.boxFilter(img_f, -1, (9, 9))
    large_mean = cv2.boxFilter(img_f, -1, (101, 101))
    color_contrast = np.linalg.norm(small_mean - large_mean, axis=2)
    color_contrast_n = color_contrast / (color_contrast.max() + 1e-6)

    keep_score = 0.45 * color_contrast_n + 0.15 * edge_density_n + 0.10 * sat_var_n + 0.30 * (1 - dist_norm)
    keep_score[subject_mask == 1] = 1.0
    keep_mask = (keep_score > 0.42).astype(np.uint8)
    keep_mask = cv2.morphologyEx(keep_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    keep_mask = cv2.morphologyEx(keep_mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))

    bg_keep = keep_mask.copy()
    bg_keep[subject_mask == 1] = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bg_keep, connectivity=8)
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: -x[1])

    def is_clean_object(i):
        region_mask = (labels == i).astype(np.uint8)
        internal_edge_density = edges_dil[region_mask == 1].mean() / 255.0
        return internal_edge_density <= 0.30

    if mode == "objects":
        keep_ids = []
        for i, a in areas:
            if a <= 400 or not is_clean_object(i):
                continue
            keep_ids.append(i)
            if len(keep_ids) >= top_n_objects:
                break
        bg_keep_filtered = np.isin(labels, keep_ids).astype(np.uint8)
        final_keep = np.maximum(subject_mask, bg_keep_filtered)
        return subject_mask, final_keep

    elif mode == "ellipse":
        ys, xs = np.where(subject_mask == 1)
        if len(ys) == 0:
            return subject_mask, subject_mask
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        bh, bw = (y1 - y0), (x1 - x0)
        # 被写体の縦横比に応じて楕円の向きを決める
        axis_y = int(bh / 2 * ellipse_margin)
        axis_x = int(bw / 2 * ellipse_margin)
        ellipse_mask = np.zeros((hs, ws), np.uint8)
        cv2.ellipse(ellipse_mask, (int(cx), int(cy)), (max(axis_x, 10), max(axis_y, 10)), 0, 0, 360, 1, -1)

        final_keep = np.maximum(subject_mask, ellipse_mask & keep_mask.astype(np.uint8) | (ellipse_mask & subject_mask))
        # 楕円の外にはみ出しても残す「ランドマーク例外」：大きく・内部エッジが騒がしくない物体のみ許可
        if allow_landmark_exception:
            landmark_ids = []
            for i, a in areas:
                if a <= 900 or not is_clean_object(i):
                    continue
                landmark_ids.append(i)
                if len(landmark_ids) >= 1:  # 例外は1個までに絞る(参考画像でも背景要素はゼロ〜1個)
                    break
            landmark_mask = np.isin(labels, landmark_ids).astype(np.uint8)
            final_keep = np.maximum(final_keep, landmark_mask)
        final_keep = cv2.morphologyEx(final_keep.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return subject_mask, final_keep

    else:
        raise ValueError(mode)


def quantize_palette(img_s, keep_mask, subject_mask=None, k=5, snap_to_tombow=False):
    bg_only = keep_mask.copy()
    if subject_mask is not None:
        bg_only[subject_mask == 1] = 0
    subj_pixels = img_s[subject_mask == 1].reshape(-1, 3) if subject_mask is not None else np.empty((0, 3), np.uint8)
    bg_pixels = img_s[bg_only == 1].reshape(-1, 3)
    # 被写体のピクセルを2倍の重みでサンプリング（k を増やしたので過度な重み付けは不要）
    if len(subj_pixels) > 0:
        pixels = np.vstack([subj_pixels, subj_pixels, bg_pixels]).astype(np.float32)
    else:
        pixels = bg_pixels.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    if snap_to_tombow:
        # Tombowにスナップする場合は、harmonize（彩度・明度の強制調整）をかけない。
        # マスターパレット自体がすでに「上品に絞られた色」なので、二重に歪めると
        # 実際の写真の色から離れてしまう（肌がピンクに寄る等の原因になっていた）
        from tombow_palette import snap_palette_to_master
        snapped, names = snap_palette_to_master(centers.astype(np.uint8))
        return snapped, names
    palette = harmonize_palette(centers.astype(np.uint8))
    return palette


def harmonize_palette(palette_bgr):
    """サンプルのような『上品だが単調すぎない』配色に寄せる後処理。
    - 彩度を少し持ち上げてくすみを取る
    - 明度を35〜88の範囲に伸ばして「真っ黒」「真っ白」を避ける
    """
    hsv = cv2.cvtColor(palette_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, 0, 0], hsv[:, 0, 1], hsv[:, 0, 2]

    s = np.clip(s * 1.22 + 6, 0, 255)

    v_min, v_max = v.min(), v.max()
    if v_max - v_min > 1e-3:
        v_stretched = (v - v_min) / (v_max - v_min) * (0.88 - 0.35) * 255 + 0.35 * 255
    else:
        v_stretched = v
    v = np.clip(v_stretched, 0, 255)

    hsv[:, 0, 1] = s
    hsv[:, 0, 2] = v
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out.reshape(-1, 3)


def apply_palette(img_s, palette):
    h, w = img_s.shape[:2]
    flat = img_s.reshape(-1, 3).astype(np.float32)
    dists = np.linalg.norm(flat[:, None, :] - palette[None, :, :].astype(np.float32), axis=2)
    nearest = np.argmin(dists, axis=1)
    return palette[nearest].reshape(h, w, 3)


def brush_texture(h, w, scales=(6, 14, 30), weights=(0.5, 0.32, 0.18), streak_angle=None):
    """複数スケールのランダムノイズを合成して『ドライブラシ』っぽい濃淡ムラを作る"""
    combined = np.zeros((h, w), np.float32)
    for s, wgt in zip(scales, weights):
        small = np.random.rand(max(2, h // s), max(2, w // s)).astype(np.float32)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        combined += wgt * up
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-6)
    if streak_angle is not None:
        length = 9
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((length / 2, length / 2), streak_angle, 1)
        kernel = cv2.warpAffine(kernel, M, (length, length))
        kernel /= kernel.sum() + 1e-6
        combined = cv2.filter2D(combined, -1, kernel)
    return combined


def apply_brush_texture(quant_img, keep_mask, strength=0.30):
    hs, ws = quant_img.shape[:2]
    angle = np.random.uniform(-25, 25)
    noise = brush_texture(hs, ws, streak_angle=angle)
    value_mod = 1 + (noise - 0.5) * strength
    out = np.clip(quant_img.astype(np.float32) * value_mod[..., None], 0, 255).astype(np.uint8)
    result = quant_img.copy()
    result[keep_mask == 1] = out[keep_mask == 1]
    return result


def make_paper_texture(h, w, base=238):
    noise = np.random.normal(0, 5, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.2)
    paper_gray = np.clip(base + noise, 216, 248).astype(np.float32)
    # 暖色系のクリーム紙に寄せる（B成分を落とし、R成分を少し足す）
    paper_bgr = np.stack([paper_gray - 10, paper_gray - 3, paper_gray + 4], axis=-1)
    return np.clip(paper_bgr, 0, 255).astype(np.uint8)


def draw_handdrawn_lines(canvas, edge_mask, ink_color, jitter=1.6, epsilon=1.8, min_len=26):
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.arcLength(cnt, False) < min_len:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, False)
        pts = approx.reshape(-1, 2).astype(np.float32)
        noise = np.random.uniform(-jitter, jitter, pts.shape)
        pts_j = (pts + noise).astype(np.int32)
        cv2.polylines(canvas, [pts_j], isClosed=False, color=ink_color, thickness=1, lineType=cv2.LINE_AA)
    return canvas


def color_grade_photo(img):
    """editorial風の軽いトーン補正（コントラスト・彩度をわずかに調整）"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    l = np.clip((l - 128) * 1.05 + 128 + 4, 0, 255)
    lab = cv2.merge([l, a, b]).astype(np.uint8)
    graded = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(graded, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.92  # ほんの少し彩度を落として上品に
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return graded


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


def build_poster(path, out_path, title_lines=("Good", "Days"), mode="objects", snap_to_tombow=False, k=5):
    CANVAS_W = 900
    HALF_H = 600  # 上下それぞれの高さ（3:4トータル= 900x1200）

    img = cv2.imread(path)
    h, w = img.shape[:2]
    scale = 900 / max(h, w)
    img_s = cv2.resize(img, (int(w * scale), int(h * scale)))
    hs, ws = img_s.shape[:2]

    subject_mask, keep_mask = get_masks(img_s, top_n_objects=3, mode=mode)

    shifted = cv2.pyrMeanShiftFiltering(img_s, sp=16, sr=32, maxLevel=1)
    pal_result = quantize_palette(shifted, keep_mask, subject_mask=subject_mask, k=k, snap_to_tombow=snap_to_tombow)
    if snap_to_tombow:
        palette, tombow_names = pal_result
        print("Tombow palette used:", tombow_names)
    else:
        palette = pal_result
    quant = apply_palette(shifted, palette)
    quant = cv2.medianBlur(quant, 7)
    quant = apply_brush_texture(quant, keep_mask, strength=0.30)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.bitwise_and(edges, edges, mask=keep_mask * 255)

    # keep_maskの外接矩形でイラスト内容をクロップ
    ys, xs = np.where(keep_mask == 1)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    pad = 22
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(hs, y1 + pad), min(ws, x1 + pad)

    content = np.ones((hs, ws, 3), dtype=np.uint8) * 238
    content[keep_mask == 1] = quant[keep_mask == 1]

    # keep_mask内の小さすぎる孤立領域（ノイズ状の点）を除去してから線を描く
    keep_clean = cv2.morphologyEx(keep_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    num_labels, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(keep_clean, connectivity=8)
    small_noise = np.zeros_like(keep_mask)
    for i in range(1, num_labels):
        if stats_cc[i, cv2.CC_STAT_AREA] < 120:
            small_noise[labels_cc == i] = 1
    content[small_noise == 1] = 238
    keep_mask_for_draw = np.maximum(keep_mask - small_noise, 0).astype(np.uint8)

    content = draw_handdrawn_lines(content, edges, (58, 70, 82))
    # 線をわずかにぼかして硬いベクター感を抜く（完全にはボケさせない）
    ink_layer = content.copy()
    content = cv2.addWeighted(content, 0.55, cv2.GaussianBlur(ink_layer, (3, 3), 0), 0.45, 0)
    content_crop = content[y0:y1, x0:x1]
    alpha_crop = keep_mask_for_draw[y0:y1, x0:x1].astype(np.float32)
    alpha_crop = cv2.GaussianBlur(alpha_crop, (9, 9), 0)

    # クロップ矩形の縁全体を内側に向けてフェードさせ、「四角いカード」感を消す
    fh, fw = alpha_crop.shape
    feather_px = 24
    ramp_y = np.ones(fh, dtype=np.float32)
    ramp_y[:feather_px] = np.linspace(0, 1, feather_px)
    ramp_y[-feather_px:] = np.linspace(1, 0, feather_px)
    ramp_x = np.ones(fw, dtype=np.float32)
    ramp_x[:feather_px] = np.linspace(0, 1, feather_px)
    ramp_x[-feather_px:] = np.linspace(1, 0, feather_px)
    border_fade = ramp_y[:, None] * ramp_x[None, :]
    alpha_crop = alpha_crop * border_fade

    # --- ボトムハーフ：紙背景に、切り出したイラストを小さく配置 ---
    bottom = make_paper_texture(HALF_H, CANVAS_W)
    ch, cw = content_crop.shape[:2]
    target_h = int(HALF_H * 0.33)  # 被写体をより小さく、余白を広く
    target_w = int(cw * (target_h / ch))
    content_resized = cv2.resize(content_crop, (target_w, target_h))
    alpha_resized = cv2.resize(alpha_crop, (target_w, target_h))[:, :, None]

    place_x = int(CANVAS_W * 0.08)
    place_y = HALF_H - target_h - int(HALF_H * 0.12)
    region = bottom[place_y:place_y + target_h, place_x:place_x + target_w].astype(np.float32)
    blended = alpha_resized * content_resized.astype(np.float32) + (1 - alpha_resized) * region
    bottom[place_y:place_y + target_h, place_x:place_x + target_w] = blended.astype(np.uint8)

    # --- タイポグラフィ（右側の余白に縦スタック） ---
    text_x = int(CANVAS_W * 0.62)
    text_y = int(HALF_H * 0.45)
    for i, line in enumerate(title_lines):
        cv2.putText(bottom, line, (text_x, text_y + i * 34),
                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (70, 65, 60), 1, cv2.LINE_AA)
    cv2.line(bottom, (text_x, text_y + len(title_lines) * 34 + 6),
             (text_x + 90, text_y + len(title_lines) * 34 + 6), (70, 65, 60), 1, cv2.LINE_AA)

    # --- トップハーフ：元写真をエディトリアル色調補正して3:4上半分に ---
    top = crop_to_ratio(img, CANVAS_W, HALF_H)
    top = color_grade_photo(top)

    canvas = np.vstack([top, bottom])
    cv2.imwrite(out_path, canvas)


if __name__ == "__main__":
    pass
