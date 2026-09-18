import cv2
import numpy as np
from poster_pipeline import (
    get_masks, quantize_palette, harmonize_palette,
    draw_handdrawn_lines
)
from tombow_palette import MASTER_BGR, MASTER_NAMES


def apply_palette_with_index(img_s, palette):
    """最近傍色だけでなく、どのパレット番号に属するかも返す"""
    h, w = img_s.shape[:2]
    flat = img_s.reshape(-1, 3).astype(np.float32)
    dists = np.linalg.norm(flat[:, None, :] - palette[None, :, :].astype(np.float32), axis=2)
    nearest = np.argmin(dists, axis=1)
    quant = palette[nearest].reshape(h, w, 3)
    index_map = nearest.reshape(h, w)
    return quant, index_map


def build_paint_by_number(path, out_prefix, top_n_objects=3, mode="objects"):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    scale = 900 / max(h, w)
    img_s = cv2.resize(img, (int(w * scale), int(h * scale)))
    hs, ws = img_s.shape[:2]

    subject_mask, keep_mask = get_masks(img_s, top_n_objects=top_n_objects, mode=mode)
    shifted = cv2.pyrMeanShiftFiltering(img_s, sp=24, sr=45, maxLevel=2)

    # k=5でクラスタ化 → Tombowマスターパレットにスナップ（ここでは色そのものより「使用パレットのインデックス」が重要）
    bg_only = keep_mask.copy()
    bg_only[subject_mask == 1] = 0
    subj_pixels = shifted[subject_mask == 1].reshape(-1, 3)
    bg_pixels = shifted[bg_only == 1].reshape(-1, 3)
    pixels = np.vstack([subj_pixels, subj_pixels, subj_pixels, bg_pixels]).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centers = cv2.kmeans(pixels, 5, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    palette = harmonize_palette(centers.astype(np.uint8))

    # Tombowマスターパレットへスナップ：パレットの各色 -> マスター上のインデックス
    def to_lab(bgr_arr):
        return cv2.cvtColor(bgr_arr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    palette_lab = to_lab(palette)
    master_lab = to_lab(MASTER_BGR.astype(np.uint8))
    palette_to_master = []
    for p in palette_lab:
        dists = np.linalg.norm(master_lab - p[None, :], axis=1)
        palette_to_master.append(int(np.argmin(dists)))

    quant, index_map = apply_palette_with_index(shifted, palette)
    # index_map の値(0..4) を master番号に変換
    master_index_map = np.vectorize(lambda i: palette_to_master[i])(index_map)

    # keep_mask内で実際に使われているmaster番号だけ抽出し、面積順に1,2,3...を割り振る
    used_master_ids = []
    for mid in np.unique(master_index_map[keep_mask == 1]):
        area = np.sum((master_index_map == mid) & (keep_mask == 1))
        used_master_ids.append((mid, area))
    used_master_ids.sort(key=lambda x: -x[1])
    master_to_number = {mid: i + 1 for i, (mid, _) in enumerate(used_master_ids)}

    # --- 手描き風の輪郭線を抽出（既存ロジック流用） ---
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.bitwise_and(edges, edges, mask=keep_mask * 255)

    # 色の境界線も追加（同じmaster番号内の塊ごとに境界がわかるように）
    # 小さな断片で線が細切れにならないよう、クロージングしてから輪郭抽出する
    color_boundary = np.zeros((hs, ws), np.uint8)
    for mid, _ in used_master_ids:
        region = ((master_index_map == mid) & (keep_mask == 1)).astype(np.uint8)
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 300]
        cv2.drawContours(color_boundary, cnts, -1, 255, 1)

    combined_edges = color_boundary  # ペイント・バイ・ナンバー用は色境界線のみ（毛並み等の写真テクスチャの線は除外）

    # --- クロップ範囲 ---
    ys, xs = np.where(keep_mask == 1)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    pad = 20
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(hs, y1 + pad), min(ws, x1 + pad)

    # --- (1) 線画＋番号だけのテンプレート（白背景） ---
    template = np.ones((hs, ws, 3), dtype=np.uint8) * 255
    template = draw_handdrawn_lines(template, combined_edges, (120, 120, 120), jitter=1.2, epsilon=1.6, min_len=10)

    # 各カラー領域の連結成分ごとに番号を書き込む（小さすぎる断片はクロージングでまとめてから、
    # それでも小さいものは塗り分け不可能として無視する）
    for mid, _ in used_master_ids:
        region = ((master_index_map == mid) & (keep_mask == 1)).astype(np.uint8)
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        num_labels, labels_cc, stats_cc, centroids = cv2.connectedComponentsWithStats(region, connectivity=8)
        for i in range(1, num_labels):
            if stats_cc[i, cv2.CC_STAT_AREA] < 500:
                continue
            cx, cy = centroids[i]
            text = str(master_to_number[mid])
            cv2.putText(template, text, (int(cx) - 6, int(cy) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)

    template_crop = template[y0:y1, x0:x1]

    # --- (2) 答え合わせ用：フルカラーの完成見本 ---
    reference = np.ones((hs, ws, 3), dtype=np.uint8) * 255
    reference[keep_mask == 1] = quant[keep_mask == 1]
    reference = draw_handdrawn_lines(reference, combined_edges, (50, 60, 70), jitter=1.2, epsilon=1.6, min_len=10)
    reference_crop = reference[y0:y1, x0:x1]

    # --- (3) 色番号の凡例 ---
    legend_h = 40 * (len(used_master_ids) + 1)
    legend = np.ones((legend_h, 500, 3), dtype=np.uint8) * 255
    cv2.putText(legend, "Color Key", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    for row, (mid, _) in enumerate(used_master_ids):
        y = 40 * (row + 1) + 25
        num = master_to_number[mid]
        color_bgr = tuple(int(c) for c in MASTER_BGR[mid])
        cv2.rectangle(legend, (10, y - 20), (50, y + 5), color_bgr, -1)
        cv2.rectangle(legend, (10, y - 20), (50, y + 5), (60, 60, 60), 1)
        cv2.putText(legend, f"{num}  =  Tombow {MASTER_NAMES[mid]}", (60, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)

    cv2.imwrite(f'{out_prefix}_template.png', template_crop)
    cv2.imwrite(f'{out_prefix}_reference.png', reference_crop)
    cv2.imwrite(f'{out_prefix}_legend.png', legend)
    return used_master_ids, master_to_number


if __name__ == "__main__":
    build_paint_by_number('/mnt/user-data/uploads/IMG_0943.jpeg', '/home/claude/pbn_dog')
    print("done")
