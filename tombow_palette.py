import cv2
import numpy as np

# Tombow ABT マスターパレット
# 土台は公式「Portrait 10」セット（879, 899, 910, 912, 942, 947, 977, 990, 992, N15）+ アクセント4色
# 注意：以下のHEX値は公式スウォッチ画像（画像のため直接読み取り不可）ではなく、
# 色名（Saddle Brown, Burnt Sienna等）から一般的に連想される近似値。
# 実際の印刷・現物との色合わせをする場合は、Tombow公式のPDFスウォッチか実物のペンで最終確認が必要。
# Tombow ABT マスターパレット
# 土台は公式「Portrait 10」セット + アクセント。
# HEX値はユーザー提供の公式カラーチャート画像(IMG_1396.PNG)から実測（ピクセルサンプリング）した値。
# 近似ではなく実測値なので、精度はこれまでより高い。
TOMBOW_MASTER_HEX = {
    "020 Peach":        "FDF7D1",
    "850 Flesh":        "FCD8C4",
    "879 Brown":        "6B3B31",
    "899 Redwood":      "863F2C",
    "912 Pale Cherry":  "E4A45B",
    "942 Tan":          "E8D2AE",
    "947 Burnt Sienna": "B14421",
    "977 Saddle Brown": "CA7F50",
    "990 Light Sand":   "FCEFB3",
    "992 Sand":         "D6AD54",
    "N15 Black":        "000022",
    "528 Navy Blue":    "016DAA",
    "373 Sea Blue":     "029394",
    "245 Sap Green":    "01A464",
    "N57 Warm Gray 5":  "8F886A",
}

def hex_to_bgr(h):
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

MASTER_BGR = np.array([hex_to_bgr(v) for v in TOMBOW_MASTER_HEX.values()], dtype=np.float32)
MASTER_NAMES = list(TOMBOW_MASTER_HEX.keys())


def snap_palette_to_master(palette_bgr):
    """harmonize_palette後のパレット色を、Tombowマスターパレットの最近傍色に置き換える（Lab空間で比較）"""
    def to_lab(bgr_arr):
        img = bgr_arr.reshape(-1, 1, 3).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    palette_lab = to_lab(palette_bgr)
    master_lab = to_lab(MASTER_BGR.astype(np.uint8))

    snapped = []
    used_names = []
    for p in palette_lab:
        dists = np.linalg.norm(master_lab - p[None, :], axis=1)
        idx = np.argmin(dists)
        snapped.append(MASTER_BGR[idx])
        used_names.append(MASTER_NAMES[idx])
    return np.array(snapped, dtype=np.uint8), used_names
