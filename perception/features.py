"""Shared pixel-feature code for the NSM position regressor (paper section 3.4.2).

Feature vector, identical to what ``Executor.pixel_to_world_dual`` builds, with
cls_id prepended (the executor accepts this form when ``n_features_in_ == 14``):

    [cls_id, px1, py1, w1, h1, conf1, px2, py2, w2, h2, conf2, ee_x, ee_y, ee_z]

Camera 1 is agentview, camera 2 is robot0_eye_in_hand. Pixel values are
``int(xywhn * 256)`` (truncated, as the executor does) and camera 2 is all zeros
when the wrist camera misses. Nothing here reads simulator object state.
"""
import numpy as np

IMG = 256
CLASS_NAMES = {0: "blue cube", 1: "green cube", 2: "red cube", 3: "yellow cube"}
# robosuite cube textures Number1..4 render as these YOLO classes.
TEXTURE_TO_CLS = {1: 0, 2: 2, 3: 1, 4: 3}
RAW_NAMES = ["cls_id", "px1", "py1", "w1", "h1", "conf1",
             "px2", "py2", "w2", "h2", "conf2", "ee_x", "ee_y", "ee_z"]


def box_to_ints(box_xywhn):
    return [int(v * IMG) for v in box_xywhn]


def best_per_class(boxes):
    """boxes: iterable of [cls, conf, x, y, w, h] (normalised). Best-conf per class."""
    best = {}
    for b in boxes:
        c = int(b[0])
        if c not in best or b[1] > best[c][1]:
            best[c] = b
    return best


def feature_row(cls_id, box_a, box_w, ee):
    """Build one 14-feature row. box_* are [cls, conf, x, y, w, h] or None."""
    a = box_to_ints(box_a[2:6])
    if box_w is None:
        w, cw = [0, 0, 0, 0], 0.0
    else:
        w, cw = box_to_ints(box_w[2:6]), float(box_w[1])
    return [float(cls_id), *a, float(box_a[1]), *w, cw, float(ee[0]), float(ee[1]), float(ee[2])]


def iou_xyxy(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def xywhn_to_xyxy_px(b):
    x, y, w, h = b[0] * IMG, b[1] * IMG, b[2] * IMG, b[3] * IMG
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def match_boxes_to_cubes(boxes, cubes, cam_key, min_iou=0.15):
    """Pair detections with cubes by class and IoU against the projected GT box.

    Used for labelling only (it reads ground truth). Duplicate colours and
    bin/platform false positives are resolved here because the wrong same-class
    box has low IoU with the cube's projection. Returns {cube_index: box}.
    """
    from scipy.optimize import linear_sum_assignment

    out = {}
    for c in {cube["cls"] for cube in cubes}:
        idx = [i for i, cube in enumerate(cubes) if cube["cls"] == c and cube.get(cam_key) is not None]
        cand = [b for b in boxes if int(b[0]) == c]
        if not idx or not cand:
            continue
        m = np.array([[iou_xyxy(cubes[i][cam_key], xywhn_to_xyxy_px(b[2:6])) for b in cand] for i in idx])
        for r, k in zip(*linear_sum_assignment(-m)):
            if m[r, k] >= min_iou:
                out[idx[r]] = cand[k]
    return out


# ---- fixed camera geometry, measured from the sim (Kinova3, 256x256) -------------
# Wrist orientation assumes the nominal EEF orientation. Reset noise moves it by
# about 2 degrees (std) in Hanoi and CubeSorting, which pixels and EEF cannot see.
F_A, F_W, CX = 309.019336, 166.812848, 128.0
C_A = np.array([0.5, 0.0, 1.35])
R_A = np.array([[0.0, 0.706148, -0.708064], [1.0, 0.0, 0.0], [0.0, -0.708064, -0.706148]])
WRIST_OFFSET = np.array([0.0498, 0.0, 0.1451])
R_W = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
Z_PRIOR = 0.82
Z_RANGE = (0.78, 1.02)
MAX_RAY_GAP = 0.025


def _rays(u, v, f, R):
    cam = np.stack([(u - CX) / f, (v - CX) / f, np.ones_like(u)], axis=1)
    return cam @ R.T


def engineer(X):
    """Geometry from pixels and EEF only. X is the 14-feature matrix.

    Returns (base_z_candidate, agent ray terms, feature matrix). The agent ray comes
    from the box centre; the wrist ray, when present, resolves depth along it.
    """
    X = np.asarray(X, dtype=float)
    cls, u1, v1, w1, h1, c1 = X[:, 0], X[:, 1] + 0.5, X[:, 2] + 0.5, X[:, 3], X[:, 4], X[:, 5]
    u2, v2, w2, h2, c2 = X[:, 6] + 0.5, X[:, 7] + 0.5, X[:, 8], X[:, 9], X[:, 10]
    ee = X[:, 11:14]
    has_w = (X[:, 8] > 0) & (X[:, 9] > 0)

    da = _rays(u1, v1, F_A, R_A)
    slope = da[:, :2] / da[:, 2:3]  # d(xy)/dz along the agent ray
    oa = C_A[None]
    dw = _rays(u2, v2, F_W, R_W)
    ow = ee + WRIST_OFFSET

    # Closest points between the agent and wrist rays.
    r = oa - ow
    a, b, c = (da * da).sum(1), (da * dw).sum(1), (dw * dw).sum(1)
    d, e = (da * r).sum(1), (dw * r).sum(1)
    den = a * c - b * b
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    t_a, t_w = (b * e - c * d) / den, (a * e - b * d) / den
    pa, pw = oa + da * t_a[:, None], ow + dw * t_w[:, None]
    gap = np.linalg.norm(pa - pw, axis=1)
    tri_ok = has_w & (gap < MAX_RAY_GAP) & (pa[:, 2] > Z_RANGE[0]) & (pa[:, 2] < Z_RANGE[1])
    z_geo = np.where(tri_ok, pa[:, 2], Z_PRIOR)

    plane = C_A[:2] + slope * (Z_PRIOR - C_A[2])
    plane_w = ow[:, :2] + (dw[:, :2] / dw[:, 2:3]) * (Z_PRIOR - ow[:, 2:3])
    no_w = ~has_w
    for arr in (gap, t_w):
        arr[no_w] = -1.0
    pw = np.where(no_w[:, None], 0.0, pw)
    pa_tri = np.where(no_w[:, None], 0.0, pa)
    plane_w = np.where(no_w[:, None], 0.0, plane_w)
    # Distance of the wrist box from the image border: clipped boxes bias the centre.
    border = np.minimum.reduce([X[:, 6] - w2 / 2, X[:, 7] - h2 / 2, 256 - X[:, 6] - w2 / 2, 256 - X[:, 7] - h2 / 2])
    border[no_w] = -1.0
    feats = np.column_stack([
        X, slope, plane, plane - ee[:, :2], z_geo, tri_ok, has_w, gap, t_w, pa_tri, pw, plane_w,
        border, w1 * h1, w1 / np.maximum(h1, 1), h1 - w1, w2 * h2,
        ee[:, 2] - z_geo,
    ])
    return z_geo, slope, feats


def ray_xy(slope, z):
    """Point on the agent ray at height z."""
    return C_A[:2] + slope * (np.asarray(z)[:, None] - C_A[2])
