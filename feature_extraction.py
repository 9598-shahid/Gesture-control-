"""
Converts MediaPipe's 21 hand landmarks into the fixed 21-float feature
vector consumed by both the C++ `gesture_native` classifiers and the pure
Python fallback. Keeping this logic in one place guarantees the two
classification paths agree on what a "feature vector" means.

MediaPipe hand landmark indices (see MediaPipe Hands docs):
  0 WRIST
  1-4   THUMB  (CMC, MCP, IP, TIP)
  5-8   INDEX  (MCP, PIP, DIP, TIP)
  9-12  MIDDLE (MCP, PIP, DIP, TIP)
  13-16 RING   (MCP, PIP, DIP, TIP)
  17-20 PINKY  (MCP, PIP, DIP, TIP)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

WRIST = 0
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]   # thumb uses IP joint as its "PIP" analogue
FINGER_MCPS = [2, 5, 9, 13, 17]
FEATURE_SIZE = 21


@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0


def _dist(a: Point, b: Point) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _angle(a: Point, b: Point, c: Point) -> float:
    """Angle at point b, formed by segments b->a and b->c, in radians."""
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.acos(cos_a)


def _palm_size(landmarks: Sequence[Point]) -> float:
    """Wrist-to-middle-MCP distance, used as a stable normalization scale
    (roughly constant regardless of hand rotation, unlike bounding-box size).
    """
    size = _dist(landmarks[WRIST], landmarks[9])
    return size if size > 1e-6 else 1e-6


def _finger_extended(landmarks: Sequence[Point], tip_idx: int, pip_idx: int,
                      mcp_idx: int, wrist_idx: int = WRIST) -> float:
    """1.0 if the finger is extended, 0.0 if folded.

    Heuristic: a finger is extended when the tip is farther from the wrist
    than the pip/mcp joints are, by a reasonable margin. Works regardless of
    hand orientation (unlike a fixed y-axis comparison).
    """
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]
    wrist = landmarks[wrist_idx]
    d_tip = _dist(tip, wrist)
    d_pip = _dist(pip, wrist)
    d_mcp = _dist(mcp, wrist)
    scale = _palm_size(landmarks)
    return 1.0 if (d_tip > d_pip and d_tip > d_mcp + 0.05 * scale) else 0.0


def extract_features(landmarks: Sequence[Point]) -> List[float]:
    """landmarks: sequence of 21 Point objects (normalized image coords,
    x/y in [0,1] as returned by MediaPipe). Returns a 21-float feature
    vector; see gesture_engine.cpp header comment for the layout.
    """
    if len(landmarks) != 21:
        raise ValueError(f"expected 21 landmarks, got {len(landmarks)}")

    scale = _palm_size(landmarks)
    wrist = landmarks[WRIST]

    # [0-4] extension flags
    ext_flags = [
        _finger_extended(landmarks, FINGER_TIPS[i], FINGER_PIPS[i], FINGER_MCPS[i])
        for i in range(5)
    ]

    # [5-8] inter-fingertip distances (adjacent fingers), normalized
    tip_pts = [landmarks[i] for i in FINGER_TIPS]
    inter_tip = [_dist(tip_pts[i], tip_pts[i + 1]) / scale for i in range(4)]

    # [9-13] wrist -> fingertip distances, normalized
    wrist_tip = [_dist(wrist, tip_pts[i]) / scale for i in range(5)]

    # [14-18] PIP bend angle per finger, normalized to [0,1] (0=straight, 1=fully bent)
    bend_angles = []
    for i in range(5):
        mcp = landmarks[FINGER_MCPS[i]]
        pip = landmarks[FINGER_PIPS[i]]
        tip = landmarks[FINGER_TIPS[i]]
        ang = _angle(mcp, pip, tip)  # pi (straight) .. 0 (fully bent)
        normalized = 1.0 - (ang / math.pi)
        bend_angles.append(max(0.0, min(1.0, normalized)))

    # [19] pinch distance thumb tip <-> index tip, normalized
    pinch_dist = _dist(landmarks[4], landmarks[8]) / scale

    # [20] thumb vertical direction relative to wrist, normalized by scale
    thumb_dir = (landmarks[4].y - wrist.y) / scale

    features = ext_flags + inter_tip + wrist_tip + bend_angles + [pinch_dist, thumb_dir]
    assert len(features) == FEATURE_SIZE
    return [float(f) for f in features]


def palm_center(landmarks: Sequence[Point]) -> Tuple[float, float]:
    """Average of wrist + finger MCPs — a stable point to drive cursor
    movement and swipe-trajectory tracking."""
    pts = [landmarks[WRIST]] + [landmarks[i] for i in FINGER_MCPS]
    x = sum(p.x for p in pts) / len(pts)
    y = sum(p.y for p in pts) / len(pts)
    return x, y
