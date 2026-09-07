"""
Pure-Python fallback for gesture_native. Mirrors the same class/method
names and semantics as native/gesture_engine.cpp so main.py can use either
implementation interchangeably. This exists so the project runs (at lower
frame rates / higher latency) even without a C++ compiler available.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from typing import Dict, List, Tuple

FEATURE_SIZE = 21


class RuleClassifier:
    def __init__(self, pinch_threshold: float = 0.08, thumb_dir_threshold: float = 0.15):
        self.pinch_threshold = pinch_threshold
        self.thumb_dir_threshold = thumb_dir_threshold

    def classify(self, f: List[float]) -> Tuple[str, float]:
        if len(f) < FEATURE_SIZE:
            raise ValueError("feature vector must have 21 values")
        thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = f[0:5]
        pinch_dist = f[19]
        thumb_dir = f[20]
        ext_sum = thumb_ext + index_ext + middle_ext + ring_ext + pinky_ext

        if pinch_dist < self.pinch_threshold:
            conf = 1.0 - (pinch_dist / self.pinch_threshold)
            return "pinch", _clamp01(conf)
        if ext_sum < 0.5:
            return "fist", 1.0
        if ext_sum > 4.5:
            return "open_palm", 1.0

        only_thumb = thumb_ext > 0.5 and index_ext < 0.5 and middle_ext < 0.5 and ring_ext < 0.5 and pinky_ext < 0.5
        if only_thumb:
            if thumb_dir < -self.thumb_dir_threshold:
                return "thumbs_up", _clamp01(-thumb_dir / 0.5)
            if thumb_dir > self.thumb_dir_threshold:
                return "thumbs_down", _clamp01(thumb_dir / 0.5)

        only_index = index_ext > 0.5 and middle_ext < 0.5 and ring_ext < 0.5 and pinky_ext < 0.5
        if only_index:
            return "point", 0.9

        index_middle = index_ext > 0.5 and middle_ext > 0.5 and ring_ext < 0.5 and pinky_ext < 0.5
        if index_middle:
            return "two_fingers", 0.9

        return "unknown", 0.0


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _euclidean(a, b) -> float:
    n = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))


def _cosine_sim(a, b) -> float:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


class KNNClassifier:
    def __init__(self, k: int = 5):
        self.k = k
        self._samples: List[List[float]] = []
        self._labels: List[str] = []

    def add_sample(self, features: List[float], label: str) -> None:
        if len(features) < FEATURE_SIZE:
            raise ValueError("feature vector must have 21 values")
        self._samples.append(list(features))
        self._labels.append(label)

    def clear(self) -> None:
        self._samples.clear()
        self._labels.clear()

    def sample_count(self) -> int:
        return len(self._samples)

    def classify(self, query: List[float]) -> Tuple[str, float]:
        if not self._samples:
            return "unknown", 0.0
        scored = []
        for i, s in enumerate(self._samples):
            dist = _euclidean(query, s) + (1.0 - _cosine_sim(query, s))
            scored.append((dist, i))
        scored.sort(key=lambda t: t[0])
        k = min(self.k, len(scored))
        top = scored[:k]

        # Distance-weighted voting (see gesture_engine.cpp for rationale):
        # avoids a majority class with merely-nearby samples outvoting a
        # minority class that has an exact/near-exact match.
        weight: Counter = Counter()
        best_dist: Dict[str, float] = {}
        total_weight = 0.0
        for dist, idx in top:
            lbl = self._labels[idx]
            w = 1.0 / (1.0 + dist)
            weight[lbl] += w
            total_weight += w
            if lbl not in best_dist or dist < best_dist[lbl]:
                best_dist[lbl] = dist

        best_label, best_weight = weight.most_common(1)[0]
        agreement = (best_weight / total_weight) if total_weight > 1e-9 else 0.0
        closeness = 1.0 / (1.0 + best_dist[best_label])
        confidence = _clamp01(0.5 * agreement + 0.5 * closeness)
        return best_label, confidence

    def samples(self) -> List[List[float]]:
        return list(self._samples)

    def labels(self) -> List[str]:
        return list(self._labels)


class TemporalVoter:
    def __init__(self, window: int = 6, min_agreement: float = 0.6):
        self.window = window
        self.min_agreement = min_agreement
        self._history: deque = deque(maxlen=window)

    def push(self, label: str) -> str:
        self._history.append(label)
        counts = Counter(self._history)
        best_label, best_count = counts.most_common(1)[0]
        agreement = best_count / len(self._history)
        return best_label if agreement >= self.min_agreement else "unknown"

    def reset(self) -> None:
        self._history.clear()


class _OneEuroFilter1D:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._init = False
        self._x_prev = 0.0
        self._dx_prev = 0.0

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, dt: float = -1.0) -> float:
        dt = dt if dt > 0 else (1.0 / self.freq)
        if not self._init:
            self._x_prev = x
            self._dx_prev = 0.0
            self._init = True
            return x
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        edx = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = self._alpha(dt, cutoff)
        ex = a * x + (1 - a) * self._x_prev
        self._x_prev = ex
        self._dx_prev = edx
        return ex

    def reset(self):
        self._init = False


class OneEuroFilter2D:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.02):
        self._fx = _OneEuroFilter1D(freq, min_cutoff, beta)
        self._fy = _OneEuroFilter1D(freq, min_cutoff, beta)

    def filter(self, x: float, y: float) -> Tuple[float, float]:
        return self._fx.filter(x), self._fy.filter(y)

    def reset(self):
        self._fx.reset()
        self._fy.reset()
