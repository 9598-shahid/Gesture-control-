"""Swipe is a *motion* gesture, not a static hand pose, so it can't be
recognized from a single frame's feature vector the way pinch/fist/etc are.
Instead we track the palm-center x-position over a short time window and
fire when its velocity exceeds a threshold, then require a short refractory
period before firing again.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional


class SwipeDetector:
    def __init__(self, window_seconds: float = 0.35, velocity_threshold: float = 1.4,
                 cooldown: float = 0.8):
        """velocity_threshold is in normalized-x-units/second (palm x is in
        [0,1] across the frame width, so ~1.4 means crossing most of the
        frame in ~0.35s)."""
        self.window_seconds = window_seconds
        self.velocity_threshold = velocity_threshold
        self.cooldown = cooldown
        self._history = deque()  # (timestamp, x)
        self._last_fired = 0.0

    def update(self, palm_x: float) -> Optional[str]:
        now = time.time()
        self._history.append((now, palm_x))
        while self._history and now - self._history[0][0] > self.window_seconds:
            self._history.popleft()

        if len(self._history) < 2:
            return None
        if now - self._last_fired < self.cooldown:
            return None

        t0, x0 = self._history[0]
        t1, x1 = self._history[-1]
        dt = t1 - t0
        if dt <= 0:
            return None
        velocity = (x1 - x0) / dt

        if velocity > self.velocity_threshold:
            self._last_fired = now
            self._history.clear()
            return "swipe_right"
        if velocity < -self.velocity_threshold:
            self._last_fired = now
            self._history.clear()
            return "swipe_left"
        return None

    def reset(self) -> None:
        self._history.clear()
