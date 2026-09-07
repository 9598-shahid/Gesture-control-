"""Thin wrapper around MediaPipe Hands: converts each frame into a list of
per-hand landmark points (in our own lightweight Point type, decoupled from
MediaPipe's protobuf types) plus drawing helpers for the debug overlay.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import mediapipe as mp

from feature_extraction import Point

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


class HandTracker:
    def __init__(self, max_hands: int = 1, detection_conf: float = 0.7,
                 tracking_conf: float = 0.6):
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def process(self, frame_bgr) -> Tuple[List[List[Point]], object]:
        """Returns (list_of_hand_landmarks, raw_mediapipe_result).
        Each hand's landmarks is a list of 21 Point(x, y, z) in normalized
        [0,1] image coordinates.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self._hands.process(frame_rgb)

        hands: List[List[Point]] = []
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                pts = [Point(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
                hands.append(pts)
        return hands, result

    def draw(self, frame_bgr, result) -> None:
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

    def close(self):
        self._hands.close()
