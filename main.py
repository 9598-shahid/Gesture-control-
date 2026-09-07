"""AI Gesture Command Center - entry point.

    python main.py --profile default [--camera 0] [--no-native]

Keys while running:
  q          quit
  c          toggle calibration mode
  1-9        (in calibration mode) record current pose as sample for
             gesture slot "custom_<digit>"
  s          save current profile to disk
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json

import cv2
import pyautogui

from feature_extraction import extract_features, palm_center
from hand_tracker import HandTracker
from action_dispatcher import ActionDispatcher
from profile_manager import ProfileManager
from swipe_detector import SwipeDetector

# --- pick native (C++) engine if built, else pure-Python fallback --------
NATIVE_AVAILABLE = False
try:
    import gesture_native as engine  # type: ignore
    NATIVE_AVAILABLE = True
except ImportError:
    import gesture_classifier_fallback as engine  # type: ignore

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(THIS_DIR, "..", "config")


def load_config() -> dict:
    with open(os.path.join(CONFIG_DIR, "gestures.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Gesture Command Center")
    parser.add_argument("--profile", default="default", help="user profile name")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--no-native", action="store_true",
                         help="force the pure-Python fallback even if the C++ extension is built")
    parser.add_argument("--headless", action="store_true",
                         help="run without an OpenCV preview window (still Ctrl+C to quit)")
    args = parser.parse_args()

    global engine, NATIVE_AVAILABLE
    if args.no_native:
        import gesture_classifier_fallback as engine  # noqa: F811
        NATIVE_AVAILABLE = False

    print(f"[engine] using {'C++ native' if NATIVE_AVAILABLE else 'pure-Python fallback'} classifier")

    config = load_config()
    conf_threshold = config.get("confidence_threshold", 0.65)

    profiles = ProfileManager(os.path.join(CONFIG_DIR, "profiles"))
    profile_data = profiles.load(args.profile)

    rule_clf = engine.RuleClassifier()
    knn_clf = engine.KNNClassifier(k=5)
    n_loaded = ProfileManager.hydrate_knn(knn_clf, profile_data)
    print(f"[profile] '{args.profile}': {n_loaded} custom samples loaded")

    voter = engine.TemporalVoter(
        window=config.get("temporal_window", 6),
        min_agreement=config.get("temporal_min_agreement", 0.6),
    )

    cursor_cfg = config.get("cursor", {})
    cursor_filter = engine.OneEuroFilter2D(
        freq=30.0,
        min_cutoff=cursor_cfg.get("smoothing_min_cutoff", 1.0),
        beta=cursor_cfg.get("smoothing_beta", 0.02),
    )

    dispatcher = ActionDispatcher(config)
    swipe_detector = SwipeDetector()
    tracker = HandTracker(max_hands=1)

    screen_w, screen_h = pyautogui.size()
    margin = cursor_cfg.get("screen_margin", 0.1)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {args.camera}", file=sys.stderr)
        return 1

    calibration_mode = False
    pending_samples: list = []  # (features, label) collected this run, unsaved
    is_dragging = False
    palm_open_since = None
    EMERGENCY_HOLD_SECONDS = 1.0

    print("Running. Press 'q' to quit, 'c' to toggle calibration mode, 's' to save profile.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: failed to read frame from camera", file=sys.stderr)
                break
            frame = cv2.flip(frame, 1)  # mirror for intuitive control
            hands, mp_result = tracker.process(frame)

            label = "none"
            confidence = 0.0

            if hands:
                landmarks = hands[0]
                features = extract_features(landmarks)

                # Rule-based classifier first (built-ins, always available).
                rule_label, rule_conf = rule_clf.classify(features)

                # Custom KNN only consulted when the rule engine is unsure
                # and the user has trained samples, so custom poses can
                # override/extend the default set.
                if rule_label == "unknown" and knn_clf.sample_count() > 0:
                    knn_label, knn_conf = knn_clf.classify(features)
                    label, confidence = knn_label, knn_conf
                else:
                    label, confidence = rule_label, rule_conf

                stable_label = voter.push(label if confidence >= conf_threshold else "unknown")

                # --- swipe (motion-based, independent of pose voting) ---
                px, py = palm_center(landmarks)
                swipe = swipe_detector.update(px)
                if swipe:
                    dispatcher.dispatch(swipe, 1.0)

                # --- calibration: record a sample instead of dispatching ---
                if calibration_mode:
                    cv2.putText(frame, "CALIBRATION MODE - press 1-9 to record",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    if stable_label == "open_palm":
                        palm_open_since = palm_open_since or time.time()
                        if time.time() - palm_open_since >= EMERGENCY_HOLD_SECONDS:
                            if not dispatcher.emergency_stop:
                                dispatcher.toggle_emergency_stop()
                                print("[safety] EMERGENCY STOP engaged")
                    else:
                        if palm_open_since is not None and dispatcher.emergency_stop:
                            dispatcher.toggle_emergency_stop()
                            print("[safety] emergency stop released")
                        palm_open_since = None

                    if stable_label == "point":
                        screen_x = int(_lerp_margin(px, margin) * screen_w)
                        screen_y = int(_lerp_margin(py, margin) * screen_h)
                        fx, fy = cursor_filter.filter(screen_x, screen_y)
                        dispatcher.move_mouse(int(fx), int(fy))
                        if is_dragging:
                            dispatcher.mouse_up()
                            is_dragging = False
                    elif stable_label == "pinch":
                        if not is_dragging:
                            dispatcher.mouse_down()
                            is_dragging = True
                        screen_x = int(_lerp_margin(px, margin) * screen_w)
                        screen_y = int(_lerp_margin(py, margin) * screen_h)
                        fx, fy = cursor_filter.filter(screen_x, screen_y)
                        dispatcher.move_mouse(int(fx), int(fy))
                    else:
                        if is_dragging:
                            dispatcher.mouse_up()
                            is_dragging = False

                    if stable_label == "two_fingers":
                        dispatcher.scroll(-40)
                    elif stable_label in ("fist", "thumbs_up", "thumbs_down"):
                        dispatcher.dispatch(stable_label, confidence)
                    elif stable_label.startswith("custom_"):
                        dispatcher.dispatch(stable_label, confidence)

                if not args.headless:
                    tracker.draw(frame, mp_result)
                    status = f"{stable_label} ({confidence:.2f})"
                    if dispatcher.emergency_stop:
                        status += " | STOPPED"
                    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 200, 0), 2)
            else:
                voter.push("none")
                swipe_detector.reset()

            if not args.headless:
                cv2.imshow("AI Gesture Command Center", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    calibration_mode = not calibration_mode
                    print(f"[calibration] {'ON' if calibration_mode else 'OFF'}")
                elif key == ord("s"):
                    all_samples = list(zip(knn_clf.samples(), knn_clf.labels())) + pending_samples
                    profiles.save(args.profile, all_samples)
                    print(f"[profile] saved {len(all_samples)} samples to '{args.profile}'")
                elif calibration_mode and hands and ord("1") <= key <= ord("9"):
                    digit = chr(key)
                    slot_label = f"custom_{digit}"
                    knn_clf.add_sample(features, slot_label)
                    pending_samples.append((features, slot_label))
                    print(f"[calibration] recorded sample for '{slot_label}' "
                          f"(total now {knn_clf.sample_count()})")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        tracker.close()

    return 0


def _lerp_margin(v: float, margin: float) -> float:
    """Maps a normalized coordinate in [margin, 1-margin] to [0,1], so the
    user doesn't have to reach screen edges of the camera frame to reach
    screen edges on the monitor."""
    lo, hi = margin, 1.0 - margin
    v = max(lo, min(hi, v))
    return (v - lo) / (hi - lo)


if __name__ == "__main__":
    raise SystemExit(main())
