"""Executes the OS-level side effects for a recognized, stabilized gesture.

Every action type has an independent cooldown so e.g. "volume up" (repeatable)
and "pause/play" (single-shot) feel natural rather than firing every frame
a pose is held.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import pyautogui

pyautogui.FAILSAFE = True  # moving mouse to a screen corner aborts, as a safety net

try:
    import paho.mqtt.publish as mqtt_publish
    _MQTT_AVAILABLE = True
except ImportError:
    _MQTT_AVAILABLE = False


class ActionDispatcher:
    def __init__(self, gesture_config: dict):
        """gesture_config: parsed config/gestures.json, e.g.
        {
          "mappings": {
            "fist": {"type": "media_key", "key": "playpause", "cooldown": 0.8},
            "thumbs_up": {"type": "hotkey", "keys": ["volumeup"], "cooldown": 0.25},
            "custom_1": {"type": "shell", "command": "notify-send Hi", "cooldown": 2.0}
          },
          "mqtt": {"enabled": false, "broker": "localhost", "port": 1883, "topic": "gesture/events"}
        }
        """
        self.mappings: Dict[str, dict] = gesture_config.get("mappings", {})
        self.mqtt_cfg: dict = gesture_config.get("mqtt", {"enabled": False})
        self._last_fired: Dict[str, float] = {}
        self.emergency_stop = False

        self._handlers: Dict[str, Callable[[dict], None]] = {
            "media_key": self._handle_media_key,
            "hotkey": self._handle_hotkey,
            "shell": self._handle_shell,
            "mouse_click": self._handle_mouse_click,
            "scroll": self._handle_scroll,
            "mqtt": self._handle_mqtt,
        }

    # -- public API ------------------------------------------------------
    def dispatch(self, label: str, confidence: float, extra: Optional[dict] = None) -> bool:
        """Fires the action bound to `label`, respecting cooldown and the
        emergency-stop flag. Returns True if an action actually fired."""
        if self.emergency_stop:
            return False
        cfg = self.mappings.get(label)
        if not cfg:
            return False

        now = time.time()
        cooldown = cfg.get("cooldown", 0.5)
        last = self._last_fired.get(label, 0.0)
        if now - last < cooldown:
            return False

        action_type = cfg.get("type")
        handler = self._handlers.get(action_type)
        if handler is None:
            return False

        merged = dict(cfg)
        if extra:
            merged.update(extra)
        handler(merged)
        self._last_fired[label] = now
        return True

    def move_mouse(self, screen_x: int, screen_y: int) -> None:
        if self.emergency_stop:
            return
        pyautogui.moveTo(screen_x, screen_y, _pause=False)

    def mouse_down(self) -> None:
        if not self.emergency_stop:
            pyautogui.mouseDown(_pause=False)

    def mouse_up(self) -> None:
        pyautogui.mouseUp(_pause=False)  # always allow releasing, even mid-stop

    def scroll(self, amount: int) -> None:
        if not self.emergency_stop:
            pyautogui.scroll(amount, _pause=False)

    def toggle_emergency_stop(self) -> None:
        self.emergency_stop = not self.emergency_stop
        if self.emergency_stop:
            try:
                pyautogui.mouseUp(_pause=False)
            except Exception:
                pass

    # -- action handlers ---------------------------------------------------
    def _handle_media_key(self, cfg: dict) -> None:
        key = cfg.get("key", "playpause")
        pyautogui.press(key)

    def _handle_hotkey(self, cfg: dict) -> None:
        keys = cfg.get("keys", [])
        if keys:
            pyautogui.hotkey(*keys)

    def _handle_shell(self, cfg: dict) -> None:
        import subprocess
        command = cfg.get("command")
        if command:
            subprocess.Popen(command, shell=True)

    def _handle_mouse_click(self, cfg: dict) -> None:
        button = cfg.get("button", "left")
        pyautogui.click(button=button, _pause=False)

    def _handle_scroll(self, cfg: dict) -> None:
        amount = cfg.get("amount", -100)
        pyautogui.scroll(amount, _pause=False)

    def _handle_mqtt(self, cfg: dict) -> None:
        if not (_MQTT_AVAILABLE and self.mqtt_cfg.get("enabled")):
            return
        import json
        payload = json.dumps({
            "gesture": cfg.get("label", cfg.get("type")),
            "confidence": cfg.get("confidence", 0),
        })
        mqtt_publish.single(
            self.mqtt_cfg.get("topic", "gesture/events"),
            payload=payload,
            hostname=self.mqtt_cfg.get("broker", "localhost"),
            port=self.mqtt_cfg.get("port", 1883),
        )
