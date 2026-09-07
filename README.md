# 🖐️ AI Gesture Command Center

**Control your computer, browser, or smart home with hand gestures — no extra hardware, just a webcam or your phone's camera.**

A real-time hand-gesture control system with two parallel implementations:
- a **desktop app** (Python + C++) that watches a webcam and drives your OS directly, and
- a **mobile web app** (JavaScript, no install) that uses your phone's camera and either acts locally or remote-controls the same desktop app over Wi-Fi.

Both share the same gesture vocabulary, the same feature-extraction math, and the same classification logic, ported 1:1 across languages so behavior stays consistent everywhere.

---

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Why two languages](#why-two-languages)
- [Project structure](#project-structure)
- [Getting started (desktop)](#getting-started-desktop)
- [Getting started (mobile)](#getting-started-mobile)
- [Default gestures](#default-gestures)
- [Custom gestures (calibration)](#custom-gestures-calibration)
- [Configuration](#configuration)
- [Smart-home / MQTT](#smart-home--mqtt)
- [Troubleshooting](#troubleshooting)
- [Roadmap ideas](#roadmap-ideas)
- [Safety](#safety)
- [License](#license)

---

## Features

- ✋ Real-time hand-gesture recognition from a webcam or phone camera
- 🖱️ Point to move the mouse, pinch to click and drag, two fingers to scroll
- ✊ / 👍 / 👎 fist and thumb gestures for media play/pause and volume
- 👋 Fast palm swipes for app switching
- 🎯 **Custom gesture training** — no code changes needed, just demonstrate a pose a few times
- 🔐 Per-user gesture profiles, persisted to disk (desktop) or the browser (mobile)
- 🖐️ Open-palm **emergency stop**, held for 1 second, on both platforms
- 📱 Mobile web app — control your phone or relay gestures to your laptop, no app store
- 🏠 Optional MQTT publishing for Home Assistant / Zigbee2MQTT-style smart-home hooks
- ⚙️ Fully declarative gesture → action mapping via a single JSON config file

## How it works

```
Webcam / phone camera
        │
        ▼
MediaPipe Hands  (21 hand landmarks, x/y/z per frame)
        │
        ▼
Feature extraction (21-value vector: finger-extension flags,
inter-fingertip distances, bend angles, pinch distance, thumb direction)
        │
        ▼
Classification
  ├─ Rule-based classifier   → the 8 built-in gestures, no training needed
  └─ KNN classifier          → your custom trained gestures
        │
        ▼
Temporal voting  (stabilizes the label across a sliding window of frames)
        │
        ▼
Action dispatch  (PyAutoGUI / media keys / shell / MQTT — or, on mobile,
                   Web APIs locally, or a WebSocket relay to the desktop app)
```

The same 21-value feature vector and the same classifier logic (rules, KNN,
temporal voting, swipe detection) exist in three parallel implementations —
C++, Python, and JavaScript — so a gesture trained on your laptop looks the
same to the math as one trained on your phone.

## Why two languages

| Layer | Language | Why |
|---|---|---|
| Camera I/O, MediaPipe glue, config, profiles, OS action dispatch | **Python** | Rich CV/ML ecosystem, fast to read and extend |
| Per-frame classification, KNN search, cursor smoothing, temporal voting | **C++** (via pybind11) | Runs 30-60×/second in the hot path; keeps latency low and scales the custom-gesture library without dropping frames |
| Browser-side camera loop and classification | **JavaScript** | The only language a mobile browser can execute; ported line-for-line from the Python/C++ logic (a WebAssembly build of the same C++ source is a drop-in upgrade path — see `mobile/README.md`) |

If the C++ extension isn't built, the app **automatically falls back** to a
pure-Python implementation of the identical logic — slower, but functionally
the same, so the project runs even without a C++ compiler on hand.

## Project structure

```
gesture_control/
├── README.md                    ← you are here
├── requirements.txt
├── native/                       # C++ hot-loop extension (pybind11)
│   ├── gesture_engine.cpp        #   RuleClassifier, KNNClassifier, TemporalVoter, OneEuroFilter
│   └── setup.py                  #   pip install . to build gesture_native
├── python/                       # Desktop app
│   ├── main.py                   #   entry point / camera loop
│   ├── hand_tracker.py           #   MediaPipe Hands wrapper
│   ├── feature_extraction.py     #   landmarks -> 21-float feature vector
│   ├── gesture_classifier_fallback.py  # pure-Python mirror of gesture_engine.cpp
│   ├── action_dispatcher.py      #   gesture -> PyAutoGUI / shell / MQTT
│   ├── profile_manager.py        #   load/save per-user KNN training samples
│   └── swipe_detector.py         #   motion-based swipe (not a static pose)
├── mobile/                       # Phone-friendly web app
│   ├── README.md                 #   mobile-specific setup & limitations
│   ├── web/                      #   camera page: MediaPipe Hands (JS) + classifier
│   └── server/                   #   HTTPS static server + WebSocket relay to desktop
└── config/
    ├── gestures.json             # gesture -> action mapping (edit this to customize)
    └── profiles/default.json     # per-user KNN training samples
```

## Getting started (desktop)

### Requirements
- Python 3.9+
- A webcam
- (Optional but recommended) a C++17 compiler — `clang` (macOS, via Xcode Command Line Tools), `gcc`/`g++` (Linux), or MSVC Build Tools (Windows) — for the fast native classifier

### Install

```bash
git clone <this-repo-url>
cd gesture_control
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Build the native extension (optional)

```bash
cd native
pip install .
cd ..
```

If this step fails or is skipped, `main.py` automatically uses the
pure-Python fallback — no action needed on your part.

### Run

```bash
cd python
python main.py --profile default
```

| Key | Action |
|---|---|
| `q` | quit |
| `c` | toggle calibration mode |
| `1`-`9` | (in calibration mode) record the current pose as a custom gesture sample |
| `s` | save the current profile to disk |

## Getting started (mobile)

The mobile app is a browser page — nothing to install. Full instructions,
including the two control targets ("this phone" vs "my laptop") and the
HTTPS requirement for camera access, are in **[`mobile/README.md`](mobile/README.md)**.

Quick version:

```bash
# on your laptop
cd mobile/server
python serve_https.py --port 8443
# optionally, for laptop-remote control:
python ws_server.py --port 8765
```

Then, on your phone (same Wi-Fi), open `https://<laptop-ip>:8443`, accept
the self-signed certificate warning, and grant camera access.

## Default gestures

No training required — these work immediately on both desktop and mobile:

| Gesture | Desktop action | Mobile ("this phone") action |
|---|---|---|
| ☝️ index finger only | move mouse cursor | — (see mobile README) |
| 🤏 pinch (thumb + index) | left click / drag while held | — |
| ✌️ index + middle | scroll | — |
| ✊ fist | pause / play media | pause / play the page's media |
| 👍 thumb up | volume up | page media volume up |
| 👎 thumb down | volume down | page media volume down |
| 🖐️ open palm, held 1s | toggle emergency stop | toggle emergency stop |
| 👋 fast horizontal swipe | switch application (alt+tab) | seek the page's media ±10s |

## Custom gestures (calibration)

1. Enter calibration mode (`c` on desktop, the **Calibrate** button on mobile).
2. Hold a pose and record it under a slot (`1`-`9` on desktop, a dropdown on
   mobile) — repeat a few times from slightly different angles for a more
   robust match.
3. Save the profile.
4. Bind the resulting label (`custom_1`, `custom_2`, …) to any action in
   `config/gestures.json` — a keyboard shortcut, a shell command, or an MQTT
   publish for a smart-home hook.

New poses are matched with a distance-weighted K-nearest-neighbor search
over your recorded samples, so you can teach the system gestures it's never
seen without touching any code.

## Configuration

All gesture → action bindings live in `config/gestures.json`:

```json
{
  "mappings": {
    "fist": { "type": "media_key", "key": "playpause", "cooldown": 0.8 },
    "thumbs_up": { "type": "hotkey", "keys": ["volumeup"], "cooldown": 0.25 },
    "custom_1": { "type": "shell", "command": "echo hello", "cooldown": 2.0 }
  },
  "confidence_threshold": 0.65,
  "temporal_window": 6
}
```

Supported action `type`s: `media_key`, `hotkey`, `shell`, `mouse_click`,
`scroll`, `mqtt`. Each mapping has its own `cooldown` (seconds) so repeatable
actions (volume) and one-shot actions (play/pause) feel natural.

## Smart-home / MQTT

Set `"mqtt": {"enabled": true, "broker": "...", "topic": "..."}` in
`config/gestures.json` and bind a gesture to `{"type": "mqtt"}` to publish
`{"gesture": ..., "confidence": ...}` events any MQTT-capable hub (Home
Assistant, Zigbee2MQTT, etc.) can subscribe to. Requires `paho-mqtt`
(already in `requirements.txt`) and a running broker.

## Troubleshooting

<details>
<summary>Gestures aren't moving the mouse / triggering actions (desktop)</summary>

- Check the on-screen confidence percentage — if it's consistently below
  `confidence_threshold` in `config/gestures.json`, try recalibrating or
  improving lighting.
- On macOS, grant your terminal app **Accessibility** permission
  (System Settings → Privacy & Security → Accessibility).
- Make sure you're not in calibration mode (`c` toggles it).
</details>

<details>
<summary>Camera won't open on my phone</summary>

- You must load the page over **HTTPS** (or `localhost`) — plain `http://<ip>`
  from another device will silently block camera access. Use
  `mobile/server/serve_https.py`.
- Accept the self-signed certificate warning on first load.
</details>

<details>
<summary>The C++ extension won't build</summary>

- Confirm a C++17 compiler is installed and on your `PATH`.
- Windows needs "Desktop development with C++" from Visual Studio Build
  Tools.
- This is optional — the app runs fine on the pure-Python fallback, just
  with a bit more latency.
</details>

## Roadmap ideas

- WebAssembly build of `gesture_engine.cpp` for the mobile app (source is
  already shared; needs an Emscripten build step)
- Streaming continuous gestures (point/pinch/scroll) from phone to laptop,
  not just discrete ones
- Two-hand gesture combinations
- A small GUI for editing `config/gestures.json` instead of hand-editing JSON

## Safety

- An open palm held for 1 second triggers an **emergency stop** that
  disables all dispatch until the pose is released — a deliberate circuit
  breaker against misclassification.
- PyAutoGUI's fail-safe is enabled: dragging the cursor into a screen corner
  aborts control immediately.

## License

MIT — see `LICENSE` (add one if you plan to publish this repo).
