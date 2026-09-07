# Gesture Command Center — Mobile

A **mobile web app** (no app store, no install) that runs entirely in your
phone's browser: it uses your phone's camera + on-device MediaPipe Hands to
recognize gestures, then routes each recognized gesture to one of two
**targets**, switchable live with the toggle in the UI:

- **📱 This phone** — real but limited actions reachable from a web page
  (play/pause + seek on the page's own media, volume of that media, haptic
  feedback, fullscreen, keep-awake). See "What a phone browser can't do" below.
- **💻 My laptop** — the phone relays recognized gestures over your local
  Wi-Fi to a small Python server running on your laptop, which uses the
  *same* `ActionDispatcher` (PyAutoGUI) as the desktop webcam app. This gets
  you the full original feature set (media keys, volume, app-switch, custom
  shell commands, MQTT/smart-home) — your phone becomes a gesture remote.

```
Phone browser                              Laptop (same Wi-Fi)
┌─────────────────────────┐                ┌───────────────────────────┐
│ camera → MediaPipe Hands │   WebSocket    │ ws_server.py               │
│ → feature vector (JS)    │ ─────────────▶ │  → ActionDispatcher        │
│ → classify (JS)          │  {label, conf} │    (PyAutoGUI, real OS     │
│ → target: phone or laptop│                │     control, MQTT, etc.)   │
└─────────────────────────┘                └───────────────────────────┘
```

## Why JavaScript here instead of C++

The desktop app uses Python + a compiled C++ extension because it controls
its own always-on process. A phone browser can't load a native `.so`/`.dll`
— its "compiled, efficient language" equivalent is **WebAssembly**. The
classification logic (`gesture_classifier.js`, `feature_extraction.js`) is
written as a direct line-for-line port of the same Python/C++ logic, and is
light enough in practice to run every frame in plain JS. If you want to push
further, `native/gesture_engine.cpp` can be compiled to WASM with Emscripten
(`emcc --bind gesture_engine.cpp -o gesture_engine.wasm ...`) and swapped in
as a drop-in replacement for `gesture_classifier.js` — not included here since
it needs the Emscripten SDK, but the C++ source is already shared and doesn't
need changes.

## What a phone browser can't do (be aware of this before you start)

Web pages are sandboxed away from the OS for security. From a mobile browser
you **cannot**: move a system cursor, change system-wide volume, switch
between other installed apps, or send arbitrary keystrokes to other apps —
that's exactly why the "💻 My laptop" target exists, and is the more useful
mode for most of the original feature set (media control, volume, app
switching). The "📱 This phone" target is real but intentionally scoped to
what the web platform actually allows.

## Setup

### 1. Serve the web app over HTTPS

Phones require a secure context (https, or localhost) before they'll grant
camera access — plain `http://<ip>` from another device will silently fail.
A small self-signed-cert server is included:

```bash
cd gesture_control/mobile/server
python serve_https.py --port 8443
```

First run auto-generates a local certificate (needs `openssl`, already on
macOS/Linux; on Windows install via `winget install OpenSSL` or use WSL).

### 2. Open it on your phone

Find your laptop's local IP (macOS: `ipconfig getifaddr en0`; Linux:
`hostname -I`; Windows: `ipconfig` → IPv4 Address), then on your phone
(same Wi-Fi) visit:

```
https://<laptop-ip>:8443
```

Your browser will warn about the self-signed certificate — this is expected
for a locally generated cert. Tap through ("Advanced → Proceed" on
Chrome/Android, "visit this website" on Safari/iOS). Grant camera access
when prompted.

### 3. (Only for the "💻 My laptop" target) start the relay server

In a separate terminal on your laptop:

```bash
cd gesture_control
pip install -r requirements.txt      # includes `websockets`
cd mobile/server
python ws_server.py --port 8765
```

In the phone UI, tap the target toggle to switch to "💻 My laptop", type in
`<laptop-ip>:8765`, and tap Connect. `ws-status` should read "connected".

## Using it

- Same built-in gesture set as desktop: ✊ fist, 🖐️ open palm (hold 1s =
  emergency stop, on either target), 👍/👎 thumbs, ✌️ two fingers, 🤏 pinch,
  ☝️ point, 👋 fast palm swipe.
- **Calibrate** button + the dropdown lets you record custom poses
  (`custom_1`-`custom_4`) into a KNN classifier, same idea as desktop —
  samples are saved to the phone's `localStorage`, not a server, so they're
  per-device.
- **Save** persists your custom samples so they survive a page reload.
- Note: continuous gestures (point → cursor movement, pinch → drag,
  two-fingers → scroll) aren't relayed to the laptop in this version — only
  discrete, one-shot gestures are (fist, thumbs, swipe, custom slots,
  emergency stop). Streaming live cursor coordinates over the WebSocket
  would be a natural next step if you want phone-driven mouse control too.

## Files

```
mobile/
├── web/
│   ├── index.html               # camera view + touch-friendly HUD
│   ├── style.css
│   ├── feature_extraction.js    # JS port of python/feature_extraction.py
│   ├── gesture_classifier.js    # JS port of the rule/KNN/voter/swipe logic
│   ├── phone_actions.js         # what's actually reachable via Web APIs
│   ├── ws_client.js             # WebSocket client -> laptop
│   └── app.js                   # wiring: camera, MediaPipe, routing, UI
└── server/
    ├── serve_https.py           # local HTTPS static server (self-signed cert)
    └── ws_server.py             # laptop-side relay, reuses ActionDispatcher
```
