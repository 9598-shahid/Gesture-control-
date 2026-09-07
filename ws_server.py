"""
Run this ON YOUR LAPTOP. It reuses the exact same ActionDispatcher and
config/gestures.json as the desktop webcam app (../../python/action_dispatcher.py)
so gesture->action bindings stay identical whether the gesture came from the
laptop's own webcam or was relayed from the phone.

    python ws_server.py [--host 0.0.0.0] [--port 8765] [--profile default]

The phone connects to ws://<this-laptop-ip>:8765 and sends JSON messages:
    {"label": "fist", "confidence": 0.92, "target": "laptop"}
    {"label": "emergency_stop_on", "confidence": 1.0}
    {"label": "emergency_stop_off", "confidence": 1.0}

Find your laptop's local IP with `ipconfig getifaddr en0` (macOS Wi-Fi),
`hostname -I` (Linux), or `ipconfig` (Windows, look for IPv4 Address).
Phone and laptop must be on the same Wi-Fi network.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# reuse the desktop app's modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from action_dispatcher import ActionDispatcher  # noqa: E402

try:
    import websockets
except ImportError:
    print("ERROR: install the 'websockets' package first: pip install websockets",
          file=sys.stderr)
    raise

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")


def load_config() -> dict:
    with open(os.path.join(CONFIG_DIR, "gestures.json"), "r", encoding="utf-8") as f:
        return json.load(f)


async def handle_client(websocket, dispatcher: ActionDispatcher):
    peer = websocket.remote_address
    print(f"[connect] phone connected from {peer}")
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            label = msg.get("label")
            confidence = float(msg.get("confidence", 0.0))
            if not label:
                continue

            if label == "emergency_stop_on":
                dispatcher.emergency_stop = True
                print("[safety] EMERGENCY STOP engaged (via phone)")
                continue
            if label == "emergency_stop_off":
                dispatcher.emergency_stop = False
                print("[safety] emergency stop released (via phone)")
                continue

            fired = dispatcher.dispatch(label, confidence)
            if fired:
                print(f"[action] {label} ({confidence:.2f}) -> dispatched")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print(f"[disconnect] {peer}")


async def main_async(host: str, port: int) -> None:
    config = load_config()
    dispatcher = ActionDispatcher(config)

    async def handler(websocket):
        await handle_client(websocket, dispatcher)

    print(f"[server] listening on ws://{host}:{port}")
    print("[server] on your phone, open the web app and set the laptop address to")
    print(f"          <this-machine-local-ip>:{port}")
    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # run forever


def main() -> int:
    parser = argparse.ArgumentParser(description="Gesture relay server (phone -> laptop actions)")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (0.0.0.0 = all interfaces)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args.host, args.port))
    except KeyboardInterrupt:
        print("\n[server] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
