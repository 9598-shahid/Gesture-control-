"""
Serves mobile/web/ over HTTPS on your local network, with an
auto-generated self-signed certificate. Phones need https (or localhost)
to grant camera access to a web page — plain http won't work once you're
loading the page from another device over Wi-Fi.

    python serve_https.py [--port 8443]

Then on your phone (same Wi-Fi), visit:
    https://<this-laptop-local-ip>:8443

The phone's browser will warn about the self-signed certificate the first
time — this is expected for a locally-generated cert; tap
"Advanced" -> "Proceed" (Chrome/Android) or "visit this website" (Safari/iOS).
"""
from __future__ import annotations

import argparse
import http.server
import os
import ssl
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "..", "web")
CERT_PATH = os.path.join(HERE, "cert.pem")
KEY_PATH = os.path.join(HERE, "key.pem")


def ensure_cert() -> None:
    if os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH):
        return
    print("[cert] generating a self-signed certificate (first run only)...")
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", KEY_PATH, "-out", CERT_PATH,
                "-days", "365", "-nodes",
                "-subj", "/CN=localhost",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: could not generate a certificate automatically.\n"
              "Install openssl, or generate cert.pem/key.pem yourself and place them in "
              f"{HERE}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[cert] wrote {CERT_PATH} and {KEY_PATH}")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        # Camera access needs a proper secure context; also disable caching
        # while developing so edits show up on refresh without a hard reset.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    ensure_cert()

    httpd = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=CERT_PATH, keyfile=KEY_PATH)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    print(f"[server] serving {os.path.abspath(WEB_DIR)}")
    print(f"[server] https://localhost:{args.port}  (on this machine)")
    print(f"[server] https://<this-machine-local-ip>:{args.port}  (from your phone)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
