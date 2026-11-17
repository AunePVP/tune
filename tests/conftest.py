"""Test helpers: the scripts as importable modules, a tiny PNG, and a fake OpenAI-compatible chat server."""
from __future__ import annotations

import json
import struct
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def png(width: int = 8, height: int = 8) -> bytes:
    """A white greyscale PNG, without Pillow."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\xff" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@pytest.fixture()
def chat_server():
    """A chat server answering every request with ``answers`` in turn; ``requests`` records the payloads."""
    state = {"answers": ["TYP: Text\nSeite"], "requests": []}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(body)
            answer = state["answers"][min(len(state["requests"]) - 1, len(state["answers"]) - 1)]
            data = json.dumps({"choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                               "usage": {"completion_tokens": len(answer.split()), "prompt_tokens": 10}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield state
    server.shutdown()
