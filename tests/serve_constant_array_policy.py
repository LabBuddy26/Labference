"""Hold-position HTTP action service for centrifuge transport smoke tests."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json


class Handler(BaseHTTPRequestHandler):
    fixed_ee_pose = None
    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size))
        if payload.get("command") == "reset":
            answer = {"status": "ok"}
        elif payload.get("command") == "infer":
            positions = payload["request"]["joint_positions"]
            answer = {"action": [self.fixed_ee_pose or positions[:8]] * 8}
        else:
            self.send_error(400, "unknown command")
            return
        encoded = json.dumps(answer).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--ee-pose", nargs=8, type=float, help="Fixed xyz+wxyz+finger-position action for EE transport smoke")
    args = parser.parse_args()
    Handler.fixed_ee_pose = args.ee_pose
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
