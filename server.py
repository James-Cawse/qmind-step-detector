"""Live demo. Run:  python3 server.py
Sensor Logger: Settings > Data Streaming > HTTP Push > paste the URL printed below.
Phone and laptop on the same hotspot. No phone?  python3 simulate.py"""
import json, socket, threading, collections, os, warnings
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
from orientation import FS, gravity_lowpass, vertical_horizontal
from steps import spectrum, features, classify, detect_steps
warnings.filterwarnings("ignore", category=RuntimeWarning)  # spurious numpy-on-Mac matmul warnings

PORT = 8000
buf = collections.deque(maxlen=8 * FS)     # (sample number, x, y, z)
grav = [None]
n_seen = [0]
steps = {"count": 0, "last": -10 ** 9}
lock = threading.Lock()


def ingest(payload):
    with lock:
        for p in payload:
            name, v = p.get("name", "").lower(), p.get("values", {})
            xyz = [v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0)]
            if name == "gravity":
                grav[0] = xyz
            elif name in ("totalacceleration", "accelerometer"):
                if name == "accelerometer" and grav[0]:      # iOS sends gravity separately
                    xyz = [a + g for a, g in zip(xyz, grav[0])]
                n_seen[0] += 1
                buf.append([n_seen[0], *xyz])


def state():
    with lock:
        b = np.array(buf)
    if len(b) < 3 * FS:
        return {"ready": False}
    idx, acc = b[:, 0].astype(int), b[:, 1:]
    if np.median(np.linalg.norm(acc, axis=1)) < 3:            # phone sent g units
        acc = acc * 9.81
    g = gravity_lowpass(acc)
    vert, horiz, down = vertical_horizontal(acc, g)
    w = slice(-2 * FS, None)
    feat = features(vert[w], horiz[w])
    st = classify(feat)
    # count steps once each, ignoring the last 0.2 s where a peak may not be finished
    for p in detect_steps(vert[-4 * FS:], st):
        gi = idx[-4 * FS:][p]
        if gi > steps["last"] + 0.25 * FS and gi < idx[-1] - 0.2 * FS:
            steps["count"] += 1; steps["last"] = gi
    f, a = spectrum(vert[w])
    keep = f <= 8
    tilt = float(np.degrees(np.arccos(abs(down[-1][2]))))
    return {"ready": True, "state": st, "steps": steps["count"], "cadence": feat["cadence_hz"],
            "spm": round(feat["cadence_hz"] * 60), "down": down[-1].tolist(), "tilt": tilt,
            "freqs": f[keep].tolist(), "amp": a[keep].tolist(), "vertical": vert[-3 * FS:].tolist()}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/state":
            self._send(json.dumps(state()))
        else:
            self._send(open(os.path.join(os.path.dirname(__file__), "dashboard.html"), "rb").read(), "text/html")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
        if self.path == "/reset":
            steps.update(count=0); self._send("{}"); return
        try:
            ingest(json.loads(body).get("payload", []))
        except Exception as e:
            print("bad packet:", e)
        self._send("{}")


if __name__ == "__main__":
    ip = socket.gethostbyname(socket.gethostname())
    print(f"Sensor Logger push URL:  http://{ip}:{PORT}/data")
    print(f"Dashboard:               http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
