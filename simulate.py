"""Fake phone in a pocket at a random angle. Walks; press Enter to switch
between walking and running. Prints the true step count to compare."""
import json, time, threading, urllib.request, os
import numpy as np
FS = 100
R, _ = np.linalg.qr(np.random.normal(size=(3, 3)))      # random phone orientation
mode = ["walk"]
def toggle():
    for _ in iter(input, None):
        mode[0] = "run" if mode[0] == "walk" else "walk"; print("mode:", mode[0])
threading.Thread(target=toggle, daemon=True).start()
phase, true_steps, t_print, t = 0.0, 0, 0, 0.0
while True:
    f, amp = (1.8, 3.0) if mode[0] == "walk" else (2.8, 9.0)
    old = phase; phase += f / FS
    if int(phase) > int(old): true_steps += 1
    s = phase % 1
    vert = amp * np.exp(-((s - 0.1) / 0.06) ** 2) - amp * 0.15        # heel strike spike
    world = np.array([0.6 * np.sin(np.pi * phase), 0.3 * np.sin(2 * np.pi * phase), 9.81 + vert])
    world += np.random.normal(0, 0.25, 3)
    x, y, z = R.T @ world
    pkt = {"payload": [{"name": "totalacceleration", "values": {"x": x, "y": y, "z": z}}]}
    urllib.request.urlopen(urllib.request.Request("http://localhost:8000/data", json.dumps(pkt).encode()))
    t += 1 / FS
    if t - t_print > 1: open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".true_steps"), "w").write(str(true_steps)); t_print = t
    time.sleep(1 / FS)
