"""Step 1, and the link to Geotab: find which way is down, then split the
acceleration into a vertical part and a horizontal part. After this, the
signals look the same however the phone is held -- the same job Geotab's
orientation algorithm does for a GO device under a dashboard."""
import numpy as np

FS = 100  # Hz -- set Sensor Logger to 100 Hz


def gravity_lowpass(acc, tau_s=1.0):
    """Gravity is the slow part of the signal: a running average over ~1 s.
    acc: (N,3) total acceleration in the phone frame, m/s^2."""
    alpha = 1 / (1 + tau_s * FS)
    g = np.empty_like(acc)
    g[0] = acc[: FS].mean(axis=0)
    for i in range(1, len(acc)):
        g[i] = g[i - 1] + alpha * (acc[i] - g[i - 1])
    return g


def gravity_complementary(acc, gyro, k=0.98):
    """Bonus: sensor fusion. The gyro says how the phone rotated since the last
    sample (smooth but drifts); the accelerometer says where down is (noisy but
    never drifts). Blend them. gyro: (N,3) rad/s."""
    g = np.empty_like(acc)
    g[0] = acc[: FS].mean(axis=0)
    dt = 1 / FS
    for i in range(1, len(acc)):
        w = gyro[i] * dt
        rotated = g[i - 1] - np.cross(w, g[i - 1])     # small-angle rotation of the down vector
        g[i] = k * rotated + (1 - k) * acc[i]
    return g


def vertical_horizontal(acc, g):
    """Project onto down. vertical: up/down bounce (foot strikes).
    horizontal: size of the sideways/forward sway."""
    down = g / np.linalg.norm(g, axis=1, keepdims=True)
    along = np.sum(acc * down, axis=1)
    vertical = along - np.linalg.norm(g, axis=1)
    horizontal = np.linalg.norm(acc - along[:, None] * down, axis=1)
    return vertical, horizontal, down
