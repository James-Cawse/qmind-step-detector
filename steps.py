"""Step 2 and 3: the frequency domain, and the step detector."""
import numpy as np
from orientation import FS


def spectrum(x):
    """FFT of a signal. Returns (freqs Hz, amplitude)."""
    x = (x - x.mean()) * np.hanning(len(x))
    return np.fft.rfftfreq(len(x), 1 / FS), np.abs(np.fft.rfft(x)) / len(x)


def cadence_hz(x):
    """Steps per second = the biggest FFT peak between 1 and 4 Hz.
    Walking is ~1.6-2.1 Hz, running ~2.5-3.2 Hz. A foot strike is a sharp
    spike, so the FFT also shows a peak at TWICE the cadence (a harmonic).
    If there is a strong peak at half the frequency, that is the real cadence."""
    f, a = spectrum(x)
    band = (f >= 1) & (f <= 4)
    fb, ab = f[band], a[band]
    best = fb[np.argmax(ab)]
    half = (fb > best / 2 - 0.3) & (fb < best / 2 + 0.3)
    if half.any() and ab[half].max() > 0.4 * ab.max():
        best = fb[half][np.argmax(ab[half])]
    return float(best)


def features(vertical, horizontal):
    """What the ML model sees for each 2 s window."""
    f, a = spectrum(vertical)
    e = a ** 2
    def band(lo, hi): return e[(f >= lo) & (f < hi)].sum()
    tot = band(0.5, 20) + 1e-9
    return {
        "cadence_hz": cadence_hz(vertical),
        "v_std": vertical.std(), "h_std": horizontal.std(),
        "v_p2p": np.ptp(vertical),
        "e_walk_band": band(1.2, 2.3) / tot, "e_run_band": band(2.3, 3.5) / tot,
        "e_high": band(5, 20) / tot,
    }


def classify(feat):
    """Baseline rule -- the team replaces this with a trained model (train.py)."""
    if feat["v_std"] < 0.4:
        return "STILL"
    return "RUNNING" if feat["cadence_hz"] > 2.35 or feat["v_std"] > 5 else "WALKING"


def detect_steps(vertical, state):
    """Peak detection on the vertical signal, tuned by the motion state.
    Returns sample indices of steps. The threshold and spacing adapt to
    walking vs running -- a single fixed threshold misses soft walking steps
    or double-counts bouncy running ones."""
    if state == "STILL":
        return np.array([], int)
    min_gap = int(FS * (0.28 if state == "RUNNING" else 0.4))
    thr = 0.6 * vertical.std()
    peaks, last = [], -min_gap
    for i in range(1, len(vertical) - 1):
        if vertical[i] > thr and vertical[i] >= vertical[i - 1] and vertical[i] > vertical[i + 1] and i - last >= min_gap:
            peaks.append(i); last = i
    return np.array(peaks, int)
