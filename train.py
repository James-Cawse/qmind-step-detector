"""The experiment for the pitch: does knowing orientation make the model better?

Put each Sensor Logger export (unzipped folder) in data/ named
    <person>_<activity>_<placement>      e.g.  brett_run_pocket, ahmed_walk_hand
activity is walk / run / still. Then:  python3 train.py

Trains the same model three ways and tests each on a phone placement it never saw:
  A  raw x/y/z from the phone             (no orientation)
  B  magnitude |a| only                   (orientation thrown away)
  C  vertical + horizontal after rotation (orientation estimated first)
"""
import glob, os, sys, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from orientation import FS, gravity_lowpass, vertical_horizontal
from steps import features

WIN = 2 * FS


def load(folder):
    def read(name):
        p = os.path.join(folder, name + ".csv")
        return pd.read_csv(p) if os.path.exists(p) else None
    tot = read("TotalAcceleration")
    if tot is None:                                    # iOS: user accel + gravity
        a, g = read("Accelerometer"), read("Gravity")
        n = min(len(a), len(g))
        acc = a[["x", "y", "z"]].values[:n] + g[["x", "y", "z"]].values[:n]
    else:
        acc = tot[["x", "y", "z"]].values
    if np.median(np.linalg.norm(acc, axis=1)) < 3:
        acc = acc * 9.81
    return acc


def windows(acc):
    g = gravity_lowpass(acc)
    v, h, _ = vertical_horizontal(acc, g)
    mag = np.linalg.norm(acc, axis=1) - 9.81
    for s in range(FS, len(acc) - WIN, FS // 2):       # skip first second, 50% overlap
        w = slice(s, s + WIN)
        A = {}
        for i, ax in enumerate("xyz"):
            A.update({f"{ax}_{k}": val for k, val in features(acc[w, i] - acc[w, i].mean(), np.zeros(WIN)).items() if k != "h_std"})
        B = {f"m_{k}": val for k, val in features(mag[w], np.zeros(WIN)).items() if k != "h_std"}
        C = features(v[w], h[w])
        yield A, B, C


rows = []
for folder in sorted(glob.glob("data/*")):
    person, activity, placement = os.path.basename(folder).split("_")[:3]
    for A, B, C in windows(load(folder)):
        rows.append(dict(person=person, activity=activity, placement=placement, A=A, B=B, C=C))
if not rows:
    sys.exit("No recordings in data/ yet.")
df = pd.DataFrame(rows)
print(f"{len(df)} windows from {df.person.nunique()} people, placements: {sorted(df.placement.unique())}\n")

for split in ["placement", "person"]:
    print(f"Leave-one-{split}-out accuracy (tested on a {split} the model never saw):")
    for name in "ABC":
        X = pd.DataFrame(list(df[name])).values
        accs = []
        for held in df[split].unique():
            tr, te = df[split] != held, df[split] == held
            m = RandomForestClassifier(n_estimators=200, random_state=0).fit(X[tr], df.activity[tr])
            accs.append(accuracy_score(df.activity[te], m.predict(X[te])))
        label = {"A": "raw x/y/z", "B": "magnitude only", "C": "orientation-corrected"}[name]
        print(f"  {name}  {label:24s} {np.mean(accs):.0%}")
    print()
