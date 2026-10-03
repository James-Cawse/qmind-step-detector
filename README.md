# Step Detector · QMIND × Geotab

## The big picture

Geotab's GO device sits in a car at a random angle. Before its accelerometer data means anything, it has to work out which way is down. That's the hard part of our whole project.

Today we practise that same problem on something we can test in a room: **counting steps with a phone, however it's held, walking or running.**

1. **Find down.** Gravity is always there, so a slow average of the accelerometer points down. (`orientation.py`)
2. **Split the motion** into vertical (bouncing) and horizontal (swaying).
3. **FFT the vertical signal.** Walking peaks near 1.8 Hz, running near 2.8 Hz. (`steps.py`)
4. **ML decides** still / walking / running, and peak detection counts the steps.
5. **The experiment** we pitch: train the model three ways and show orientation is what makes it work. (`train.py`)
   - A: raw x/y/z
   - B: magnitude only
   - C: rotated to vertical + horizontal

## The files

| File | What it does |
|---|---|
| `orientation.py` | Finds down, splits vertical / horizontal |
| `steps.py` | FFT, cadence, features, step counting |
| `train.py` | The A/B/C experiment |
| `server.py` + `dashboard.html` | Live demo: phone streams in, dashboard shows steps |
| `simulate.py` | Fake phone, for testing without one |
| `data/` | Everyone's recordings go here |

## 1. Set up (5 min)

1. Send James your GitHub username so he can add you. Accept the invite email.
2. Install the free **Sensor Logger** app. In its settings, set the rate to 100 Hz and turn on Accelerometer, Gravity and Gyroscope.
3. On your laptop:
   ```
   git clone https://github.com/James-Cawse/qmind-step-detector.git
   cd qmind-step-detector
   pip install -r requirements.txt
   python3 server.py
   ```
   In a second terminal, run `python3 simulate.py`. Open the URL that `server.py` prints. If the step count moves, you're set up.

## 2. Record (first 25 min)

Each person records 50 counted steps for each of these:
- walk, phone in hand
- walk, phone in pocket
- run, phone in hand
- run, phone in pocket
- 20 seconds standing still

In Sensor Logger, export each recording as CSV and unzip it. Rename the folder `<name>_<walk|run|still>_<hand|pocket>`, for example `brett_run_pocket`. Write the real step count in a `steps.txt` file inside that folder.

## 3. Upload your recordings

**Easiest way, in the browser:**
1. Open the repo on GitHub and click into `data/`.
2. Click **Add file → Upload files**.
3. Drag your recording folders in.
4. Click **Commit changes**.

**From the terminal:**
```
git pull
git add data/
git commit -m "brett recordings"
git push
```

Everyone runs `git pull` to get everyone else's data.

## 4. Upload code changes

Always run `git pull` before you start.

To share a change:
```
git add -A
git commit -m "what you changed"
git pull
git push
```

If `git push` says it was rejected, run `git pull` and push again. Stick to the files your pair owns (see below) and you'll avoid conflicts.

## 5. Build, in pairs (0:25 to 1:45)

| Pair | Owns | Goal |
|---|---|---|
| Orientation | `orientation.py` | Check that down is found correctly. Is the gyro version (`gravity_complementary`) steadier when running? |
| Frequency | `steps.py`: `cadence_hz` | Spectrograms of walk vs run. Show the cadence peak and its 2× harmonic. |
| ML | `train.py` | Run it, then report A vs B vs C. Swap the rule in `steps.classify` for the trained model. |
| Accuracy | `steps.py`: `detect_steps` | Compare detected steps with real steps on every recording. Target: within 3%. |

## 6. Demo and pitch (1:45 to 2:30)

To run the demo:
1. Put the laptop and phone on the same hotspot.
2. Run `python3 server.py`.
3. In Sensor Logger, go to Data Streaming → HTTP Push, paste the printed URL, and start.

Pitch (5 min + 2 min Q&A):
1. Geotab's problem: a cheap sensor mounted at an unknown angle.
2. Live demo with a phone in a pocket.
3. FFT: the cadence peak moves from walking to running.
4. A vs B vs C: orientation is what makes it generalise.
5. Accuracy, and what didn't work.
6. Next: the same pipeline on real GO vehicle data.
