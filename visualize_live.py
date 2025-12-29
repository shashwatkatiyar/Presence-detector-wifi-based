import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import subprocess, re, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

WIN_SIZE = 20  # ~10s window if interval=0.5s

def get_rssi_percent():
    try:
        out = subprocess.check_output(["netsh", "wlan", "show", "interfaces"], shell=True).decode("utf-8", errors="ignore")
        m = re.search(r"Signal\s*:\s*(\d+)%", out)
        if m: return int(m.group(1))
    except Exception:
        return None
    return None

def featurize(seq):
    arr = np.array(seq, dtype=float)
    if len(arr) == 0: return np.zeros(10)
    arr = arr - np.mean(arr)
    fft = np.fft.rfft(arr)
    mag = np.abs(fft)
    return np.array([
        np.mean(arr), np.std(arr), np.max(arr)-np.min(arr),
        np.sum(mag), np.mean(mag), np.std(mag),
        mag[1] if len(mag)>1 else 0.0,
        mag[2] if len(mag)>2 else 0.0,
        mag[3] if len(mag)>3 else 0.0,
        mag[4] if len(mag)>4 else 0.0
    ])

def build_dataset(csv_path, window=WIN_SIZE):
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("ts")
    X, y = [], []
    buf = deque(maxlen=window)
    labs = deque(maxlen=window)
    for _, row in df.iterrows():
        buf.append(row["rssi_percent"])
        labs.append(row["label"])
        if len(buf) == window:
            X.append(featurize(list(buf)))
            y.append(pd.Series(labs).mode()[0])
    return np.array(X), np.array(y)

def train(csv_path):
    X, y = build_dataset(csv_path)
    clf = RandomForestClassifier(n_estimators=400, random_state=42)
    clf.fit(X, y)
    return clf

def visualize(clf):
    buf = deque(maxlen=WIN_SIZE)
    fig, ax = plt.subplots()
    xs, ys = [], []
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 200)
    ax.set_xlabel("Samples")
    ax.set_ylabel("RSSI (%)")
    text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment='top')

    def update(frame):
        rssi = get_rssi_percent()
        if rssi is not None:
            buf.append(rssi)
            ys.append(rssi)
            xs.append(len(xs))
            line.set_data(xs[-200:], ys[-200:])
            if len(buf) == WIN_SIZE:
                x = featurize(list(buf)).reshape(1, -1)
                pred = clf.predict(x)[0]
                probs = clf.predict_proba(x)[0]
                text.set_text(f"{pred} ({np.max(probs):.2f})")
        return line, text

    ani = animation.FuncAnimation(fig, update, interval=250)
    plt.show()

if __name__ == "__main__":
    clf = train("rssi_log.csv")
    visualize(clf)