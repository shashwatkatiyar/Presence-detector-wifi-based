import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import deque
import time, subprocess, re

WIN_SIZE = 40  # ~20s window if interval=0.5s

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
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xtr, ytr)
    print(classification_report(yte, clf.predict(Xte)))
    return clf

def live_detect(clf):
    print("Live detection. Move/stand to test...")
    buf = deque(maxlen=WIN_SIZE)
    while True:
        rssi = get_rssi_percent()
        if rssi is not None:
            buf.append(rssi)
            if len(buf) == WIN_SIZE:
                x = featurize(list(buf)).reshape(1, -1)
                pred = clf.predict(x)[0]
                probs = clf.predict_proba(x)[0]
                print(f"RSSI={rssi}% => {pred}  conf={np.max(probs):.2f}")
        else:
            print("RSSI=None (check Wi‑Fi)")
        time.sleep(0.5)

if __name__ == "__main__":
    clf = train("rssi_log.csv")
    live_detect(clf)