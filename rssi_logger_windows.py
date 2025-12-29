import subprocess, time, re, csv
from datetime import datetime

def get_rssi_percent():
    out = subprocess.check_output(["netsh", "wlan", "show", "interfaces"], shell=True).decode()
    m = re.search(r"Signal\s*:\s*(\d+)%", out)
    return int(m.group(1)) if m else None

with open("rssi_log.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["timestamp","rssi_percent","label"])
    print("Logging RSSI... Ctrl+C to stop")
    while True:
        ts = datetime.now().isoformat(timespec="seconds")
        rssi = get_rssi_percent()
        w.writerow([ts, rssi, "empty"])   # for now just 'empty'
        print(ts, rssi)
        time.sleep(0.5)