import subprocess, time, re, csv, argparse
from datetime import datetime
from pathlib import Path

def get_rssi_percent():
    out = subprocess.check_output(["netsh", "wlan", "show", "interfaces"], shell=True).decode("utf-8", errors="ignore")
    m = re.search(r"Signal\s*:\s*(\d+)%", out)
    return int(m.group(1)) if m else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=["empty","occupied","moving"])
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--outfile", default="rssi_log.csv")
    args = parser.parse_args()

    path = Path(args.outfile)
    new_file = not path.exists() or path.stat().st_size == 0

    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp","rssi_percent","label"])
        print(f"Logging label={args.label} every {args.interval}s. Ctrl+C to stop.")
        while True:
            ts = datetime.now().isoformat(timespec="seconds")
            rssi = get_rssi_percent()
            w.writerow([ts, rssi, args.label])
            print(f"{ts}  RSSI={rssi}%  {args.label}")
            time.sleep(args.interval)

if __name__ == "__main__":
    main()