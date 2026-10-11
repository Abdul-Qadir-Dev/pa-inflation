"""Download NAB (three real-data folders) and the NASA SMAP/MSL telemanom release into data/."""
import argparse
import io
import json
import os
import time
import zipfile

import requests

NAB_RAW = "https://raw.githubusercontent.com/numenta/NAB/master/"
NAB_DIRS = ["realAWSCloudwatch", "realKnownCause", "realTweets"]
NASA_ZIP = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
NASA_LABELS = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"

DATA = "data"


def fetch(url, tries=3, timeout=60):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            print(f"  try {i + 1} failed for {url}: {e}")
            time.sleep(2 * (i + 1))
    return None


def get_nab():
    out = os.path.join(DATA, "nab")
    os.makedirs(out, exist_ok=True)
    raw = fetch(NAB_RAW + "labels/combined_windows.json")
    if raw is None:
        print("could not get NAB labels, skipping NAB")
        return
    windows = json.loads(raw)
    with open(os.path.join(out, "combined_windows.json"), "wb") as f:
        f.write(raw)

    # the labels file lists every series, so use it as the file index
    names = sorted(k for k in windows if k.split("/")[0] in NAB_DIRS)
    skipped = []
    for name in names:
        dest = os.path.join(out, name)
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        content = fetch(NAB_RAW + "data/" + name)
        if content is None:
            skipped.append(name)
            continue
        with open(dest, "wb") as f:
            f.write(content)
    print(f"NAB: {len(names) - len(skipped)} of {len(names)} series")
    if skipped:
        print("skipped:", ", ".join(skipped))


def get_nasa():
    out = os.path.join(DATA, "nasa")
    if os.path.exists(os.path.join(out, "test")):
        print("NASA: already there")
        return True
    os.makedirs(out, exist_ok=True)
    labels = fetch(NASA_LABELS)
    blob = fetch(NASA_ZIP, timeout=600)
    if labels is None or blob is None:
        print("NASA download failed, run will be NAB only")
        return False
    with open(os.path.join(out, "labeled_anomalies.csv"), "wb") as f:
        f.write(labels)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        # the zip has a top-level data/ folder with train/ and test/ inside
        for member in z.namelist():
            parts = member.split("/")
            if len(parts) < 3 or not parts[-1].endswith(".npy"):
                continue
            dest = os.path.join(out, parts[-2], parts[-1])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(z.read(member))
    print("NASA: done")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-nasa", action="store_true")
    args = ap.parse_args()
    get_nab()
    if not args.skip_nasa:
        get_nasa()
