"""Download official M2D runtime and weights; verify both before inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url, target):
    req = urllib.request.Request(url, headers={"User-Agent":"ConformalMOS/0.1"})
    with urllib.request.urlopen(req, timeout=120) as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=ROOT/"external")
    p.add_argument("--runtime-only", action="store_true")
    a = p.parse_args()
    a.destination.mkdir(parents=True, exist_ok=True)
    m = json.loads((ROOT/"assets/manifest.json").read_text())
    with tempfile.TemporaryDirectory(dir=a.destination) as td:
        td = Path(td)
        runtime = a.destination/"portable_m2d.py"
        if not runtime.exists():
            temp = td/"portable_m2d.py"
            download(m["m2d_runtime"]["url"], temp)
            if digest(temp) != m["m2d_runtime"]["sha256"]:
                raise ValueError("Runtime checksum mismatch")
            temp.rename(runtime)
        if digest(runtime) != m["m2d_runtime"]["sha256"]:
            raise ValueError("Existing runtime checksum mismatch")
        if a.runtime_only:
            print(runtime)
            return
        weights = a.destination/m["m2d_weights"]["directory"]/"checkpoint-300.pth"
        if not weights.exists():
            print("Downloading the official M2D archive (large, approximately GB scale).")
            archive = td/"m2d.zip"
            download(m["m2d_weights"]["url"], archive)
            with zipfile.ZipFile(archive) as z:
                matches = [n for n in z.namelist() if Path(n).name == "checkpoint-300.pth"]
                if len(matches) != 1:
                    raise ValueError("Expected exactly one checkpoint-300.pth")
                temp = td/"checkpoint-300.pth"
                with z.open(matches[0]) as src, temp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            if digest(temp) != m["m2d_weights"]["sha256"]:
                raise ValueError("Weight checksum mismatch; original archive remains untrusted")
            weights.parent.mkdir(parents=True, exist_ok=True)
            temp.rename(weights)
        if digest(weights) != m["m2d_weights"]["sha256"]:
            raise ValueError("Existing weight checksum mismatch")
    print(f"Ready: {weights}")

if __name__ == "__main__":
    main()
