"""Download the four OSF files (project dmhrw) into data/ if missing, and verify SHA-256 against data/MANIFEST.md. Usage: python -I src/download_data.py"""
import hashlib, sys, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / "data"; D.mkdir(exist_ok=True)
FILES = {"RV20_Mood_MC_20230519.sav": "yqxck", "RV20_Questionnaires_MC_20230519.sav": "5dwsz", "RV20_EmotionRecognition_MC_20230519.sav": "nyg62", "Script_MC_RV20_EmotionRecognition.R": "8hwte"}
manifest = {l.split()[1]: l.split()[0] for l in (D / "MANIFEST.md").read_text().splitlines() if l.strip()}
bad = 0
for name, osf_id in FILES.items():
    f = D / name
    if not f.exists():
        print("downloading", name); urllib.request.urlretrieve(f"https://osf.io/download/{osf_id}/", f)
    h = hashlib.sha256(f.read_bytes()).hexdigest(); ok = manifest.get(name) == h; bad += not ok; print(f"{'OK  ' if ok else 'BAD '} {name} ({f.stat().st_size:,} bytes)")
sys.exit(1 if bad else 0)
