"""One-time download of the fonts and the free local voice model."""
from pathlib import Path

import requests

ROOT = Path(__file__).parent
FILES = {
    "assets/fonts/Montserrat-ExtraBold.ttf":
        "https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-ExtraBold.ttf",
    "assets/fonts/Montserrat-Black.ttf":
        "https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-Black.ttf",
    "assets/models/kokoro-v1.0.onnx":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "assets/models/voices-v1.0.bin":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}

for rel, url in FILES.items():
    path = ROOT / rel
    if path.exists():
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    print("downloading", rel)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
(ROOT / "assets/music/curious").mkdir(parents=True, exist_ok=True)
print("done. Put YouTube Audio Library tracks in assets/music/<mood>/")
