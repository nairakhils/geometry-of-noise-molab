"""One-command provenance harness.

Regenerates every committed .npz under data/, hashes the resulting files,
writes data/manifest.json, and runs the pytest suite. The committed
manifest.json is the reference; a fresh run should match it byte-for-byte
as long as the fixed seeds in the precompute scripts remain untouched.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    scripts_in_order = [
        "scripts/precompute_arrays.py",
        "scripts/linear_score_fit.py",
        "scripts/train_score_mlp.py",
    ]
    for script in scripts_in_order:
        print(f"Running {script}...")
        subprocess.run([sys.executable, script], check=True, cwd=root)

    data_dir = root / "data"
    hashes: dict[str, str] = {}
    for npz in sorted(data_dir.glob("*.npz")):
        h = hashlib.sha256(npz.read_bytes()).hexdigest()[:16]
        hashes[npz.name] = h

    (data_dir / "manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")

    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True, cwd=root)

    print("Reproduction complete. Manifest:")
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
