#!/usr/bin/env python3
"""Standalone surface-math reference + test-vector generator.

This is the executable spec for Slitscan Studio's profiles. The math is COPIED
from the CLI (slitscan/profiles/{ramp,reverse,tent}.py) — it deliberately does
NOT import the CLI package, so the app subdir is fully self-contained (spec §3).

The Swift implementation (Sources/SlitscanCore/Profiles.swift) and the Metal
shader (Gather.metal) must reproduce these vectors within tolerance. Run this to
regenerate the fixture the parity test loads:

    python3 reference/generate_vectors.py
"""

import json
import os


# --- normalized delay profiles (delay in [0,1]); copied from the CLI ---------

def ramp(t: float, vanguard: float) -> float:
    # profiles/ramp.py: vanguard<=0.5 -> t ; else 1-t
    return t if vanguard <= 0.5 else 1.0 - t


def reverse(t: float, vanguard: float) -> float:
    # profiles/reverse.py: vanguard>=0.5 -> 1-t ; else t
    return (1.0 - t) if vanguard >= 0.5 else t


def tent(t: float, vanguard: float) -> float:
    # profiles/tent.py: clip(|t-vanguard| / max(max(v,1-v),1e-9), 0, 1)
    denom = max(max(vanguard, 1.0 - vanguard), 1e-9)
    return min(max(abs(t - vanguard) / denom, 0.0), 1.0)


PROFILES = {"ramp": ramp, "reverse": reverse, "tent": tent}


def delays(fn, n: int, vanguard: float) -> list[float]:
    # mirrors the CLI: t = x_coords / max(n-1, 1)
    denom = max(n - 1, 1)
    return [fn(i / denom, vanguard) for i in range(n)]


def main() -> None:
    ns = [1, 2, 5, 16, 33, 100]
    vanguards = [0.0, 0.25, 0.5, 0.5001, 0.75, 1.0]
    cases = []
    for name, fn in PROFILES.items():
        for n in ns:
            for v in vanguards:
                cases.append(
                    {
                        "profile": name,
                        "n": n,
                        "vanguard": v,
                        "delays": delays(fn, n, v),
                    }
                )

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "Tests", "SlitscanCoreTests", "Resources")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_vectors.json")
    with open(out_path, "w") as f:
        json.dump({"cases": cases}, f, indent=2)
    print(f"wrote {len(cases)} cases -> {os.path.relpath(out_path, here)}")


if __name__ == "__main__":
    main()
