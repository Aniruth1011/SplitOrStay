"""Offline verification helper. Uses existing local uv archives without installing packages.

Run with the Python ABI matching cached wheels (3.13 in the development environment).
Normal users should use `uv sync`; this helper is not a dependency lock or deployment method.
"""
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for archive in sorted((root / ".uv-cache/archive-v0").iterdir()):
    sys.path.append(str(archive))
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))
if len(sys.argv) < 2:
    raise SystemExit("usage: python3.13 scripts/cached_python.py MODULE [ARGS...]")
module = sys.argv.pop(1)
sys.argv[0] = module
runpy.run_module(module, run_name="__main__")
