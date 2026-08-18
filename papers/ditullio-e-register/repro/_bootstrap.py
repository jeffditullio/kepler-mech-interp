"""Path/env bootstrap for the repro stage scripts. Import FIRST, before src/registry.

Pins KEPLER_CKPT_DIR to papers/ditullio-e-register/models (unless already set in the environment)
and puts the repo root and papers/ditullio-e-register/ on sys.path, so every stage script runs
from any working directory. Exposes the shared path constants.

Script-mode only: model_analysis.py is imported by reproduce_analysis.py (which does its
own bootstrapping) and must not import this.
"""

import os
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
ROOT = PAPER_DIR.parent.parent
FIGURES = PAPER_DIR / "figures"
TABLES = PAPER_DIR / "tables"
MODELS = PAPER_DIR / "models"

os.environ.setdefault("KEPLER_CKPT_DIR", str(MODELS))
sys.path.insert(0, str(PAPER_DIR))
sys.path.insert(0, str(ROOT))

# The ONE registry import: _bootstrap owns sys.path, so it owns the import
# that depends on it. Stages write `from _bootstrap import model_registry` and can
# never hit the import-order hazard (ruff sorting `import model_registry` above
# the path setup).
import model_registry
