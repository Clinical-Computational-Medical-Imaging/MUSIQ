import sys
from unittest.mock import MagicMock

# SimpleITK/gdcm are heavy compiled deps pulled in transitively via musiq.utils; importing the
# real libraries has been observed to segfault under constrained environments (e.g. SLURM cgroups).
# Stub them out before any test module imports a musiq submodule.
sys.modules.setdefault("SimpleITK", MagicMock())
sys.modules.setdefault("gdcm", MagicMock())
