"""py_lap_solver: A unified framework for Linear Assignment Problem solvers."""

from . import solvers
from .base import LapSolver

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # Python < 3.8
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("py-lap-solver")
except PackageNotFoundError:
    # Package not installed
    __version__ = "unknown"

__all__ = ["LapSolver", "solvers", "__version__"]
