"""Public API and version retrieval for the fleetmaster package."""

from importlib.metadata import PackageNotFoundError, version

DIST_NAME: str = "fleetmaster"

try:
    __version__: str = version(DIST_NAME)
except PackageNotFoundError:
    __version__ = "unknown"

__all__: list[str] = [
    "__version__",
]
