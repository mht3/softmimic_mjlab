"""SoftMimic deployment utilities."""

from importlib import metadata


def __getattr__(name: str) -> str:
    if name == "__version__":
        try:
            return metadata.version("softmimic-deploy")
        except metadata.PackageNotFoundError:
            return "0.0.0"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = []
