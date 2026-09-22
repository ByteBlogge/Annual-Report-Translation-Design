"""Package marker for the optional web demo."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Lazy re-export so importing the package does not require FastAPI."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
