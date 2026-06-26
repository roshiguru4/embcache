"""Enable ``python -m embcache`` as an alias for the ``embcache`` CLI."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
