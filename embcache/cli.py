"""Command-line interface: inspect a cache without writing Python.

``embcache`` ships three subcommands for looking at a cache file that's shared
across runs or processes:

    embcache report <path>   # cumulative savings stored in the cache
    embcache info   <path>   # entries, size, models, timestamps
    embcache clear  <path>   # delete cached vectors (lifetime stats kept)

By default the target is a SQLite file (the package default). Point at Redis
instead with ``--backend redis --redis-url redis://...``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from . import __version__
from .cache import EmbeddingCache
from .info import format_info


def _add_target_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "path",
        nargs="?",
        default="./emb.db",
        help="path to the SQLite cache file (default: ./emb.db)",
    )
    p.add_argument(
        "--backend",
        choices=["sqlite", "redis"],
        default="sqlite",
        help="cache backend (default: sqlite)",
    )
    p.add_argument(
        "--redis-url",
        default="redis://localhost:6379/0",
        help="Redis URL when --backend redis",
    )
    p.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="model name for the report header (default: text-embedding-3-small)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embcache",
        description="Inspect and manage an embcache cache.",
    )
    parser.add_argument(
        "--version", action="version", version=f"embcache {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="print the cumulative savings report")
    _add_target_args(report)

    info = sub.add_parser("info", help="print cache contents and metadata")
    _add_target_args(info)

    clear = sub.add_parser("clear", help="delete all cached vectors")
    _add_target_args(clear)
    clear.add_argument(
        "--yes", action="store_true", help="skip the confirmation prompt"
    )
    clear.add_argument(
        "--reset-stats",
        action="store_true",
        help="also zero the cumulative savings counters",
    )

    return parser


def _open_cache(args: argparse.Namespace, *, must_exist: bool) -> EmbeddingCache:
    if args.backend == "redis":
        return EmbeddingCache(
            backend="redis", redis_url=args.redis_url, model=args.model
        )
    if must_exist and not os.path.exists(args.path):
        raise FileNotFoundError(args.path)
    return EmbeddingCache(backend="sqlite", path=args.path, model=args.model)


def _cmd_report(args: argparse.Namespace) -> int:
    cache = _open_cache(args, must_exist=True)
    try:
        print(cache.report(scope="lifetime"))
    finally:
        cache.close()
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    cache = _open_cache(args, must_exist=True)
    try:
        print(format_info(cache.info()))
    finally:
        cache.close()
    return 0


def _cmd_clear(args: argparse.Namespace) -> int:
    cache = _open_cache(args, must_exist=True)
    try:
        count = cache.backend.count()
        target = args.redis_url if args.backend == "redis" else args.path
        if not args.yes:
            noun = "entries" if count is None else f"{count:,} entries"
            resp = input(f"Delete {noun} from {target}? [y/N] ").strip().lower()
            if resp not in ("y", "yes"):
                print("Aborted.")
                return 1
        removed = cache.clear(reset_stats=args.reset_stats)
        suffix = " and reset stats" if args.reset_stats else ""
        print(f"Removed {removed:,} entries{suffix}.")
    finally:
        cache.close()
    return 0


_COMMANDS = {
    "report": _cmd_report,
    "info": _cmd_info,
    "clear": _cmd_clear,
}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except FileNotFoundError as exc:
        print(f"error: no cache found at {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
