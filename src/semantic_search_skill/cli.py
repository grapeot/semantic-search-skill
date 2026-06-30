from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic search over local text files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query = subparsers.add_parser("query", help="Search an indexed file list")
    query.add_argument("--file-list", required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--top-k", type=int, default=10)
    query.add_argument("--cache-dir", default=None)

    rebuild = subparsers.add_parser("rebuild", help="Build or refresh cache for a file list")
    rebuild.add_argument("--file-list", required=True)
    rebuild.add_argument("--cache-dir", default=None)
    rebuild.add_argument("--workers", type=int, default=64)
    rebuild.add_argument("--batch-size", type=int, default=128)

    doctor = subparsers.add_parser("doctor", help="Validate cache health")
    doctor.add_argument("--cache-dir", default=None)

    stats = subparsers.add_parser("stats", help="Print cache statistics")
    stats.add_argument("--cache-dir", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    parser.error(f"command not implemented yet: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
