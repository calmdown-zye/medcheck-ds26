from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any


def describe(value: Any, depth: int = 0, max_depth: int = 5) -> Any:
    if depth >= max_depth:
        return type(value).__name__
    if isinstance(value, dict):
        return {
            str(key).replace("\ufeff", ""): describe(
                child, depth + 1, max_depth
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        if not value:
            return []
        return [describe(value[0], depth + 1, max_depth)]
    return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()

    path = Path(args.json_path).expanduser().resolve()
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    print("[스키마]")
    print(json.dumps(describe(data, max_depth=args.max_depth), ensure_ascii=False, indent=2))
    print("\n[원본 예시]")
    print(json.dumps(data, ensure_ascii=False, indent=2)[:5000])


if __name__ == "__main__":
    main()
