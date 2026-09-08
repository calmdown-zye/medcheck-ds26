from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from medcheck.index import MedicalHybridIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--embedding-model",
        default="intfloat/multilingual-e5-small",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    index = MedicalHybridIndex.build(
        data_root=args.data_root,
        output_dir=args.output_dir,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
        strict=args.strict,
    )

    print(f"완료: {len(index.records):,}개 레코드")
    print(f"저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
