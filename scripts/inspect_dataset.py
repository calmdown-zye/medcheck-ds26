from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from medcheck.data import iter_json_paths, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    paths = list(iter_json_paths(args.data_root))
    records, errors = load_dataset(args.data_root, strict=False)

    split_counts = Counter(r.split for r in records)
    domain_counts = Counter(r.domain_name for r in records)
    q_type_counts = Counter(r.q_type_name for r in records)

    empty_question = [r for r in records if not r.question]
    empty_answer = [r for r in records if not r.answer]

    question_map: dict[str, list[str]] = defaultdict(list)
    id_map: dict[str, list[str]] = defaultdict(list)
    for record in records:
        question_map[record.question.strip()].append(record.source_file)
        id_map[record.qa_id].append(record.source_file)

    duplicate_questions = {
        key: value for key, value in question_map.items()
        if key and len(value) > 1
    }
    duplicate_ids = {
        key: value for key, value in id_map.items()
        if key and len(value) > 1
    }

    report = {
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "json_files": len(paths),
        "records": len(records),
        "failed_files": len(errors),
        "split_counts": dict(split_counts),
        "domain_counts": dict(domain_counts),
        "question_type_counts": dict(q_type_counts),
        "empty_question": len(empty_question),
        "empty_answer": len(empty_answer),
        "duplicate_question_groups": len(duplicate_questions),
        "duplicate_qa_id_groups": len(duplicate_ids),
        "errors": errors[:50],
        "examples": [
            {
                "qa_id": r.qa_id,
                "domain": r.domain_name,
                "q_type": r.q_type_name,
                "question": r.question[:300],
                "answer": r.answer[:300],
                "source_file": r.source_file,
            }
            for r in records[: args.examples]
        ],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.report:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
