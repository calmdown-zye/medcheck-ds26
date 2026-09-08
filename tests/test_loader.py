from __future__ import annotations

import json
from pathlib import Path

from medcheck.data import load_dataset, load_json_file


def test_single_aihub_record(tmp_path: Path) -> None:
    folder = tmp_path / "Training" / "02.라벨링데이터" / "TL_비뇨의학과"
    folder.mkdir(parents=True)
    path = folder / "전문_281.json"
    path.write_text(
        json.dumps(
            {
                "\ufeffqa_id": "281",
                "domain": "8",
                "q_type": "1",
                "question": "질문입니다.",
                "answer": "1) 정답입니다.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = load_json_file(path)
    assert len(records) == 1
    assert records[0].qa_id == "281"
    assert records[0].domain_name == "비뇨의학과"
    assert records[0].q_type_name == "객관식"
    assert records[0].split == "Training"


def test_nested_list(tmp_path: Path) -> None:
    path = tmp_path / "nested.json"
    path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "qa_id": 1,
                        "domain": 17,
                        "q_type": 3,
                        "question": "내과 질문",
                        "answer": "내과 답변",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records, errors = load_dataset(tmp_path)
    assert not errors
    assert len(records) == 1
    assert records[0].domain_name == "내과"
