from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .constants import DOMAIN_CODE_TO_NAME, QUESTION_TYPE_CODE_TO_NAME
from .models import MedicalQARecord


def _clean_key(key: Any) -> str:
    return str(key).replace("\ufeff", "").strip()


def _normalize_object(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {_clean_key(k): _normalize_object(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_object(v) for v in obj]
    return obj


def _iter_qa_objects(obj: Any) -> Iterator[dict[str, Any]]:
    """중첩 JSON 안에서 question/answer를 가진 객체를 재귀적으로 탐색합니다."""
    if isinstance(obj, dict):
        keys = {_clean_key(k) for k in obj.keys()}
        if {"question", "answer"}.issubset(keys):
            yield _normalize_object(obj)
            return
        for value in obj.values():
            yield from _iter_qa_objects(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_qa_objects(value)


def infer_split(path: Path) -> str:
    lower_parts = [part.lower() for part in path.parts]
    if any("training" == part or part.startswith("train") for part in lower_parts):
        return "Training"
    if any("validation" == part or part.startswith("valid") or part.startswith("val") for part in lower_parts):
        return "Validation"
    return "Unknown"


def infer_domain_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        for prefix in ("TL_", "VL_"):
            if part.startswith(prefix):
                return part[len(prefix):]
    return "미분류"


def load_json_file(path: str | Path) -> list[MedicalQARecord]:
    path = Path(path).expanduser().resolve()
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 파싱 실패: {path} ({exc})") from exc
    except OSError as exc:
        raise OSError(f"파일 읽기 실패: {path} ({exc})") from exc

    split = infer_split(path)
    path_domain = infer_domain_from_path(path)
    records: list[MedicalQARecord] = []

    for index, item in enumerate(_iter_qa_objects(raw)):
        domain_code = str(item.get("domain", "")).strip()
        q_type_code = str(item.get("q_type", "")).strip()
        qa_id = str(item.get("qa_id", "")).strip() or f"{path.stem}:{index}"

        domain_name = DOMAIN_CODE_TO_NAME.get(domain_code, path_domain)
        q_type_name = QUESTION_TYPE_CODE_TO_NAME.get(q_type_code, "미분류")

        record = MedicalQARecord(
            qa_id=qa_id,
            domain_code=domain_code,
            domain_name=domain_name,
            q_type_code=q_type_code,
            q_type_name=q_type_name,
            question=item.get("question", ""),
            answer=item.get("answer", ""),
            source_file=str(path),
            split=split,
        )
        records.append(record)

    if not records:
        raise ValueError(
            f"question/answer 필드를 가진 레코드를 찾지 못했습니다: {path}"
        )
    return records


def iter_json_paths(data_root: str | Path) -> Iterator[Path]:
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"데이터 경로가 존재하지 않습니다: {root}")
    if root.is_file():
        if root.suffix.lower() != ".json":
            raise ValueError(f"JSON 파일이 아닙니다: {root}")
        yield root
        return

    yield from sorted(root.rglob("*.json"))


def load_dataset(
    data_root: str | Path,
    *,
    strict: bool = False,
) -> tuple[list[MedicalQARecord], list[dict[str, str]]]:
    records: list[MedicalQARecord] = []
    errors: list[dict[str, str]] = []

    for path in iter_json_paths(data_root):
        try:
            records.extend(load_json_file(path))
        except Exception as exc:
            if strict:
                raise
            errors.append({"path": str(path), "error": str(exc)})

    return records, errors


def save_records_jsonl(records: list[MedicalQARecord], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json(ensure_ascii=False) + "\n")


def load_records_jsonl(path: str | Path) -> list[MedicalQARecord]:
    records: list[MedicalQARecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(MedicalQARecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number} 레코드 파싱 실패") from exc
    return records
