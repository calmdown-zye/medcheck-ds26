from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct"
DEFAULT_EMBEDDING_MODEL = PROJECT_ROOT / "models" / "multilingual-e5-small"
DEFAULT_INDEX = PROJECT_ROOT / "artifacts" / "medical_index"
load_dotenv(PROJECT_ROOT / ".env")


def resolve_path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name, "").strip()
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def check_model(path: Path) -> list[str]:
    errors: list[str] = []
    config = path / "config.json"
    index_file = path / "model.safetensors.index.json"
    if not config.is_file():
        return [f"모델 config.json 없음: {config}"]
    if not index_file.is_file():
        if not (path / "model.safetensors").is_file():
            errors.append(f"모델 가중치 없음: {path}")
        return errors

    payload = json.loads(index_file.read_text(encoding="utf-8"))
    files = set(payload.get("weight_map", {}).values())
    missing = sorted(name for name in files if not (path / name).is_file())
    if missing:
        errors.append(f"누락된 모델 shard: {missing}")
    expected = int(payload.get("metadata", {}).get("total_size", 0))
    actual = sum((path / name).stat().st_size for name in files if (path / name).is_file())
    if expected and actual < expected:
        errors.append(f"불완전한 모델: {actual:,}/{expected:,} bytes")
    return errors


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    model_path = resolve_path("LOCAL_LLM_MODEL", DEFAULT_MODEL)
    embedding_path = resolve_path("MEDCHECK_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    index_path = resolve_path("MEDCHECK_INDEX_DIR", DEFAULT_INDEX)
    adapter_value = os.getenv("LOCAL_LLM_ADAPTER", "").strip()

    print("Python:", sys.version.split()[0])
    print("OS:", platform.platform())
    print("아키텍처:", platform.machine())
    print("프로젝트:", PROJECT_ROOT)
    print("모델:", model_path)
    print("임베딩 모델:", embedding_path)
    print("인덱스:", index_path)

    if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
        errors.append("Python 3.10~3.12가 필요합니다.")

    try:
        import torch

        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print("PyTorch:", torch.__version__)
        print("자동 선택 장치:", device)
        if device == "cpu":
            warnings.append("CPU 실행은 가능하지만 생성 속도가 매우 느릴 수 있습니다.")
    except ImportError:
        errors.append("torch가 설치되지 않았습니다.")

    errors.extend(check_model(model_path))
    if not (embedding_path / "modules.json").is_file():
        errors.append(f"임베딩 모델 modules.json 없음: {embedding_path}")
    for name in ("records.jsonl", "embeddings.npy", "tokens.pkl", "index_config.json"):
        if not (index_path / name).is_file():
            errors.append(f"인덱스 파일 없음: {index_path / name}")

    if adapter_value:
        adapter_path = resolve_path("LOCAL_LLM_ADAPTER", PROJECT_ROOT)
        print("어댑터:", adapter_path)
        if not (adapter_path / "adapter_config.json").is_file():
            errors.append(f"adapter_config.json 없음: {adapter_path}")
        if not any(adapter_path.glob("adapter_model.*")):
            errors.append(f"adapter_model 파일 없음: {adapter_path}")
    else:
        print("어댑터: 사용하지 않음")

    for warning in warnings:
        print("WARN:", warning)
    for error in errors:
        print("ERROR:", error)
    print("결과:", "PASS" if not errors else "FAIL")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
