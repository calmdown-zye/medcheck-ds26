from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct"
EMBEDDING_REPOSITORY = "intfloat/multilingual-e5-small"
EMBEDDING_OUTPUT = PROJECT_ROOT / "models" / "multilingual-e5-small"


def main() -> None:
    parser = argparse.ArgumentParser(description="MedCheck 기본 LLM을 다운로드합니다.")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="검색용 multilingual-e5-small 다운로드를 생략합니다.",
    )
    args = parser.parse_args()

    output = args.output.expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("저장소:", args.repository)
    print("저장 위치:", output)
    snapshot_download(
        repo_id=args.repository,
        local_dir=output,
        force_download=args.force,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.md",
            "LICENSE",
        ],
    )
    print("모델 다운로드 완료:", output)

    if not args.skip_embedding:
        print("검색 모델:", EMBEDDING_REPOSITORY)
        print("저장 위치:", EMBEDDING_OUTPUT)
        snapshot_download(
            repo_id=EMBEDDING_REPOSITORY,
            local_dir=EMBEDDING_OUTPUT,
            force_download=args.force,
            allow_patterns=[
                "config.json",
                "modules.json",
                "sentence_bert_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "sentencepiece.bpe.model",
                "model.safetensors",
                "1_Pooling/config.json",
                "README.md",
                "LICENSE",
            ],
        )
        print("검색 모델 다운로드 완료:", EMBEDDING_OUTPUT)


if __name__ == "__main__":
    main()
