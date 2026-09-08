from __future__ import annotations

import ast
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from json_repair import repair_json
from packaging.version import Version
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version


load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct"


class LocalLLMError(RuntimeError):
    pass


def _resolve_local_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _validate_model_files(model_path: Path) -> None:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json이 없는 모델 경로입니다: {model_path}")

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        if not (model_path / "model.safetensors").is_file():
            raise FileNotFoundError(
                "모델 가중치가 없습니다. "
                "python scripts/download_models.py 를 먼저 실행하세요."
            )
        return

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_files = set(index.get("weight_map", {}).values())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"모델 인덱스를 읽을 수 없습니다: {index_path}") from exc

    missing = sorted(name for name in weight_files if not (model_path / name).is_file())
    if missing:
        raise FileNotFoundError(f"모델 가중치 파일이 누락되었습니다: {missing}")

    expected_size = int(index.get("metadata", {}).get("total_size", 0))
    actual_size = sum((model_path / name).stat().st_size for name in weight_files)
    if expected_size and actual_size < expected_size:
        raise ValueError(
            f"모델 다운로드가 불완전합니다: {actual_size:,}/{expected_size:,} bytes. "
            "python scripts/download_models.py --force 를 실행하세요."
        )


def resolve_device() -> str:
    requested = os.getenv("MEDCHECK_DEVICE", "auto").strip().lower()

    if requested != "auto":
        if requested not in {"cuda", "mps", "cpu"}:
            raise ValueError(
                "MEDCHECK_DEVICE는 auto, cuda, mps, cpu 중 하나여야 합니다."
            )

        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다.")

        if requested == "mps":
            mps_available = (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
            if not mps_available:
                raise RuntimeError("MPS를 사용할 수 없습니다.")

        return requested

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


class LocalTransformersClient:
    """
    Ollama API 없이 Hugging Face Transformers 모델을 직접 실행합니다.
    """

    def __init__(
        self,
        *,
        model_path: str | None = None,
        timeout: int | None = None,
    ) -> None:
        del timeout  # 직접 추론에서는 HTTP timeout이 필요하지 않습니다.

        configured_path = (model_path or os.getenv("LOCAL_LLM_MODEL", "")).strip()
        self.model_path = (
            _resolve_local_path(configured_path)
            if configured_path
            else DEFAULT_MODEL_PATH.resolve()
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"로컬 LLM 경로가 존재하지 않습니다: {self.model_path}"
            )

        _validate_model_files(self.model_path)

        adapter_value = os.getenv("LOCAL_LLM_ADAPTER", "").strip()
        self.adapter_path = _resolve_local_path(adapter_value) if adapter_value else None
        if self.adapter_path is not None:
            if not (self.adapter_path / "adapter_config.json").is_file():
                raise FileNotFoundError(
                    "adapter_config.json이 없는 어댑터 경로입니다: "
                    f"{self.adapter_path}"
                )
            if not any(self.adapter_path.glob("adapter_model.*")):
                raise FileNotFoundError(
                    f"adapter_model 파일이 없는 어댑터 경로입니다: {self.adapter_path}"
                )

        self.device = resolve_device()
        self.max_input_tokens = int(
            os.getenv("LOCAL_LLM_MAX_INPUT_TOKENS", "8192")
        )
        self.max_new_tokens = int(
            os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "2056")
        )
        self.trust_remote_code = (
            os.getenv("LOCAL_LLM_TRUST_REMOTE_CODE", "") == "1"
        )

        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            print(f"[MedCheck] Loading tokenizer: {self.model_path}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=self.trust_remote_code,
            )

            # 긴 검색 근거가 있으면 앞부분보다 최근 입력을 보존합니다.
            self._tokenizer.truncation_side = "left"

            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            if self.device in {"cuda", "mps"}:
                dtype = torch.float16
            else:
                dtype = torch.float32

            print(
                f"[MedCheck] Loading local LLM\n"
                f"  path   : {self.model_path}\n"
                f"  device : {self.device}\n"
                f"  dtype  : {dtype}"
            )

            dtype_argument = (
                {"dtype": dtype}
                if Version(transformers_version) >= Version("4.56")
                else {"torch_dtype": dtype}
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=self.trust_remote_code,
                **dtype_argument,
            )
            if self.adapter_path is not None:
                self._model = PeftModel.from_pretrained(
                    self._model,
                    str(self.adapter_path),
                    local_files_only=True,
                )
                print(f"[MedCheck] 어댑터 적용: {self.adapter_path}")

            self._model.to(self.device)
            self._model.eval()

        return self._model

    @staticmethod
    def _find_json_candidates(text: str) -> list[str]:
        """
        모델 출력에서 문자열 내부 중괄호를 무시하면서
        균형이 맞는 JSON 객체 후보를 추출합니다.
        """
        cleaned = re.sub(
            r"```(?:json)?\s*",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.replace("```", "").strip()

        candidates: list[str] = []

        start: int | None = None
        depth = 0
        in_string = False
        escaped = False

        for index, char in enumerate(cleaned):
            if start is None:
                if char == "{":
                    start = index
                    depth = 1
                    in_string = False
                    escaped = False
                continue

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

                if depth == 0 and start is not None:
                    candidates.append(cleaned[start:index + 1])
                    start = None

        # 중괄호가 완전히 닫히지 않은 경우에도 복구 라이브러리에 전달
        if not candidates:
            first = cleaned.find("{")
            if first >= 0:
                candidates.append(cleaned[first:])

        return candidates

    @staticmethod
    def _save_invalid_output(text: str) -> Path:
        """
        JSON 복구에 실패한 원본 출력을 로그에 저장합니다.
        """
        debug_dir = Path(
            os.getenv("MEDCHECK_DEBUG_DIR", "logs/invalid_json")
        )
        debug_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_path = debug_dir / f"invalid-{timestamp}.txt"
        output_path.write_text(text, encoding="utf-8")

        return output_path


    @staticmethod
    def _unwrap_single_key(payload: dict[str, Any]) -> dict[str, Any]:
            """
            모델이 스키마를 한 겹 더 감싸는 경우를 풀어줍니다.

            프롬프트에 넣는 JSON Schema의 title(예: ClinicalDecision)을
            모델이 최상위 키로 오해하는 사례가 있습니다.

            입력 예: {"clinical_decision": {"case_summary": "...", "abstain": false}}
            출력 예: {"case_summary": "...", "abstain": false}
            """
            WRAPPER_KEYS = {
                "clinical_decision", "clinicaldecision", "decision",
                "clarification_assessment", "clarificationassessment", "assessment",
                "verification_result", "verificationresult", "verification",
                "result", "output", "response", "data", "json",
            }


            if os.getenv("MEDCHECK_PRINT_DEBUG", "0") == "1":
                print(f"[DEBUG] payload keys: {list(payload.keys())}")
            if isinstance(payload, dict):
                for key in list(payload.keys()):
                    inner = payload[key]
                    if key.lower() in WRAPPER_KEYS and isinstance(inner, dict):
                        # 래퍼 밖에 남은 형제 필드도 병합합니다.
                        # 모델이 중괄호를 잘못 닫아 next_steps 등이
                        # 래퍼 바깥에 오는 사례가 있습니다.
                        #
                        # 입력 예: {"clinical_decision": {"red_flags": [...]},
                        #           "next_steps": [...], "abstain": false}
                        # 출력 예: {"red_flags": [...], "next_steps": [...],
                        #           "abstain": false}
                        merged = dict(inner)
                        for other_key, other_value in payload.items():
                            if other_key != key:
                                merged.setdefault(other_key, other_value)
                        return merged

            return payload




    @classmethod
    def _extract_json(cls, text: str) -> dict[str, Any]:
        """
        파싱 순서:
        1. 정상 JSON
        2. Python dict 형태
        3. json-repair를 통한 문법 복구
        """
        text = text.strip()

        if not text:
            raise LocalLLMError("모델 응답이 비어 있습니다.")

        errors: list[str] = []
        candidates = cls._find_json_candidates(text)

        for candidate_index, candidate in enumerate(candidates, start=1):
            # 1. 정상 JSON
            try:
                value = json.loads(candidate)

                if isinstance(value, dict):
                    return cls._unwrap_single_key(value)

                errors.append(
                    f"candidate {candidate_index}: "
                    "최상위 값이 JSON 객체가 아님"
                )
            except Exception as exc:
                errors.append(
                    f"candidate {candidate_index}, json.loads: {exc}"
                )

            # 2. Python dict처럼 작은따옴표를 사용한 경우
            try:
                value = ast.literal_eval(candidate)

                if isinstance(value, dict):
                    return cls._unwrap_single_key(value)
            except Exception as exc:
                errors.append(
                    f"candidate {candidate_index}, literal_eval: {exc}"
                )

            # 3. 누락된 쉼표, 따옴표, 닫는 괄호 등을 복구
            try:
                repaired_text = repair_json(candidate)
                value = json.loads(repaired_text)

                if isinstance(value, dict):
                    return cls._unwrap_single_key(value)
            except Exception as exc:
                errors.append(
                    f"candidate {candidate_index}, json_repair: {exc}"
                )

        output_path = cls._save_invalid_output(text)

        raise LocalLLMError(
            "모델 응답을 JSON으로 파싱하거나 복구하지 못했습니다.\n"
            f"원본 출력 저장 위치: {output_path}\n"
            f"파싱 오류: {' | '.join(errors[-6:])}"
        )

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if json_schema is not None:
            schema_text = json.dumps(
                json_schema,
                ensure_ascii=False,
                indent=2,
            )
            user_prompt = (
                f"{user_prompt}\n\n"
                "[반드시 준수할 JSON Schema]\n"
                f"{schema_text}\n\n"
                "설명이나 Markdown 없이 JSON 객체 하나만 출력하라."
            )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.05,
        }

        if temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": max(temperature, 0.01),
                    "top_p": 0.9,
                }
            )
        else:
            generation_kwargs.update(
                {
                    "do_sample": False,
                    "temperature": None,
                    "top_p": None,
                    "top_k": None,
                }
            )

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]
        
        output_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        
        if "</think>" in output_text:
            output_text = output_text.split("</think>", 1)[1].strip()
        

        if os.getenv("MEDCHECK_PRINT_RAW_OUTPUT", "0") == "1":
            print("\n========== RAW LLM OUTPUT ==========")
            print(output_text)
            print("====================================\n")

        return self._extract_json(output_text)

    def chat_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]

        return self.tokenizer.decode(
            output_ids[0, input_length:],
            skip_special_tokens=True,
        ).strip()


# 기존 query_cli.py와 app.py를 수정하지 않기 위한 호환용 별칭입니다.
OllamaClient = LocalTransformersClient
OllamaError = LocalLLMError
