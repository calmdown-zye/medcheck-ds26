from __future__ import annotations

import re

_WORD_RE = re.compile(r"[가-힣]+|[A-Za-z]+|\d+(?:\.\d+)?|[^\s]")


def korean_medical_tokenize(text: str) -> list[str]:
    """
    Java/KoNLPy 의존성 없이 동작하는 경량 토크나이저입니다.

    - 공백 기반 단어
    - 영문 소문자화
    - 긴 한글 토큰은 2-gram을 추가하여 조사/활용 변화에 대한 검색 재현율을 보완
    """
    tokens: list[str] = []
    for raw in _WORD_RE.findall(text or ""):
        token = raw.lower()
        tokens.append(token)

        if re.fullmatch(r"[가-힣]+", token) and len(token) >= 3:
            tokens.extend(token[i:i + 2] for i in range(len(token) - 1))

    return tokens
