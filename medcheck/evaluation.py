from __future__ import annotations

import re
from collections import Counter


_OPTION_RE = re.compile(r"(?:^|\s)([1-9])\s*[\)\.]")
_HANGUL_LATIN_NUM = re.compile(r"[가-힣A-Za-z0-9]+")


def extract_option_number(text: str) -> str | None:
    match = _OPTION_RE.search(text or "")
    return match.group(1) if match else None


def normalized_tokens(text: str) -> list[str]:
    return [token.lower() for token in _HANGUL_LATIN_NUM.findall(text or "")]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalized_tokens(prediction)
    ref_tokens = normalized_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = normalized_tokens(prediction)
    ref = normalized_tokens(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    previous = [0] * (len(ref) + 1)
    for p in pred:
        current = [0]
        for j, r in enumerate(ref, start=1):
            if p == r:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current

    lcs = previous[-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
