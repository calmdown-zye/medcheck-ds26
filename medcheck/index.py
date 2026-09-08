from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .data import load_dataset, load_records_jsonl, save_records_jsonl
from .models import EvidenceItem, MedicalQARecord
from .tokenizer import korean_medical_tokenize


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_embedding_model(model_name: str) -> str:
    """번들된 임베딩 모델이 있으면 Hugging Face ID보다 우선합니다."""

    configured = os.getenv("MEDCHECK_EMBEDDING_MODEL", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())

    bundled = PROJECT_ROOT / "models" / model_name.rsplit("/", 1)[-1]
    return str(bundled) if bundled.is_dir() else model_name


@dataclass
class SearchResult:
    record_index: int
    score: float
    dense_rank: int | None
    bm25_rank: int | None


class MedicalHybridIndex:
    def __init__(
        self,
        records: list[MedicalQARecord],
        embeddings: np.ndarray,
        tokenized_corpus: list[list[str]],
        embedding_model_name: str,
    ) -> None:
        if len(records) != len(embeddings) or len(records) != len(tokenized_corpus):
            raise ValueError("records, embeddings, tokenized_corpus 길이가 일치하지 않습니다.")

        self.records = records
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.tokenized_corpus = tokenized_corpus
        self.embedding_model_name = embedding_model_name
        self._model: SentenceTransformer | None = None
        self._bm25 = BM25Okapi(tokenized_corpus)

    # @property
    # def model(self) -> SentenceTransformer:
    #     if self._model is None:
    #         self._model = SentenceTransformer(self.embedding_model_name)
    #     return self._model
    
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            offline = os.getenv("MEDCHECK_OFFLINE", "0") == "1"
            self._model = SentenceTransformer(
                resolve_embedding_model(self.embedding_model_name),
                device="cpu",
                local_files_only=offline,
            )
        return self._model

    @classmethod
    def build(
        cls,
        data_root: str | Path,
        output_dir: str | Path,
        *,
        embedding_model_name: str = "intfloat/multilingual-e5-small",
        batch_size: int = 64,
        strict: bool = False,
    ) -> "MedicalHybridIndex":
        records, errors = load_dataset(data_root, strict=strict)
        if not records:
            raise RuntimeError("인덱싱할 정상 레코드가 없습니다.")

        # 동일 파일 중첩 또는 압축 해제 중복에 대비한 보수적 중복 제거
        deduped: list[MedicalQARecord] = []
        seen: set[tuple[str, str, str]] = set()
        for record in records:
            key = (record.qa_id, record.question, record.answer)
            if key not in seen:
                seen.add(key)
                deduped.append(record)
        records = deduped

        model = SentenceTransformer(embedding_model_name)
        texts = [f"passage: {r.retrieval_text}" for r in records]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        tokenized_corpus = [
            korean_medical_tokenize(r.retrieval_text)
            for r in tqdm(records, desc="BM25 토큰화")
        ]

        index = cls(
            records=records,
            embeddings=embeddings,
            tokenized_corpus=tokenized_corpus,
            embedding_model_name=embedding_model_name,
        )
        index._model = model
        index.save(output_dir, errors=errors)
        return index

    def save(
        self,
        output_dir: str | Path,
        *,
        errors: list[dict[str, str]] | None = None,
    ) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        save_records_jsonl(self.records, output / "records.jsonl")
        np.save(output / "embeddings.npy", self.embeddings)

        with (output / "tokens.pkl").open("wb") as f:
            pickle.dump(self.tokenized_corpus, f, protocol=pickle.HIGHEST_PROTOCOL)

        config = {
            "embedding_model_name": self.embedding_model_name,
            "num_records": len(self.records),
            "embedding_dim": int(self.embeddings.shape[1]),
            "build_errors": errors or [],
        }
        with (output / "index_config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str | Path) -> "MedicalHybridIndex":
        root = Path(index_dir)
        records = load_records_jsonl(root / "records.jsonl")
        embeddings = np.load(root / "embeddings.npy", mmap_mode="r")

        with (root / "tokens.pkl").open("rb") as f:
            tokenized_corpus = pickle.load(f)

        with (root / "index_config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)

        return cls(
            records=records,
            embeddings=embeddings,
            tokenized_corpus=tokenized_corpus,
            embedding_model_name=config["embedding_model_name"],
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        dense_weight: float = 0.65,
        bm25_weight: float = 0.35,
        domain: str | None = None,
        candidate_k: int = 100,
        rrf_k: int = 60,
    ) -> list[EvidenceItem]:
        if not query.strip():
            raise ValueError("검색 질의가 비어 있습니다.")
        if top_k <= 0:
            raise ValueError("top_k는 1 이상이어야 합니다.")

        mask = np.ones(len(self.records), dtype=bool)
        if domain and domain != "전체":
            mask = np.array(
                [
                    domain in {r.domain_name, r.domain_code}
                    for r in self.records
                ],
                dtype=bool,
            )

        valid_indices = np.flatnonzero(mask)
        if len(valid_indices) == 0:
            return []

        query_embedding = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)

        dense_scores = np.asarray(self.embeddings[valid_indices]) @ query_embedding
        dense_order_local = np.argsort(-dense_scores)[:candidate_k]
        dense_order = valid_indices[dense_order_local]

        bm25_scores_all = np.asarray(
            self._bm25.get_scores(korean_medical_tokenize(query)),
            dtype=np.float32,
        )
        bm25_scores = bm25_scores_all[valid_indices]
        bm25_order_local = np.argsort(-bm25_scores)[:candidate_k]
        bm25_order = valid_indices[bm25_order_local]

        fused: dict[int, SearchResult] = {}

        for rank, idx in enumerate(dense_order, start=1):
            score = dense_weight / (rrf_k + rank)
            fused[int(idx)] = SearchResult(
                record_index=int(idx),
                score=score,
                dense_rank=rank,
                bm25_rank=None,
            )

        for rank, idx in enumerate(bm25_order, start=1):
            idx = int(idx)
            score = bm25_weight / (rrf_k + rank)
            if idx in fused:
                fused[idx].score += score
                fused[idx].bm25_rank = rank
            else:
                fused[idx] = SearchResult(
                    record_index=idx,
                    score=score,
                    dense_rank=None,
                    bm25_rank=rank,
                )

        ranked = sorted(fused.values(), key=lambda x: x.score, reverse=True)[:top_k]

        evidence: list[EvidenceItem] = []
        for position, item in enumerate(ranked, start=1):
            record = self.records[item.record_index] # 인덱싱된 원본 QA 레코드
            evidence.append(
                EvidenceItem(
                    evidence_id=f"E{position}", # 검색 순위대로 E1,E2,E3.. 
                    qa_id=record.qa_id,
                    domain=record.domain_name,
                    q_type=record.q_type_name,
                    question=record.question, # 원본 QA 의 질문
                    answer=record.answer, # 원본 QA의 답변 
                    source_file=record.source_file,
                    score=float(item.score),
                )
            )
        return evidence
