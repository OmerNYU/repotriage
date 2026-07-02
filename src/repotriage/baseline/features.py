"""TF-IDF feature extraction for baseline training."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from repotriage.baseline.models import BaselineCandidateConfig, TfidfParams


def build_vectorizer(params: TfidfParams) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=params.analyzer,
        ngram_range=params.ngram_range,
        lowercase=params.lowercase,
        min_df=params.min_df,
        sublinear_tf=params.sublinear_tf,
        norm=params.norm,
    )


def fit_vectorizer(vectorizer: TfidfVectorizer, train_texts: list[str]) -> TfidfVectorizer:
    if not train_texts:
        raise ValueError("train_texts must not be empty")
    vectorizer.fit(train_texts)
    return vectorizer


def transform_texts(vectorizer: TfidfVectorizer, texts: list[str]):
    return vectorizer.transform(texts)


def candidate_vectorizer(candidate: BaselineCandidateConfig) -> TfidfVectorizer:
    return build_vectorizer(candidate.tfidf)
