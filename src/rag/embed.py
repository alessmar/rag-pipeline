from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-small"  # English + Italian, 512 token limit


@lru_cache
def model() -> SentenceTransformer:
    return SentenceTransformer(MODEL)


# e5 models need these prefixes to embed documents and queries in the same space
def embed_passages(texts: list[str]) -> list[list[float]]:
    return model().encode([f"passage: {t}" for t in texts], normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return model().encode(f"query: {text}", normalize_embeddings=True).tolist()
