import numpy as np
from sentence_transformers import CrossEncoder
from hybrid_search import rrf_fusion, df

model = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query_text, shortlist_size=10, top_k=5):
    candidates = rrf_fusion(query_text, top_k=shortlist_size)

    pairs = []
    for c in candidates:
        abstract = df[df["arxiv_id"] == c["arxiv_id"]]["abstract"].values[0]
        pairs.append([query_text, f"{c['title']}. {abstract}"])

    scores = model.predict(pairs)

    results = []
    for c, score in zip(candidates, scores):
        results.append({
            "arxiv_id": c["arxiv_id"],
            "title": c["title"],
            "rerank_score": score
        })

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    results = rerank("hybrid retrieval and reranking for RAG systems")
    for r in results:
        print(r["rerank_score"], r["title"])