import numpy as np
from retrieve import embed_query, cosine_similarity, doc_matrix, df
from bm25_search import tokenize, bm25, bm25_arxiv_ids

def get_full_ranking(scores, ids):
    order = np.argsort(scores)[::-1]
    ranked_ids = [ids[i] for i in order]
    return ranked_ids

def get_dense_ranking(query_text):
    query_vec = embed_query(query_text)
    scores = cosine_similarity(query_vec, doc_matrix)
    return get_full_ranking(scores, df["arxiv_id"].tolist())

def get_bm25_ranking(query_text):
    query_tokens = tokenize(query_text)
    scores = bm25.get_scores(query_tokens)
    return get_full_ranking(scores, bm25_arxiv_ids)


def rrf_fusion(query_text, k=60, top_k=5):
    dense_ranking = get_dense_ranking(query_text)   # list of arxiv_ids, rank 1 first
    bm25_ranking = get_bm25_ranking(query_text)      # same

    rrf_scores = {}

    for rank, arxiv_id in enumerate(dense_ranking, start=1):
        rrf_scores[arxiv_id] = rrf_scores.get(arxiv_id, 0) + 1 / (k + rank)

    for rank, arxiv_id in enumerate(bm25_ranking, start=1):
        rrf_scores[arxiv_id] = rrf_scores.get(arxiv_id, 0) + 1 / (k + rank)

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for arxiv_id, score in sorted_ids[:top_k]:
        title = df[df["arxiv_id"] == arxiv_id]["title"].values[0]
        results.append({"arxiv_id": arxiv_id, "title": title, "rrf_score": score})
    return results

if __name__ == "__main__":
    results = rrf_fusion("hybrid retrieval and reranking for RAG systems")
    for r in results:
        print(r["rrf_score"], r["title"])