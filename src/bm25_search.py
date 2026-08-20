import os 
import json
import re
import numpy as np
from rank_bm25 import BM25Okapi


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
raw_dir = os.path.join(project_root, "data", "raw")

papers = []
for filename in os.listdir(raw_dir):
    filepath = os.path.join(raw_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        papers.append(json.load(f))

print(f"Loaded {len(papers)} papers from {raw_dir}")



def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

tokenized_corpus = []
bm25_arxiv_ids = []

for paper in papers:
    title = paper["title"].rstrip(".!?:")
    combined = f"{title}. {paper['abstract']}"
    tokens = tokenize(combined)
    tokenized_corpus.append(tokens)
    bm25_arxiv_ids.append(paper["arxiv_id"])

bm25 = BM25Okapi(tokenized_corpus)

# print(tokenized_corpus[0])

def bm25_search(query_text, top_k=5):
    query_tokens = tokenize(query_text)
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "arxiv_id": bm25_arxiv_ids[idx],
            "title": papers[idx]["title"],
            "score": scores[idx]
        })
    return results
# results = bm25_search("hybrid retrieval and reranking for RAG systems")
# for r in results:
#     print(r["score"], r["title"])


if __name__ == "__main__":
    results = bm25_search("hybrid retrieval and reranking for RAG systems")
    for r in results:
        print(r["score"], r["title"])
