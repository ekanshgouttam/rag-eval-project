import feedparser
import json
import urllib.parse
import os

import requests

categories = ["cs.CL", "cs.LG", "cs.IR"]
keywords = ["retrieval", "RAG", "embeddings", "gradient descent", "regression"]

cat_query = " OR ".join(f"cat:{c}" for c in categories)
kw_query = " OR ".join(f'(ti:"{k}" OR abs:"{k}")' for k in keywords)
search_query = f"({cat_query}) AND ({kw_query})"

print(search_query)

base_url = "http://export.arxiv.org/api/query"
params = {
    "search_query": search_query,
    "start": 0,
    "max_results": 50,
    "sortBy": "relevance",
    "sortOrder": "descending"
}

query_string = urllib.parse.urlencode(params)
full_url = f"{base_url}?{query_string}"

print(full_url)


print("Fetching from arXiv...")
response = requests.get(full_url)

if response.status_code != 200:
    raise RuntimeError(f"arXiv request failed: HTTP {response.status_code}")


print("HTTP status:", response.status_code)

feed = feedparser.parse(response.text)

if feed.bozo:
    raise RuntimeError(f"Failed to parse Atom feed: {feed.bozo_exception}")

if len(feed.entries) == 0:
    raise RuntimeError("Query succeeded but returned 0 papers — check search_query syntax.")

print(f"Got {len(feed.entries)} entries")


script_dir = os.path.dirname(os.path.abspath(__file__))       # .../rag-eval-project/src
project_root = os.path.dirname(script_dir)                     # .../rag-eval-project
raw_dir = os.path.join(project_root, "data", "raw")
os.makedirs(raw_dir, exist_ok=True)

paper_count = 0

for entry in feed.entries:
    arxiv_id = entry.id.split("/abs/")[-1]
    authors = [author.name for author in entry.authors]
    
    paper = {
        "arxiv_id": arxiv_id,
        "title": entry.title.strip().replace("\n", " "),
        "abstract": entry.summary.strip().replace("\n", " "),
        "authors": authors,
    }

    safe_id = arxiv_id.replace("/", "_")
    filepath = os.path.join(raw_dir, f"{safe_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(paper, f, indent=2, ensure_ascii=False)

    paper_count += 1

print(f"Saved {paper_count} papers to {raw_dir}")

