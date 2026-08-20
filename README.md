## Evaluation

### Methodology

I hand-labeled 13 queries against the 50-paper corpus, split across four categories designed to stress-test different aspects of retrieval:

- **Easy/lexical** (5 queries) — close paraphrases of a paper's core idea; a sanity-check baseline.
- **Semantic** (4 queries) — phrased to avoid a paper's exact title vocabulary, testing whether a method understands meaning rather than matching keywords.
- **Multi-candidate** (1 query) — a deliberately broad query with 5 relevant papers, testing whether a method retrieves a good *set*, not just one lucky hit.
- **Adversarial** (2 queries) — queries that exploit vocabulary overlap between unrelated topics in the corpus (e.g. "phase retrieval," a signal-processing concept, vs. "information retrieval"), and one query with **no relevant paper in the corpus at all**, testing whether a method stays appropriately uncertain rather than confidently returning a wrong answer.

One adversarial query (no valid answer by design) is excluded from the quantitative metrics below and discussed separately, since Recall@k/MRR aren't meaningful without a defined ground truth.

### Results (Recall@5 / MRR, averaged across 12 scorable queries)

| Method | Recall@5 | MRR |
|---|---|---|
| Dense (OpenAI `text-embedding-3-small`) | 0.883 | 0.806 |
| BM25 | 0.617 | 0.569 |
| RRF Fusion (dense + BM25) | 0.742 | 0.833 |
| Cross-encoder reranked (`bge-reranker-base`) | 0.908 | 0.799 |

### Findings

**1. RRF fusion is not strictly better than its best input — it inherits the failure mode of its weakest input on adversarial queries.**
On the phase-retrieval-vs-information-retrieval adversarial query, dense retrieval alone scored perfect recall, but RRF fusion scored **zero** — matching BM25's complete failure exactly. Rank-based fusion rewards *consensus* between methods, not *correctness* — when BM25 confidently and consistently misranks a query (due to literal keyword overlap between unrelated topics), fusion has no mechanism to discount that signal, even when a second, more accurate method disagrees. This is a real limitation of RRF, not just a corpus quirk: any production hybrid system needs a way to detect when one retriever is confidently wrong, not just average rankings together.

**2. Recall and MRR measure different things, and the gap between them is diagnostic, not just noise.**
Cross-encoder reranking hit perfect recall (1.0) on semantic queries but a lower MRR (0.52) on the same category — meaning every relevant paper was surfaced somewhere in the top 5, but not consistently ranked first. Whether this matters depends on downstream usage: for a RAG pipeline that reads all top-k retrieved documents into a generation prompt, recall is the more important signal; MRR matters more if the pipeline is constrained to very few documents or sensitive to position within the context window.

### The adversarial "no valid answer" case

One query — *"estimators that outperform baseline models in regression tasks"* — has no genuinely relevant paper in the corpus (verified by direct corpus search, not just by absence from top results). Rather than scoring this quantitatively, I compared each method's top result and confidence:

- BM25 and dense retrieval both confidently returned a pure optimization-theory paper as their top hit, with scores in line with their typical high-confidence results.
- The cross-encoder reranker's top score dropped to ~0.49 — notably lower than its typical 0.87–0.97 range on queries with genuine matches — suggesting its confidence score may be a usable signal for detecting "no good answer exists," even without an explicit abstention mechanism.

### Note: catching a ground-truth gap

While manually reviewing the multi-candidate query's results, I found a clearly relevant paper ("Improving Retrieval for RAG based Question Answering Models on Financial Documents") missing from the original ground truth — it appeared near the top of 3 of 4 methods despite not being labeled as relevant. After adding it and re-running the eval, all four methods converged to the same Recall@5 (0.500) on this query — but this was coincidental, not a sign of equal performance. Working out the underlying hit counts (numerator/6) showed Dense was the only method that failed to retrieve the newly-added paper; BM25, RRF, and reranking all picked it up, likely because the paper's title shared substantial literal vocabulary with the query ("retrieval," "RAG") — a case where BM25's usual weakness (literal term matching) became an advantage. This is a useful reminder that manually-labeled eval sets are only as good as the labeling effort behind them, and that identical aggregate scores can mask real differences underneath.