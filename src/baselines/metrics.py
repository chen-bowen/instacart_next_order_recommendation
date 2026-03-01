"""
Compute information-retrieval metrics for baseline comparison with SBERT.

Matches the metrics reported by sentence_transformers InformationRetrievalEvaluator:
Accuracy@k, Recall@k, MRR@10, NDCG@10, MAP@100.
"""

from __future__ import annotations


def _precision_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """
    Precision at k: fraction of top-k that are relevant.

    Args:
        relevant: Set of relevant product IDs.
        ranked: Ranked list of product IDs.
        k: Cut-off position.

    Returns:
        Precision at k (0.0 to 1.0).
    """
    if k <= 0:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for pid in top_k if pid in relevant)
    return hits / k


def _recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """
    Recall at k: fraction of relevant items found in top-k.

    Args:
        relevant: Set of relevant product IDs.
        ranked: Ranked list of product IDs.
        k: Cut-off position.

    Returns:
        Recall at k (0.0 to 1.0), or 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for pid in top_k if pid in relevant)
    return hits / len(relevant)


def _average_precision(relevant: set[str], ranked: list[str], k: int | None = None) -> float:
    """
    Average precision (for MAP): sum of P@j * rel(j) for j in 1..k, divided by min(|relevant|, k).

    Args:
        relevant: Set of relevant product IDs.
        ranked: Ranked list of product IDs.
        k: Optional cut-off; uses full list if None.

    Returns:
        Average precision (0.0 to 1.0), or 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0
    if k is not None:
        ranked = ranked[:k]
    score = 0.0
    num_hits = 0
    for j, pid in enumerate(ranked, start=1):
        if pid in relevant:
            num_hits += 1
            score += num_hits / j
    return score / min(len(relevant), len(ranked)) if ranked else 0.0


def _reciprocal_rank(relevant: set[str], ranked: list[str], k: int) -> float:
    """
    Reciprocal rank of first relevant item in top-k (0 if none).

    Args:
        relevant: Set of relevant product IDs.
        ranked: Ranked list of product IDs.
        k: Cut-off position.

    Returns:
        1/j where j is position of first relevant item, or 0.0.
    """
    for j, pid in enumerate(ranked[:k], start=1):
        if pid in relevant:
            return 1.0 / j
    return 0.0


def _ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """
    NDCG@k: DCG / IDCG. Relevance is binary (1 if in relevant, 0 else).

    Args:
        relevant: Set of relevant product IDs.
        ranked: Ranked list of product IDs.
        k: Cut-off position.

    Returns:
        NDCG at k (0.0 to 1.0).
    """
    def dcg(rel_list: list[float]) -> float:
        return sum(r / __lg(i + 2) for i, r in enumerate(rel_list))

    def __lg(x: float) -> float:
        import math
        return math.log2(x) if x > 0 else 0.0

    top_k = ranked[:k]
    rel_list = [1.0 if pid in relevant else 0.0 for pid in top_k]
    dcg_val = dcg(rel_list)
    ideal = sorted(rel_list, reverse=True)
    idcg_val = dcg(ideal)
    if idcg_val <= 0:
        return 0.0
    return dcg_val / idcg_val


def compute_ir_metrics(
    query_rankings: dict[str, list[str]],
    relevant_docs: dict[str, set[str]],
) -> dict[str, float]:
    """
    Compute IR metrics over all queries.

    Args:
        query_rankings: Dict mapping query_id to list of product_id (ranked by score desc).
        relevant_docs: Dict mapping query_id to set of relevant product_id.

    Returns:
        Dict with keys: accuracy_at_1, accuracy_at_3, accuracy_at_5, accuracy_at_10,
        recall_at_10, mrr_at_10, ndcg_at_10, map_at_100.
    """
    qids = [q for q in query_rankings if q in relevant_docs and relevant_docs[q]]
    if not qids:
        return {
            "accuracy_at_1": 0.0,
            "accuracy_at_3": 0.0,
            "accuracy_at_5": 0.0,
            "accuracy_at_10": 0.0,
            "recall_at_10": 0.0,
            "mrr_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "map_at_100": 0.0,
        }

    acc1 = sum(1 for q in qids if relevant_docs[q] & set(query_rankings[q][:1])) / len(qids)
    acc3 = sum(1 for q in qids if relevant_docs[q] & set(query_rankings[q][:3])) / len(qids)
    acc5 = sum(1 for q in qids if relevant_docs[q] & set(query_rankings[q][:5])) / len(qids)
    acc10 = sum(1 for q in qids if relevant_docs[q] & set(query_rankings[q][:10])) / len(qids)
    recall10 = sum(
        _recall_at_k(relevant_docs[q], query_rankings[q], 10) for q in qids
    ) / len(qids)
    mrr10 = sum(
        _reciprocal_rank(relevant_docs[q], query_rankings[q], 10) for q in qids
    ) / len(qids)
    ndcg10 = sum(
        _ndcg_at_k(relevant_docs[q], query_rankings[q], 10) for q in qids
    ) / len(qids)
    map100 = sum(
        _average_precision(relevant_docs[q], query_rankings[q], 100) for q in qids
    ) / len(qids)

    return {
        "accuracy_at_1": acc1,
        "accuracy_at_3": acc3,
        "accuracy_at_5": acc5,
        "accuracy_at_10": acc10,
        "recall_at_10": recall10,
        "mrr_at_10": mrr10,
        "ndcg_at_10": ndcg10,
        "map_at_100": map100,
    }
