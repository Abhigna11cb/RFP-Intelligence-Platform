"""
Expanded evaluation set — 15+ Q&A pairs covering both bids as required by Part B 6.4.

Metrics reported:
  - Recall@5:  Was the correct passage in top-5?
  - Recall@10: Was the correct passage in top-10?
  - MRR:       Mean Reciprocal Rank (average 1/rank of first correct hit)

Configurations compared:
  1. vector_only   — text-embedding-3-large cosine similarity
  2. keyword_only  — PostgreSQL tsvector BM25
  3. hybrid        — RRF of vector + BM25
  4. hybrid_rerank — RRF + cross-encoder re-rank (ms-marco-MiniLM-L-6-v2)

Run:
    python -m rfp_platform.eval.evaluate
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── 15+ Ground-truth eval questions ──────────────────────────────────────────
# Covers both bids, multiple doc types, exact values and conceptual queries.

EVAL_QUESTIONS = [
    # ─── BID 1 — JA-207652 Student & Staff Computing Devices ───────────────
    {
        "id": "B1-01",
        "bid_id": "Bid1",
        "question": "What is the official bid number for the student computing devices RFP?",
        "expected_terms": ["JA-207652"],
        "field": "Bid Number",
    },
    {
        "id": "B1-02",
        "bid_id": "Bid1",
        "question": "What is the final submission deadline including addendum extensions?",
        "expected_terms": ["July 9", "2024", "14:00"],
        "field": "Due Date",
    },
    {
        "id": "B1-03",
        "bid_id": "Bid1",
        "question": "Is a bid bond or security deposit required for Bid1?",
        "expected_terms": ["bond", "security"],
        "field": "Bid Bond Requirement",
    },
    {
        "id": "B1-04",
        "bid_id": "Bid1",
        "question": "What Dell laptop model is specified in the student computing devices bid?",
        "expected_terms": ["Latitude", "CC7802"],
        "field": "Model_no",
    },
    {
        "id": "B1-05",
        "bid_id": "Bid1",
        "question": "What changed in Addendum 2 compared to the original RFP?",
        "expected_terms": ["addendum", "due date", "deadline"],
        "field": "Due Date",
    },
    {
        "id": "B1-06",
        "bid_id": "Bid1",
        "question": "What are the payment terms for Bid1?",
        "expected_terms": ["net 30", "payment", "invoice"],
        "field": "Payment Terms",
    },
    {
        "id": "B1-07",
        "bid_id": "Bid1",
        "question": "Is there a pre-bid conference or meeting required for Bid1?",
        "expected_terms": ["pre-bid", "conference", "meeting"],
        "field": "Pre Bid Meeting",
    },
    {
        "id": "B1-08",
        "bid_id": "Bid1",
        "question": "Who is the procurement contact for the student computing devices bid?",
        "expected_terms": ["contact", "email", "phone", "procurement"],
        "field": "contact_info",
    },
    {
        "id": "B1-09",
        "bid_id": "Bid1",
        "question": "What additional documentation or affidavits are required for Bid1?",
        "expected_terms": ["affidavit", "certificate", "insurance", "W-9"],
        "field": "Any Additional Documentation Required",
    },
    {
        "id": "B1-10",
        "bid_id": "Bid1",
        "question": "What is the contract term or duration for Bid1?",
        "expected_terms": ["term", "year", "renewal"],
        "field": "Term of Bid",
    },

    # ─── BID 2 — E20P4600040 Dell Laptops w/ Extended Warranty ─────────────
    {
        "id": "B2-01",
        "bid_id": "Bid2",
        "question": "What is the solicitation number for the Dell laptop bid?",
        "expected_terms": ["E20P4600040"],
        "field": "Bid Number",
    },
    {
        "id": "B2-02",
        "bid_id": "Bid2",
        "question": "What is the submission deadline for the Dell laptop bid?",
        "expected_terms": ["June", "2024", "deadline"],
        "field": "Due Date",
    },
    {
        "id": "B2-03",
        "bid_id": "Bid2",
        "question": "What are the technical specifications for the Dell laptops including CPU, RAM and storage?",
        "expected_terms": ["CPU", "RAM", "SSD", "storage"],
        "field": "Product Specification",
    },
    {
        "id": "B2-04",
        "bid_id": "Bid2",
        "question": "What affidavits are required for the Dell laptop bid submission?",
        "expected_terms": ["affidavit", "contract", "Mercury"],
        "field": "Any Additional Documentation Required",
    },
    {
        "id": "B2-05",
        "bid_id": "Bid2",
        "question": "What is the issuing organization or agency for the Dell laptop RFP?",
        "expected_terms": ["county", "district", "agency", "department"],
        "field": "company_name",
    },
    {
        "id": "B2-06",
        "bid_id": "Bid2",
        "question": "What is the part number or SKU for the Dell laptop?",
        "expected_terms": ["part", "SKU", "210-"],
        "field": "Part_no",
    },
    {
        "id": "B2-07",
        "bid_id": "Bid2",
        "question": "How should bids be submitted for the Dell laptop procurement?",
        "expected_terms": ["submit", "portal", "email", "sealed", "electronic"],
        "field": "Bid Submission Type",
    },
]


# ── Metric helpers ────────────────────────────────────────────────────────────

def _recall_at_k(results: list[dict], expected_terms: list[str], k: int) -> float:
    for r in results[:k]:
        text = r.get("chunk_text", "").lower()
        if any(term.lower() in text for term in expected_terms):
            return 1.0
    return 0.0


def _mrr(results: list[dict], expected_terms: list[str]) -> float:
    for i, r in enumerate(results):
        text = r.get("chunk_text", "").lower()
        if any(term.lower() in text for term in expected_terms):
            return 1.0 / (i + 1)
    return 0.0


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_evaluation(top_k: int = 10, include_reranker: bool = True) -> dict[str, Any]:
    """
    Run the full evaluation benchmark across all retrieval configurations.

    Returns:
        {
          "vector_only":   {"recall@5": x, "recall@10": x, "mrr": x},
          "keyword_only":  {"recall@5": x, "recall@10": x, "mrr": x},
          "hybrid":        {"recall@5": x, "recall@10": x, "mrr": x},
          "hybrid_rerank": {"recall@5": x, "recall@10": x, "mrr": x},
          "n_questions": n,
          "per_question": [...],
        }
    """
    from rfp_platform.search.tools import semantic_search, keyword_search, hybrid_search
    from rfp_platform.search.reranker import rerank

    configs: dict[str, Any] = {
        "vector_only":  lambda q, bid: semantic_search(q, bid_id=bid, top_k=top_k),
        "keyword_only": lambda q, bid: keyword_search(q, bid_id=bid, top_k=top_k),
        "hybrid":       lambda q, bid: hybrid_search(q, bid_id=bid, top_k=top_k),
    }

    if include_reranker:
        configs["hybrid_rerank"] = lambda q, bid: rerank(
            q, hybrid_search(q, bid_id=bid, top_k=20), top_k=top_k
        )

    totals = {name: {"recall_5": 0.0, "recall_10": 0.0, "mrr": 0.0}
              for name in configs}

    per_question = []

    for q in EVAL_QUESTIONS:
        question = q["question"]
        bid_id   = q["bid_id"]
        terms    = q["expected_terms"]
        row = {"id": q["id"], "question": question[:60], "bid_id": bid_id}

        for config_name, search_fn in configs.items():
            try:
                results = search_fn(question, bid_id)
                r5  = _recall_at_k(results, terms, 5)
                r10 = _recall_at_k(results, terms, 10)
                mrr = _mrr(results, terms)
                totals[config_name]["recall_5"]  += r5
                totals[config_name]["recall_10"] += r10
                totals[config_name]["mrr"]       += mrr
                row[config_name] = {"recall@5": r5, "recall@10": r10, "mrr": round(mrr, 3)}
            except Exception as e:
                logger.error(f"Eval [{config_name}] '{question[:40]}': {e}")
                row[config_name] = {"error": str(e)}

        per_question.append(row)

    n = len(EVAL_QUESTIONS)
    summary = {}
    for name, scores in totals.items():
        summary[name] = {
            "recall@5":  round(scores["recall_5"]  / n, 3),
            "recall@10": round(scores["recall_10"] / n, 3),
            "mrr":       round(scores["mrr"]       / n, 3),
        }

    summary["n_questions"] = n
    summary["per_question"] = per_question
    return summary


def print_eval_report(results: dict) -> None:
    n = results["n_questions"]
    print("\n" + "=" * 70)
    print("  RFP Platform — Retrieval Evaluation Report")
    print(f"  Questions: {n}  |  Both Bids")
    print("=" * 70)
    print(f"\n  {'Config':<20} {'Recall@5':>10} {'Recall@10':>10} {'MRR':>8}")
    print("  " + "-" * 52)

    configs_order = ["vector_only", "keyword_only", "hybrid", "hybrid_rerank"]
    best_mrr = max(
        results[c]["mrr"] for c in configs_order if c in results
    )
    for name in configs_order:
        if name not in results:
            continue
        r = results[name]
        marker = "  <-- BEST" if r["mrr"] == best_mrr else ""
        print(f"  {name:<20} {r['recall@5']:>10.3f} {r['recall@10']:>10.3f} {r['mrr']:>8.3f}{marker}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    print("Running retrieval evaluation (17 questions, 4 configs)...")
    print("This will take ~30 seconds (cross-encoder re-ranking included).\n")
    results = run_evaluation()
    print_eval_report(results)
    # Save full results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Full results saved to eval_results.json")
