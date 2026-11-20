"""Separate support-document retrieval from answer-claim support metrics."""

from __future__ import annotations


def diagnose_pipeline(
    required_doc_ids: set[str],
    retrieved_doc_ids: set[str],
    claims: list[str],
    supported_claim_ids: set[str],
) -> dict:
    if not required_doc_ids:
        raise ValueError("required_doc_ids must not be empty")
    if any(not isinstance(claim, str) or not claim for claim in claims):
        raise ValueError("claims must be nonempty string IDs")
    found = required_doc_ids & retrieved_doc_ids
    supported = set(claims) & supported_claim_ids
    return {
        "required_document_count": len(required_doc_ids),
        "retrieved_required_document_count": len(found),
        "retrieval_recall": len(found) / len(required_doc_ids),
        "all_required_documents_retrieved": found == required_doc_ids,
        "claim_count": len(claims),
        "supported_claim_count": len(supported),
        "answer_support_rate": len(supported) / len(claims) if claims else None,
        "unsupported_claim_ids": sorted(set(claims) - supported_claim_ids),
    }


def run(seed: int = 42) -> dict:
    docs = {"runbook-v2", "policy-v4"}
    claims = ["restart-action", "safe-window"]
    return {
        "evidence_type": "synthetic",
        "seed": seed,
        "diagnosis": diagnose_pipeline(docs, docs, claims, {"restart-action"}),
        "limitations": [
            "Document and claim IDs are a synthetic fixture; no retriever or LLM is executed.",
            "The support labels are assumed input and require independent human annotation in a real study.",
        ],
    }
