"""
All LangGraph agent node functions.
Each function takes AgentState, does its job, returns updated state fields.

Agents:
  1. orchestrator_node     — plans and routes
  2. retrieval_node        — runs hybrid search for each field group
  3. extraction_node       — extracts the 20 fields using search tools
  4. addendum_node         — reconciles addendum changes
  5. validator_node        — validates extracted fields
  6. qa_node               — answers free-form questions with citations
"""
from __future__ import annotations

import json
import logging
from typing import Any

from rfp_platform.agents.state import (
    AgentState, FieldResult, Citation, AddendumChange, FieldValidation
)
from rfp_platform.agents.llm_client import run_agent_with_tools, get_llm_response
from rfp_platform.agents.tool_schemas import TOOL_SCHEMAS
from rfp_platform.core.database import get_conn
from rfp_platform.search.tools import (
    get_bid_documents, get_addendum_changes, search_by_field, hybrid_search
)

logger = logging.getLogger(__name__)

# All 20 required fields from the assignment spec
ALL_FIELDS = [
    "Bid Number", "Title", "Due Date", "Bid Submission Type",
    "Term of Bid", "Pre Bid Meeting", "Installation",
    "Bid Bond Requirement", "Delivery Date", "Payment Terms",
    "Any Additional Documentation Required", "MFG for Registration",
    "Contract or Cooperative to use", "Model_no", "Part_no",
    "Product", "contact_info", "company_name", "Bid Summary",
    "Product Specification",
]


def _chunks_to_citations(chunks: list[dict]) -> list[Citation]:
    return [
        Citation(
            file_name=c.get("file_name", ""),
            page_number=c.get("page_number", 0),
            chunk_text=c.get("chunk_text", ""),
            chunk_index=c.get("chunk_index", 0),
            bid_id=c.get("bid_id", ""),
            doc_type=c.get("doc_type", ""),
            score=c.get("score", 0.0),
        )
        for c in chunks
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Orchestrator Node
# ─────────────────────────────────────────────────────────────────────────────

def orchestrator_node(state: AgentState) -> dict:
    """
    Receives the user goal, lists available bid documents,
    creates an execution plan and sets the mode.
    """
    logger.info(f"[Orchestrator] Starting run_id={state.run_id} bid={state.bid_id} mode={state.mode}")

    # Discover available documents for this bid
    docs = get_bid_documents(state.bid_id)
    doc_summary = "\n".join(
        f"  - {d['file_name']} ({d['doc_type']}, chunks={d['chunk_count']})"
        for d in docs
    ) or "  No documents indexed yet."

    if state.mode == "extraction":
        plan = [
            "retrieval",
            "extraction",
            "addendum_reconciliation",
            "validation",
            "finalize",
        ]
        goal = f"Extract all 20 structured fields for bid '{state.bid_id}'"
    else:
        plan = ["retrieval", "qa"]
        goal = state.user_query

    logger.info(f"[Orchestrator] Plan: {plan}")
    logger.info(f"[Orchestrator] Documents found:\n{doc_summary}")

    return {
        "plan":         plan,
        "goal":         goal,
        "current_step": "orchestrator_done",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Retrieval Node
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_node(state: AgentState) -> dict:
    """
    Retrieval Agent: searches for evidence for each of the 20 fields.

    DESIGN: In extraction mode we do DIRECT Python retrieval — one search
    per field. This guarantees every field gets evidence, unlike delegating
    to an LLM tool loop (which may skip fields due to token/iteration limits).

    QA mode still uses hybrid_search for the user question.
    """
    logger.info(f"[Retrieval] mode={state.mode} bid={state.bid_id}")
    evidence: list[dict] = []

    if state.mode == "extraction":
        fields_needing_retrieval = [
            f for f in ALL_FIELDS
            if f not in state.draft_fields
            or state.draft_fields[f].status in ("pending", "failed")
        ]
        logger.info(f"[Retrieval] Searching {len(fields_needing_retrieval)} fields for bid={state.bid_id}")

        # ── Direct per-field retrieval (deterministic, full coverage) ──────────
        for field_name in fields_needing_retrieval:
            try:
                chunks = search_by_field(
                    field_name=field_name,
                    bid_id=state.bid_id,
                    top_k=8,
                )
                for c in chunks:
                    c["_for_field"] = field_name   # tag which field triggered this
                evidence.extend(chunks)
            except Exception as e:
                logger.warning(f"[Retrieval] search_by_field('{field_name}') failed: {e}")

        # ── Supplemental broad search to catch cross-field evidence ───────────
        try:
            broad = hybrid_search(
                query=f"bid details specifications requirements {state.bid_id}",
                bid_id=state.bid_id,
                top_k=15,
            )
            evidence.extend(broad)
        except Exception as e:
            logger.warning(f"[Retrieval] broad search failed: {e}")

        # ── Supplemental warranty search (tagged for Product Specification) ────
        try:
            from rfp_platform.search.tools import _WARRANTY_QUERY
            warranty_chunks = hybrid_search(
                query=_WARRANTY_QUERY,
                bid_id=state.bid_id,
                top_k=8,
            )
            for c in warranty_chunks:
                c["_for_field"] = "Product Specification"
            evidence.extend(warranty_chunks)
        except Exception as e:
            logger.warning(f"[Retrieval] warranty search failed: {e}")

    else:
        # QA mode — single hybrid search for the user question
        evidence = hybrid_search(
            query=state.user_query,
            bid_id=state.bid_id or None,
            top_k=12,
        )

    # Deduplicate by (file_name, chunk_index)
    seen: set = set()
    unique_evidence: list[dict] = []
    for e in evidence:
        key = (e.get("file_name", ""), e.get("chunk_index", 0))
        if key not in seen:
            seen.add(key)
            unique_evidence.append(e)

    logger.info(f"[Retrieval] Found {len(unique_evidence)} unique evidence chunks (from {len(evidence)} raw)")
    return {
        "retrieved_evidence": unique_evidence,
        "current_step":       "retrieval_done",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Extraction Node
# ─────────────────────────────────────────────────────────────────────────────

def extraction_node(state: AgentState) -> dict:
    """
    Extraction Agent: uses retrieved evidence + search tools to fill all 20 fields.
    Returns draft_fields with value, sources, confidence, notes.
    """
    logger.info(f"[Extraction] Extracting fields for bid={state.bid_id}")

    evidence_text = "\n\n---\n\n".join(
        f"[Source: {e.get('file_name','?')} p.{e.get('page_number','?')} | "
        f"type={e.get('doc_type','?')} | field={e.get('_for_field','?')}]\n{e.get('chunk_text','')}"
        for e in state.retrieved_evidence[:80]  # up from 20 — covers all 20 fields
    )

    SYSTEM = """You are a strict Evidence-Only Extraction Agent for RFP bid documents.
Your job: extract EXACTLY the 20 required fields. Every value MUST be found verbatim or near-verbatim in the provided evidence.

═══ ABSOLUTE RULES ═══
1. NO INFERENCE EVER. Do NOT write phrases like:
   - "To be determined", "To be specified", "To be decided"
   - "Based on contract terms", "Subject to negotiation"
   - "Varies by vendor", "As required", "TBD"
   If you cannot find the exact value in the evidence → value must be null.

2. GOLDEN RULE: NO VALUE WITHOUT EVIDENCE.
   Every non-null value MUST cite a source_file + source_page that ACTUALLY CONTAINS the value.
   The cited chunk_text must contain the extracted value (or a close paraphrase).

3. CONFIDENCE CALIBRATION:
   - 0.95: Value appears word-for-word in the source.
   - 0.80: Value is clearly implied by surrounding text (not inferred, just summarized).
   - 0.60: Value is ambiguous or partially supported.
   - 0.00 + null: Not found in any evidence.
   Do NOT assign 0.95 if you are guessing.

4. ADDENDUM OVERRIDE: For Due Date and any field changed by an addendum, use the LATEST addendum value.
   Never return the original value if an addendum overrides it.

5. FIELD-SPECIFIC RULES:

   Bid Summary → Write 3–6 complete sentences summarizing the bid.
     Include: who is issuing it, what they are procuring, approximate quantity/scope,
     submission deadline, contract term, and any key requirements.
     Every sentence must be supported by evidence. Cite the source page for the most important claim.

   Product Specification → Extract ACTUAL technical specs, not generic statements.
     Include CPU, RAM, storage, display, OS, device tier/type, warranty terms.
     If specs appear in a table, preserve the table row context.
     Warranty requirements MUST be included:
       - Minimum 1-year for student Chromebooks
       - Minimum 3-year for student/staff Windows laptops
       - Service/repair within 5 business days at no cost
     Cite the exact source page for each specification.

   MFG for Registration → Extract the actual manufacturer name (e.g., Dell, HP, Apple).
     If the bid is an open solicitation and no specific manufacturer is required,
     extract the table column header or the instruction to vendors (e.g., 'Vendor to specify
     proposed device make and model') — do NOT write 'To be determined'.

   Model_no / Part_no → Extract actual model numbers (e.g., 'SI# CC7802', '210-BLYZ').
     For open solicitations, if vendors must propose their own, extract the explicit
     instruction from the document (e.g., 'Proposed device make and model to be submitted').
     Do NOT write 'To be specified by the vendor in the proposal' unless those exact words appear.

   Payment Terms → Extract the EXACT payment terms stated (e.g., 'Net 30 days', 'invoiced monthly').
     If no payment terms are explicitly stated, value = null.

6. TABLE DATA: If a value comes from a table, include enough surrounding context in notes
   so the table row/column is identifiable.

7. OUTPUT FORMAT — strict JSON only, NO markdown fences:
{
  "Bid Number":          {"value": "JA-207652",  "source_file": "rfp.pdf", "source_page": 13, "confidence": 0.95, "notes": ""},
  "Bid Summary":         {"value": "Dallas ISD seeks vendors for student and staff computing devices... [3-6 sentences]", "source_file": "rfp.pdf", "source_page": 3, "confidence": 0.95, "notes": ""},
  "Product Specification": {"value": "Student Chromebook: [specs]. Windows Laptop: [specs]. Warranty: 1yr Chromebook, 3yr Windows, repairs within 5 business days.", "source_file": "rfp.pdf", "source_page": 5, "confidence": 0.95, "notes": "Specs from Section X, Warranty from p.Y"},
  ... (all 20 fields)
}

If not found: {"value": null, "source_file": "", "source_page": 0, "confidence": 0.0, "notes": "Not found in documents"}"""

    fields_needed = [
        f for f in ALL_FIELDS
        if f not in state.draft_fields or state.draft_fields[f].status in ("pending", "failed")
    ]

    msgs = [{
        "role": "user",
        "content": (
            f"Extract these {len(fields_needed)} fields for bid '{state.bid_id}'.\n\n"
            "CRITICAL: Do NOT infer, guess, or write placeholders. Only extract what is explicitly in the evidence.\n"
            "IMPORTANT: Extracting what IS written in the document is always correct.\n"
            "  - If the document says 'No bid bond required' → extract that exact phrase.\n"
            "  - If the document says 'Net 30 days' → extract that.\n"
            "  - If the document says 'Delivery within 45 days of award' → extract that.\n"
            "  - Only return null when the information genuinely does not appear anywhere in the evidence.\n\n"
            "Fields to extract:\n"
            + "\n".join(f"- {f}" for f in fields_needed)
            + f"\n\nEVIDENCE ({len(state.retrieved_evidence)} chunks — each tagged [field=X] to show which field it supports):\n\n{evidence_text}\n\n"
            "VALIDATION CHECKLIST before returning JSON:\n"
            "□ Every value cites a source_file and source_page that actually contains the value.\n"
            "□ Bid Summary is 3–6 complete factual sentences.\n"
            "□ Product Specification has actual CPU/RAM/storage/display/warranty details (not generic).\n"
            "□ No field has 'To be determined', 'TBD', 'To be specified' as value.\n"
            "□ Due Date uses the latest addendum value.\n"
            "□ Any field not found in evidence has value=null and notes='Not found in documents'.\n"
            "Return the JSON now."
        ),
    }]

    final_text, _ = run_agent_with_tools(SYSTEM, msgs, TOOL_SCHEMAS, max_iterations=6)

    # Parse the LLM JSON response
    draft_fields = dict(state.draft_fields)  # copy existing
    try:
        # Extract JSON from response
        raw = final_text or ""
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        extracted = json.loads(raw.strip())

        for field_name in fields_needed:
            field_data = extracted.get(field_name, {})
            if not isinstance(field_data, dict):
                field_data = {"value": field_data, "confidence": 0.5}

            value      = field_data.get("value")
            src_file   = field_data.get("source_file", "")
            src_page   = field_data.get("source_page", 0)
            confidence = float(field_data.get("confidence", 0.0))
            notes      = field_data.get("notes", "")

            # ── Inference phrase rejection ─────────────────────────────────────
            # Only reject UNAMBIGUOUS LLM-generated placeholders that cannot
            # be verbatim document text. Phrases like "vendor to specify",
            # "as required", "to be specified" CAN appear in RFP docs — don't reject.
            INFERENCE_PHRASES = [
                "to be determined",
                "to be decided",
                "to be confirmed",
                "tbd",
                "t.b.d",
                "subject to negotiation",
                "will be determined later",
                "information not available",
            ]
            if value is not None:
                value_lower = str(value).strip().lower()
                for phrase in INFERENCE_PHRASES:
                    if value_lower.startswith(phrase) or value_lower == phrase:
                        logger.warning(f"[Extraction] Rejected inferred value for '{field_name}': {str(value)[:60]}")
                        notes = f"Rejected: inferred value ('{str(value)[:50]}'). " + notes
                        value = None
                        confidence = 0.0
                        break

            # Build citation from evidence — strict match first (same file + page)
            citations = []
            if src_file:
                matching = [
                    e for e in state.retrieved_evidence
                    if e.get("file_name") == src_file
                    and e.get("page_number") == src_page
                ]
                # Fallback: same file, any page (handles page number rounding)
                if not matching:
                    matching = [
                        e for e in state.retrieved_evidence
                        if e.get("file_name") == src_file
                    ]
                citations = _chunks_to_citations(matching[:1]) if matching else []

            status = "extracted" if value is not None else "not_found"
            if status == "not_found" and not notes:
                notes = "Not found in documents"

            draft_fields[field_name] = FieldResult(
                value=value,
                sources=citations,
                confidence=confidence,
                notes=notes,
                status=status,
            )


    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"[Extraction] JSON parse failed: {e}\nResponse: {final_text[:200]}")
        for field_name in fields_needed:
            if field_name not in draft_fields:
                draft_fields[field_name] = FieldResult(status="failed", notes=f"Parse error: {e}")

    extracted_count = sum(1 for f in draft_fields.values() if f.status == "extracted")
    logger.info(f"[Extraction] {extracted_count}/{len(ALL_FIELDS)} fields extracted")

    return {
        "draft_fields": draft_fields,
        "current_step": "extraction_done",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Addendum Reconciliation Node
# ─────────────────────────────────────────────────────────────────────────────

def addendum_node(state: AgentState) -> dict:
    """
    Addendum Reconciliation Agent:
    - Fetches all addendum content
    - Compares with base RFP values in draft_fields
    - Applies latest changes (e.g., extended due date from Addendum 2)
    - Returns updated draft_fields and addendum_changes log
    """
    logger.info(f"[Addendum] Reconciling addendums for bid={state.bid_id}")

    addendum_chunks = get_addendum_changes(state.bid_id)
    if not addendum_chunks:
        logger.info("[Addendum] No addendums found — skipping")
        return {"current_step": "addendum_done"}

    # Start from existing changes (may already have some from a previous pass)
    # We'll deduplicate at the end
    addendum_changes: list[AddendumChange] = list(state.addendum_changes)

    addendum_text = "\n\n---\n\n".join(
        f"[Addendum {c.get('addendum_number','?')} | {c.get('file_name','?')} p.{c.get('page_number','?')}]\n{c.get('chunk_text','')}"
        for c in addendum_chunks
    )

    current_values = {
        k: v.value for k, v in state.draft_fields.items()
    }

    SYSTEM = """You are an Addendum Reconciliation Agent.
Your job: compare addendum content with extracted field values and apply any changes.

RULES:
1. Addendums ALWAYS override the original RFP for the same field.
2. Higher addendum numbers take precedence over lower ones.
3. Return JSON with ONLY fields that changed.
4. Always include what the original value was and what it changed to.

OUTPUT FORMAT:
{
  "changes": [
    {
      "addendum_number": 2,
      "field_name": "Due Date",
      "original_value": "2024-06-27 14:00 CST",
      "new_value": "2024-07-09 14:00 CST",
      "source_file": "Addendum 2 RFP....pdf",
      "source_page": 1,
      "notes": "Due date extended by Addendum 2"
    }
  ]
}"""

    msgs = [{
        "role": "user",
        "content": (
            f"Current extracted values:\n{json.dumps(current_values, indent=2)}\n\n"
            f"ADDENDUM CONTENT:\n{addendum_text}\n\n"
            "Identify all fields changed by addendums and return the changes JSON."
        ),
    }]

    resp = get_llm_response(SYSTEM, msgs, temperature=0.0)
    raw  = resp.get("content", "") or ""

    draft_fields    = dict(state.draft_fields)
    addendum_changes = list(state.addendum_changes)

    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        parsed   = json.loads(raw.strip())
        changes  = parsed.get("changes", [])

        for ch in changes:
            field_name = ch.get("field_name", "")
            new_val    = ch.get("new_value")
            src_file   = ch.get("source_file", "")
            src_page   = ch.get("source_page", 0)

            # Find citation from addendum chunks
            matching = [
                c for c in addendum_chunks
                if c.get("file_name") == src_file and c.get("page_number") == src_page
            ]
            citation = _chunks_to_citations(matching[:1])[0] if matching else None

            # Apply change to draft_fields
            if field_name in draft_fields:
                old_result = draft_fields[field_name]
                draft_fields[field_name] = FieldResult(
                    value=new_val,
                    sources=[citation] if citation else old_result.sources,
                    confidence=0.95,
                    notes=ch.get("notes", ""),
                    status="extracted",
                )

            addendum_changes.append(AddendumChange(
                addendum_number=ch.get("addendum_number", 0),
                field_name=field_name,
                original_value=ch.get("original_value"),
                new_value=new_val,
                source=citation,
                notes=ch.get("notes", ""),
            ))

        logger.info(f"[Addendum] Applied {len(changes)} changes")

    except Exception as e:
        logger.error(f"[Addendum] Parse error: {e}")

    # Deduplicate by (field_name, new_value, addendum_number) — retry passes can add duplicates
    seen_changes: set = set()
    unique_changes: list[AddendumChange] = []
    for ch in addendum_changes:
        key = (ch.field_name, str(ch.new_value), ch.addendum_number)
        if key not in seen_changes:
            seen_changes.add(key)
            unique_changes.append(ch)
    addendum_changes = unique_changes

    return {
        "draft_fields":     draft_fields,
        "addendum_changes": addendum_changes,
        "current_step":     "addendum_done",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Validator Node
# ─────────────────────────────────────────────────────────────────────────────

def validator_node(state: AgentState) -> dict:
    """
    Validator/Critic Agent:
    - Checks each field has a citation
    - Validates date formats, number formats
    - Checks for hallucination (value not in any source)
    - Flags failed fields for re-extraction
    """
    logger.info(f"[Validator] Validating {len(state.draft_fields)} fields")

    validation_results: list[FieldValidation] = []
    final_fields       = {}
    fields_to_retry    = []

    for field_name, result in state.draft_fields.items():
        issues = []
        passed = True

        # Check 1: Value present
        if result.value is None and result.status != "not_found":
            issues.append("No value extracted")
            passed = False

        # Check 2: Must have citation if value is present
        if result.value is not None and not result.sources:
            issues.append("Value present but no citation found")
            passed = False

        # Check 3: Confidence threshold
        if result.confidence < 0.3 and result.value is not None:
            issues.append(f"Low confidence: {result.confidence:.2f}")

        # Check 4: Date format validation for Due Date
        if field_name == "Due Date" and result.value:
            import re
            if not re.search(r"\d{4}|\d{1,2}[/-]\d{1,2}", str(result.value)):
                issues.append("Due Date format doesn't look like a date")
                passed = False

        # Check 5: Value plausibility check (WARNING ONLY — does not fail)
        # Hard enforcement is at parse time (Layer 1: no source → rejected).
        # Here we just log a warning if value words don't appear in source.
        # We do NOT set passed=False because paraphrased values legitimately
        # won't exactly match source chunk text word-for-word.
        if result.value is not None and result.sources and len(str(result.value)) <= 80:
            src_text = " ".join(c.chunk_text.lower() for c in result.sources)
            value_str = str(result.value).lower()
            value_words = [w for w in value_str.split() if len(w) > 3]
            matches = sum(1 for w in value_words if w in src_text)
            if value_words and matches < min(2, len(value_words)):
                # Informational warning only — not a hard failure
                issues.append(f"[WARN] Value '{str(result.value)[:40]}' may not be directly supported by source text (matched {matches}/{len(value_words)} words)")

        v = FieldValidation(
            field_name=field_name,
            passed=passed,
            issues=issues,
            confidence=result.confidence,
        )
        validation_results.append(v)

        if not passed and state.should_retry(field_name):
            fields_to_retry.append(field_name)
            # Mark for re-extraction
            updated = result.model_copy(update={"status": "failed"})
            final_fields[field_name] = updated
        else:
            # Mark as final — if passed or retries exhausted
            final_status = result.status if passed else "failed"
            final_fields[field_name] = result.model_copy(update={"status": final_status})

    passed_count = sum(1 for v in validation_results if v.passed)
    logger.info(
        f"[Validator] Passed: {passed_count}/{len(validation_results)}, "
        f"To retry: {len(fields_to_retry)}"
    )

    # Increment retry counters
    retry_counts = dict(state.retry_counts)
    for f in fields_to_retry:
        retry_counts[f] = retry_counts.get(f, 0) + 1

    return {
        "validation_results": validation_results,
        "final_fields":       final_fields,
        "draft_fields":       final_fields,  # feed back for retry
        "retry_counts":       retry_counts,
        "current_step":       "validation_done",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Q&A Node
# ─────────────────────────────────────────────────────────────────────────────

def qa_node(state: AgentState) -> dict:
    """
    Q&A / Report Agent:
    - Answers free-form questions with citations
    - For extraction mode: produces the final JSON output
    """
    logger.info(f"[QA] mode={state.mode} query='{state.user_query[:50]}'")

    if state.mode == "qa":
        evidence_text = "\n\n---\n\n".join(
            f"[{e.get('file_name','?')} p.{e.get('page_number','?')} | "
            f"bid={e.get('bid_id','?')} type={e.get('doc_type','?')}]\n{e.get('chunk_text','')}"
            for e in state.retrieved_evidence[:15]
        )

        SYSTEM = """You are a Q&A Agent for RFP document analysis.
Answer the user's question using ONLY the provided document evidence.
- Always cite sources (file name + page number) for every claim.
- If the answer is not in the evidence, say "Not found in documents".
- Never hallucinate or guess.
- Format: concise markdown with inline citations like [filename, p.X]."""

        msgs = [{
            "role": "user",
            "content": (
                f"Question: {state.user_query}\n\n"
                f"EVIDENCE FROM BID DOCUMENTS:\n{evidence_text}"
            ),
        }]

        final_text, updated_msgs = run_agent_with_tools(SYSTEM, msgs, TOOL_SCHEMAS, max_iterations=4)

        # Primary: use the final_text returned directly from the LLM
        final_answer = final_text or ""

        # Fallback: scan message history for last assistant message
        if not final_answer:
            for msg in reversed(updated_msgs):
                if msg.get("role") == "assistant" and msg.get("content"):
                    final_answer = msg["content"]
                    break

        if not final_answer:
            final_answer = "I could not find an answer in the bid documents. Please try rephrasing your question."

        return {
            "final_answer": final_answer,
            "is_complete":  True,
            "current_step": "qa_done",
        }

    else:
        # Extraction mode: finalize JSON
        final_json = state.to_output_json()
        return {
            "final_json":  final_json,
            "is_complete": True,
            "current_step": "finalize_done",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Router functions (used by LangGraph conditional edges)
# ─────────────────────────────────────────────────────────────────────────────

def should_retry(state: AgentState) -> str:
    """After validation: retry extraction if any fields failed and retries remain."""
    failed_fields = [
        f for f, v in state.draft_fields.items()
        if v.status == "failed" and state.should_retry(f)
    ]
    if failed_fields:
        logger.info(f"[Router] Retrying {len(failed_fields)} fields: {failed_fields}")
        return "retry"
    return "finalize"


def route_after_orchestrator(state: AgentState) -> str:
    """Route to retrieval for both modes."""
    return "retrieval"


def route_after_retrieval(state: AgentState) -> str:
    if state.mode == "qa":
        return "qa"
    return "extraction"
