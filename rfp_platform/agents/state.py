"""
Shared LangGraph state — the single source of truth passed between all agents.
Every agent reads from and writes to this object.
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Citation (every extracted value must have at least one) ───────────────────

class Citation(BaseModel):
    file_name:       str
    page_number:     int
    chunk_text:      str
    chunk_index:     int
    bid_id:          str
    doc_type:        str
    score:           float = 0.0


# ── Single extracted field ────────────────────────────────────────────────────

class FieldResult(BaseModel):
    value:      Any             = None
    sources:    list[Citation]  = Field(default_factory=list)
    confidence: float           = 0.0
    notes:      str             = ""
    status:     Literal["extracted", "not_found", "failed", "pending"] = "pending"


# ── Addendum change record ────────────────────────────────────────────────────

class AddendumChange(BaseModel):
    addendum_number: int
    field_name:      str
    original_value:  Any
    new_value:       Any
    source:          Citation | None = None
    notes:           str = ""


# ── Validation result per field ───────────────────────────────────────────────

class FieldValidation(BaseModel):
    field_name:  str
    passed:      bool
    issues:      list[str] = Field(default_factory=list)
    confidence:  float     = 0.0


# ── Main shared agent state ───────────────────────────────────────────────────

class AgentState(BaseModel):
    """
    Shared state object passed through the LangGraph pipeline.
    All agents read from and write to this.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    run_id:     str  = Field(default_factory=lambda: str(uuid.uuid4()))
    bid_id:     str  = ""
    mode:       Literal["extraction", "qa"] = "extraction"

    # ── User input ────────────────────────────────────────────────────────────
    user_query: str  = ""          # for QA mode
    goal:       str  = ""          # e.g. "Extract all fields for Bid1"

    # ── Orchestration ─────────────────────────────────────────────────────────
    plan:           list[str]       = Field(default_factory=list)
    current_step:   str             = ""
    steps_completed: list[str]      = Field(default_factory=list)

    # ── Retrieved evidence ────────────────────────────────────────────────────
    retrieved_evidence: list[dict[str, Any]] = Field(default_factory=list)

    # ── Extracted fields ──────────────────────────────────────────────────────
    # Keys are the 20 field names from the assignment spec
    draft_fields:  dict[str, FieldResult] = Field(default_factory=dict)
    final_fields:  dict[str, FieldResult] = Field(default_factory=dict)

    # ── Addendum reconciliation ───────────────────────────────────────────────
    addendum_changes: list[AddendumChange] = Field(default_factory=list)

    # ── Validation ────────────────────────────────────────────────────────────
    validation_results: list[FieldValidation] = Field(default_factory=list)
    retry_counts:       dict[str, int]         = Field(default_factory=dict)
    max_retries:        int                    = 2

    # ── Final output ──────────────────────────────────────────────────────────
    final_answer:   str  = ""      # for QA mode: cited markdown answer
    final_json:     dict = Field(default_factory=dict)  # for extraction mode

    # ── Metadata ──────────────────────────────────────────────────────────────
    errors:       list[str]        = Field(default_factory=list)
    is_complete:  bool             = False
    total_tokens: int              = 0

    class Config:
        arbitrary_types_allowed = True

    def mark_step_done(self, step: str) -> None:
        if step not in self.steps_completed:
            self.steps_completed.append(step)
        self.current_step = step

    def add_error(self, error: str) -> None:
        self.errors.append(error)

    def should_retry(self, field_name: str) -> bool:
        return self.retry_counts.get(field_name, 0) < self.max_retries

    def increment_retry(self, field_name: str) -> None:
        self.retry_counts[field_name] = self.retry_counts.get(field_name, 0) + 1

    def to_output_json(self) -> dict:
        """Produce the final assignment-spec JSON output.
        
        Every field includes: value, sources (file+page+chunk_id+doc_type+text), confidence, notes.
        Golden rule enforced: no value ships without at least one source.
        """
        fields_out = {}
        for name, result in self.final_fields.items():
            # Enforce golden rule in output: strip value if no sources
            value = result.value
            if value is not None and not result.sources:
                value = None   # never emit unsupported value

            fields_out[name] = {
                "value":      value,
                "sources": [
                    {
                        "file":       c.file_name,
                        "page":       c.page_number,
                        "chunk_id":   c.chunk_index,
                        "doc_type":   c.doc_type,
                        "bid_id":     c.bid_id,
                        "text":       c.chunk_text[:500],   # enough to verify the value
                    }
                    for c in result.sources
                ],
                "confidence": result.confidence,
                "notes":      result.notes if result.notes else (
                    "Not found in documents" if value is None else ""
                ),
            }

        passed  = sum(1 for r in self.final_fields.values() if r.status == "extracted")
        failed  = sum(1 for r in self.final_fields.values() if r.status == "failed")
        missing = sum(1 for r in self.final_fields.values() if r.status == "not_found")

        return {
            "bid_id":           self.bid_id,
            "fields":           fields_out,
            "addendum_changes": [c.model_dump() for c in self.addendum_changes],
            "validation":       {"passed": passed, "failed": failed, "not_found": missing},
            "run_id":           self.run_id,
        }
