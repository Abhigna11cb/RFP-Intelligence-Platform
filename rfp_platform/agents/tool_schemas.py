"""
OpenAI function-calling tool definitions for all 8 search tools.
These are passed to LangGraph agents as the `tools` parameter.

Decision on your summary_agent tool:
  REMOVED — In LangGraph, agents communicate via the shared AgentState
  (Pydantic model), not via a "write response" tool. The Q&A agent writes
  the final answer directly into state. This is cleaner and avoids
  the LLM having to call a tool just to format its own response.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas (OpenAI function-calling format)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [

    # ── 1. hybrid_search (PRIMARY) ───────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": (
                "PRIMARY SEARCH TOOL — Hybrid semantic + BM25 keyword search "
                "with Reciprocal Rank Fusion. Use this for all general queries, "
                "field extraction, and Q&A over bid documents. "
                "Returns chunks ranked by combined relevance score with citations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language or keyword query"
                    },
                    "bid_id": {
                        "type": "string",
                        "description": "Filter to a specific bid ('Bid1', 'Bid2'). Optional."
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["rfp", "addendum", "specs", "affidavit", "bid_page"],
                        "description": "Filter by document type. Optional."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)",
                        "default": 10
                    },
                    "file_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of specific filenames to restrict search to."
                    },
                    "addendum_number": {
                        "type": "integer",
                        "description": "Filter to a specific addendum number. Optional."
                    }
                },
                "required": ["query"]
            }
        }
    },

    # ── 2. semantic_search ───────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Pure vector/semantic similarity search using text-embedding-3-large. "
                "Best for conceptual queries, finding related information by meaning, "
                "and thematic matches. Use when hybrid_search returns too many "
                "unrelated keyword hits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to find semantically similar content"
                    },
                    "bid_id": {
                        "type": "string",
                        "description": "Filter to a specific bid. Optional."
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["rfp", "addendum", "specs", "affidavit", "bid_page"],
                        "description": "Filter by document type. Optional."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 10)",
                        "default": 10
                    },
                    "file_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of specific filenames to search within."
                    }
                },
                "required": ["query"]
            }
        }
    },

    # ── 3. keyword_search ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": (
                "BM25 keyword full-text search using PostgreSQL tsvector. "
                "Best for finding EXACT terms: bid numbers (JA-207652, E20P4600040), "
                "model numbers (CC7802, WD22TB4), part SKUs (210-BLYZ), "
                "specific names, dates, or technical terminology. "
                "Use when you need exact string matches, not conceptual similarity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or exact phrase to search for"
                    },
                    "bid_id": {
                        "type": "string",
                        "description": "Filter to a specific bid. Optional."
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["rfp", "addendum", "specs", "affidavit", "bid_page"],
                        "description": "Filter by document type. Optional."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 10)",
                        "default": 10
                    },
                    "file_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of specific filenames to search within."
                    }
                },
                "required": ["query"]
            }
        }
    },

    # ── 4. retrieve_nearby_chunks ─────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "retrieve_nearby_chunks",
            "description": (
                "Retrieves surrounding chunks for context expansion around a specific chunk. "
                "Use when a search result is relevant but needs more surrounding context "
                "to understand the full clause, section, or table it belongs to. "
                "You must know the file_name and chunk_index from a previous search result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Exact filename from a previous search result (e.g., 'Addendum 2 RFP JA-207652.pdf')"
                    },
                    "chunk_index": {
                        "type": "integer",
                        "description": "The chunk_index from a previous search result"
                    },
                    "context_window": {
                        "type": "integer",
                        "description": "Number of chunks before and after to retrieve (default: 2)",
                        "default": 2
                    }
                },
                "required": ["file_name", "chunk_index"]
            }
        }
    },

    # ── 5. retrieve_pdf_page_image ────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "retrieve_pdf_page_image",
            "description": (
                "Renders specific PDF pages as images for visual inspection. "
                "Use when extracted text is garbled, scrambled, or missing, "
                "or when a page contains complex tables/diagrams. "
                "The returned base64 image can be passed to a vision model for OCR."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Exact PDF filename (e.g., 'Dell_Laptop_Specs.pdf')"
                    },
                    "page_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of 1-indexed page numbers to render"
                    }
                },
                "required": ["file_name", "page_numbers"]
            }
        }
    },

    # ── 6. get_bid_documents ──────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_bid_documents",
            "description": (
                "Lists all indexed documents for a bid with their metadata. "
                "Use at the start of any extraction run to understand what "
                "files are available, what doc_types exist, and which "
                "addendum numbers are present."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bid_id": {
                        "type": "string",
                        "description": "The bid folder identifier ('Bid1', 'Bid2')"
                    }
                },
                "required": ["bid_id"]
            }
        }
    },

    # ── 7. get_addendum_changes ───────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_addendum_changes",
            "description": (
                "Fetches ALL addendum content for a bid, ordered by addendum number. "
                "Use this as the Addendum Reconciliation Agent's primary tool to "
                "detect what changed between the original RFP and its addendums "
                "(e.g., due date extensions, requirement changes). "
                "Returns all chunks from addendum documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bid_id": {
                        "type": "string",
                        "description": "The bid folder identifier ('Bid1', 'Bid2')"
                    }
                },
                "required": ["bid_id"]
            }
        }
    },

    # ── 8. search_by_field ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_by_field",
            "description": (
                "Targeted hybrid search optimised for extracting a specific field "
                "from the 20 required extraction fields. Uses pre-built query expansions "
                "for each field (e.g., 'Due Date' expands to "
                "'due date submission deadline closing date proposals'). "
                "Use this instead of hybrid_search when extracting structured fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "enum": [
                            "Bid Number", "Title", "Due Date", "Bid Submission Type",
                            "Term of Bid", "Pre Bid Meeting", "Installation",
                            "Bid Bond Requirement", "Delivery Date", "Payment Terms",
                            "Any Additional Documentation Required",
                            "MFG for Registration", "Contract or Cooperative to use",
                            "Model_no", "Part_no", "Product", "contact_info",
                            "company_name", "Bid Summary", "Product Specification"
                        ],
                        "description": "One of the 20 required extraction fields"
                    },
                    "bid_id": {
                        "type": "string",
                        "description": "Filter to a specific bid ('Bid1', 'Bid2')"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 8)",
                        "default": 8
                    }
                },
                "required": ["field_name"]
            }
        }
    },
]
