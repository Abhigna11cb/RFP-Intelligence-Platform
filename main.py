from __future__ import annotations

import sys

# Force UTF-8 on Windows so emoji in log messages don't crash the terminal
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import asyncio
import json
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_index(args):
    """Index bid folders into PostgreSQL."""
    from rfp_platform.core.database import init_database
    from rfp_platform.ingestion.pipeline import ingest_bid_folder

    print("Initialising database schema...")
    if not init_database():
        sys.exit(1)

    async def _run():
        for folder in args.bid:
            print(f"\nIndexing: {folder}")
            results = await ingest_bid_folder(
                folder_path=folder,
                force_reindex=args.force,
            )
            indexed = sum(1 for r in results if r["status"] == "indexed")
            skipped = sum(1 for r in results if r["status"] == "skipped")
            errors  = sum(1 for r in results if r["status"] == "error")
            print(f"  Indexed: {indexed}  Skipped: {skipped}  Errors: {errors}")
            for r in results:
                status_icon = "[OK]" if r["status"] == "indexed" else "[SKIP]" if r["status"] == "skipped" else "[ERR]"
                fname = r.get('file', r.get('message', ''))
                print(f"  {status_icon} chunks={r['chunks']:>4}  tokens={r['tokens']:>6}  {fname}")

    asyncio.run(_run())
    print("\nAll bids indexed!")


def cmd_extract(args):
    """Run the extraction agent pipeline for a bid."""
    from rfp_platform.agents.orchestrator import run_extraction

    print(f"Running extraction for bid '{args.bid}'...")
    result = run_extraction(args.bid)

    print("\n" + "=" * 60)
    print(f"  Extraction Results — {args.bid}")
    print("=" * 60)
    fields = result.get("fields", {})
    for field_name, data in fields.items():
        value  = data.get("value", "NOT FOUND")
        conf   = data.get("confidence", 0.0)
        src    = data.get("sources", [{}])[0].get("file", "") if data.get("sources") else ""
        page   = data.get("sources", [{}])[0].get("page", "") if data.get("sources") else ""
        print(f"  {field_name:<40} {str(value)[:30]:<32} (conf={conf:.2f}) [{src} p.{page}]")

    val = result.get("validation", {})
    print(f"\n  Validation — Passed: {val.get('passed',0)}  "
          f"Not Found: {val.get('not_found',0)}  Failed: {val.get('failed',0)}")

    changes = result.get("addendum_changes", [])
    if changes:
        print(f"\n  Addendum Changes ({len(changes)}):")
        for c in changes:
            print(f"    Addendum {c.get('addendum_number','?')} — "
                  f"{c.get('field_name','?')}: "
                  f"{c.get('original_value','?')} → {c.get('new_value','?')}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Full results saved to: {args.output}")

    print("=" * 60)


def cmd_ask(args):
    """Ask a free-form question."""
    from rfp_platform.agents.orchestrator import run_qa

    bid_id = args.bid or None
    print(f"Question: {args.question}")
    if bid_id:
        print(f"Bid scope: {bid_id}")
    print("\nSearching...")

    result = run_qa(args.question, bid_id)
    print("\n" + "=" * 60)
    print("Answer:")
    print(result.get("answer", "No answer generated."))
    print("=" * 60)


def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn
    from rfp_platform.core.config import get_settings
    cfg = get_settings()
    print(f"Starting API server at http://{cfg.api_host}:{cfg.api_port}")
    print("Docs available at: http://localhost:8000/docs")
    uvicorn.run(
        "rfp_platform.api.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=args.reload,
        log_level="info",
    )


def cmd_eval(args):
    """Run retrieval evaluation benchmark."""
    from rfp_platform.eval.evaluate import run_evaluation, print_eval_report
    print("Running retrieval evaluation benchmark...")
    results = run_evaluation()
    print_eval_report(results)


def main():
    parser = argparse.ArgumentParser(
        description="RFP Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Index bid folders into PostgreSQL")
    p_index.add_argument("--bid", action="append", required=True,
                         help="Path to bid folder (repeat for multiple)")
    p_index.add_argument("--force", action="store_true",
                         help="Re-index even if already present")

    # extract
    p_extract = sub.add_parser("extract", help="Run extraction pipeline")
    p_extract.add_argument("--bid", required=True, help="Bid ID (e.g. Bid1)")
    p_extract.add_argument("--output", default=None,
                           help="Save results to JSON file")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question about bid documents")
    p_ask.add_argument("question", help="Natural language question")
    p_ask.add_argument("--bid", default=None, help="Restrict to specific bid")

    # serve
    p_serve = sub.add_parser("serve", help="Start the FastAPI REST API server")
    p_serve.add_argument("--reload", action="store_true",
                         help="Enable auto-reload (development)")

    # eval
    sub.add_parser("eval", help="Run retrieval evaluation benchmark")

    args = parser.parse_args()
    {
        "index":   cmd_index,
        "extract": cmd_extract,
        "ask":     cmd_ask,
        "serve":   cmd_serve,
        "eval":    cmd_eval,
    }[args.command](args)


if __name__ == "__main__":
    main()
