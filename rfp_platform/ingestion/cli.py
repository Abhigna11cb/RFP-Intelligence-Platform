"""
CLI entrypoint for the ingestion pipeline.

Usage:
    # Index a single bid folder
    python -m rfp_platform.ingestion.cli --bid ./Bid1 --bid-id Bid1

    # Index multiple bid folders
    python -m rfp_platform.ingestion.cli --bid ./Bid1 --bid ./Bid2

    # Force re-index (overwrite existing)
    python -m rfp_platform.ingestion.cli --bid ./Bid1 --force
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RFP Platform — Ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--bid",
        action="append",
        dest="bids",
        metavar="FOLDER",
        required=True,
        help="Path to a bid folder (can be repeated for multiple bids)",
    )
    parser.add_argument(
        "--bid-id",
        action="append",
        dest="bid_ids",
        metavar="ID",
        default=None,
        help="Override bid ID (must match --bid order; defaults to folder name)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-index files even if they already exist in the database",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        default=False,
        help="Initialise/migrate the database schema before indexing",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # Optional DB init
    if args.init_db:
        from rfp_platform.core.database import init_database
        logger.info("Initialising database schema…")
        if not init_database():
            logger.error("Database init failed. Aborting.")
            sys.exit(1)

    from rfp_platform.ingestion.pipeline import ingest_bid_folder

    bid_ids = args.bid_ids or [None] * len(args.bids)
    if len(bid_ids) != len(args.bids):
        logger.error("--bid-id count must match --bid count")
        sys.exit(1)

    all_results = []
    for folder, bid_id in zip(args.bids, bid_ids):
        logger.info(f"▶ Processing folder: {folder}")
        results = await ingest_bid_folder(
            folder_path=folder,
            bid_id=bid_id,
            force_reindex=args.force,
        )
        all_results.extend(results)

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'FILE':<35} {'STATUS':<10} {'CHUNKS':>6} {'TOKENS':>8}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r.get('file',''):<35} "
            f"{r['status']:<10} "
            f"{r['chunks']:>6} "
            f"{r['tokens']:>8}"
        )
    print("=" * 60)

    errors = [r for r in all_results if r["status"] == "error"]
    if errors:
        logger.error(f"{len(errors)} file(s) failed to index.")
        sys.exit(1)
    else:
        logger.info("All files indexed successfully ✅")


if __name__ == "__main__":
    asyncio.run(main())
