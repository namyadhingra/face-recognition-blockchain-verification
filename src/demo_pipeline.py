"""Face-Match Provenance Pipeline — CLI Demo.

End-to-end: Face scan → Web search → Blockchain verification.

Usage:
    python src/demo_pipeline.py <probe_image>
    python src/demo_pipeline.py data/probes/probe_match.jpg
    python src/demo_pipeline.py data/probes/probe_match.jpg --threshold 0.3
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Load .env before any other imports that may need env vars
from dotenv import load_dotenv
load_dotenv()

from src.pipeline import FaceMatchPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Face-Match Provenance Pipeline"
    )
    parser.add_argument("probe_image", help="Path to the face image")
    parser.add_argument("--threshold", type=float, default=0.38,
                        help="Cosine similarity threshold (default: 0.38)")
    parser.add_argument("--serpapi-key", default=None,
                        help="SerpApi key (overrides SERPAPI_KEY env var)")
    args = parser.parse_args()

    probe = args.probe_image
    if not Path(probe).exists():
        print(f"\n  ✗ ERROR: Image not found: {probe}")
        sys.exit(1)

    # Header
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                                                                  ║")
    print("║   FACE-MATCH PROVENANCE PIPELINE                                 ║")
    print("║   Face scan → Web search → Blockchain verification               ║")
    print("║                                                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"    Probe:     {probe}")
    print(f"    Threshold: {args.threshold}")
    print(f"    Time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    has_key = bool(args.serpapi_key or os.environ.get("SERPAPI_KEY"))
    print(f"    SerpApi:   {'✓ Key configured' if has_key else '✗ No key — will use fallback search'}")

    # Run
    pipeline = FaceMatchPipeline(
        threshold=args.threshold,
        serpapi_key=args.serpapi_key,
    )
    result = pipeline.run(probe)

    # Save full output
    out = "data/pipeline_result.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    # Final summary
    print()
    print("═" * 66)
    status = result.get("status", "unknown")
    if status == "match":
        print(f"  ✓ MATCH FOUND & ANCHORED")
        print(f"    URL:        {result['record']['matched_source_url'][:60]}")
        print(f"    Similarity: {result['record']['similarity']}")
        print(f"    Digest:     {result['digest'][:32]}…")
        print(f"    TX:         {result['tx_hash'][:34]}…")
        print(f"    Verified:   {result['verified']}")
        td = result.get("tamper_demo", {})
        print(f"    Tamper:     {'Detected ✓' if td.get('tamper_detected') else 'Not tested'}")
    elif status == "no_match":
        print(f"  → NO MATCH (best similarity: {result.get('best_similarity')} < {result.get('threshold')})")
        print(f"    This is a valid outcome — the system can say 'no'.")
    elif status == "no_face":
        print(f"  ✗ NO FACE DETECTED: {result.get('error')}")
    elif status == "no_results":
        print(f"  ✗ NO WEB RESULTS FOUND")
    print(f"\n  Full output: {out}")
    print("═" * 66)
    print()


if __name__ == "__main__":
    main()
