# Face-Match Provenance Pipeline

This project implements the architecture described in the included design notes: a pipeline that accepts a probe face image, performs a genuine web search for a public profile candidate, and anchors a canonical match record to a blockchain-style record for tamper-evident verification.

## Included components

- A face encoding module with a deterministic fallback encoder for environments without InsightFace
- A web-search step that performs a live query and parses actual HTML results
- A canonical match-record format and SHA-256 hashing routine
- A simulated blockchain ledger to store the digest and verify it later
- A CLI demo entry point that shows the complete pipeline flow

## Running the project

```bash
Set-Location "C:\Users\Akshaya sree\Documents\face_recognition"
python -m pytest -q
python src/demo_pipeline.py data/sample_probe.png --query "public faculty profile face"
```

## Architecture note

This implementation stays on the “lookup, not enrolment” side of the design. The blockchain verifies the integrity of the anchored record; it does not prove that the underlying facial match is factually correct. The record captures provenance, threshold, and digest so any later change is detectable.

## Demo output

The CLI prints JSON containing the query, best matching result URL, the anchored digest, and a verification flag showing whether the chained record still matches the current digest.
