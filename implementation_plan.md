# Updated Implementation Plan — Face Identification & Blockchain Verification

## Problem Statement (from Task #3)

> Face scan input → Web/social media search (find matching post) → Blockchain upload/verification of the discovered data

**Key requirements from the PDF:**
1. **Face identification** — detect and encode a face from an input image
2. **Web/social media search** — use the face to find at least one real, matching social media post (reverse image search, API, or scripted search). **Must be a genuine search step, not hardcoded.**
3. **Blockchain verification** — upload the post (or hash/fingerprint) to a blockchain for tamper-evident re-verification. Any chain (public testnet, mainnet, local/simulated) is acceptable.
4. **No website required** — focus on the pipeline itself
5. **GitHub repo** with README covering what it does, how to run, which blockchain, limitations
6. **Screen recording** of the pipeline working end-to-end

> [!IMPORTANT]
> The deadline is **Sept 7, 2026, 11:59 PM** — that's today. This plan prioritises getting a working end-to-end demo as fast as possible.

---

## Revised Architecture

```
┌─────────────────────────────────────────────────┐
│  STAGE 1 — Face Encoding                        │
│  Input image → InsightFace SCRFD detect          │
│  → ArcFace 512-d embedding + quality gate        │
│  (fallback: PIL-based deterministic encoder)     │
└─────────────────┬───────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────┐
│  STAGE 2 — Web Search (LIVE, genuine)            │
│  Probe image → SerpApi Google Lens API           │
│  → list of matching URLs, thumbnails, titles     │
│  → download candidate images                     │
│  → encode candidates → cosine similarity         │
│  → best match above threshold → MatchRecord      │
└─────────────────┬───────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────┐
│  STAGE 3 — Blockchain Anchoring                  │
│  MatchRecord → canonical JSON → SHA-256 digest   │
│  → anchor(digest) on chain                       │
│  → verify(digest) → tamper detection demo        │
└─────────────────────────────────────────────────┘
```

---

## Proposed Changes

### Stage 1 — Face Encoder

#### [MODIFY] [face_utils.py](file:///c:/Namya/Projects/face-recognition-blockchain-verification/src/face_utils.py)

Keep the existing fallback encoder but add an InsightFace-based encoder that activates when the library is available. The fallback ensures the pipeline works even without InsightFace installed (e.g., in CI or on machines without ONNX support).

---

### Stage 2 — Web Search (NEW architecture)

#### [NEW] `src/web_search.py` (overwrite existing)

Replace the current DuckDuckGo HTML scraper with a **SerpApi Google Lens** integration:

1. Upload the probe image to SerpApi's Google Lens endpoint
2. Receive matching results (URLs, titles, thumbnails, source pages)
3. Download candidate images from the results
4. Encode candidates locally with InsightFace
5. Compare cosine similarity against the probe embedding
6. Return the best match as a `MatchRecord` or explicit no-match

**SerpApi Google Lens** is the easiest to set up:
- Sign up at [serpapi.com](https://serpapi.com) → free 100 searches/month
- Single API key, Python package: `pip install google-search-results`
- Endpoint: `google_lens` with `url` parameter (image URL or base64)

**Fallback**: If SerpApi is unavailable, fall back to a DuckDuckGo text search with the image filename as query (degraded but functional).

---

### Stage 3 — Blockchain Anchoring

#### [KEEP] `src/pipeline.py` — canonicalisation and hashing (already works)
#### [KEEP] `src/blockchain.py` — simulated blockchain (acceptable per task: "local/simulated chain" is fine)

The task says: *"Any blockchain may be used — public testnet, mainnet, or a local/simulated chain — as long as you can demonstrate re-verifying the data against the on-chain record."*

The simulated blockchain we already have is **sufficient for the deadline**. We can optionally upgrade to Anvil/Amoy later.

---

### Demo Entry Point

#### [MODIFY] `src/demo_pipeline.py`

Rewrite to run the full live pipeline:
1. Accept a probe image path
2. Encode the face
3. Run SerpApi Google Lens search
4. Find the best matching result
5. Build MatchRecord
6. Anchor on blockchain (simulated)
7. Verify untampered
8. Demonstrate tamper detection (mutate one field → digest changes → verification fails)
9. Print clear JSON output at each stage

---

### Files to Update

| File | Action |
|---|---|
| `requirements.txt` | Add `google-search-results` (SerpApi package) |
| `src/web_search.py` | Rewrite with SerpApi Google Lens + fallback |
| `src/face_utils.py` | Add InsightFace encoder alongside fallback |
| `src/pipeline.py` | Minor updates to integrate new search flow |
| `src/demo_pipeline.py` | Rewrite for live end-to-end demo |
| `README.md` | Update with how-to-run, API key setup, limitations |
| `.env` / `.env.example` | Add `SERPAPI_KEY` |

### Files That Stay As-Is

| File | Reason |
|---|---|
| `src/blockchain.py` | Simulated chain is acceptable per task requirements |
| `tests/test_pipeline.py` | Existing tests still valid |

### Files/Dirs No Longer Needed for MVP

| Item | Reason |
|---|---|
| `corpus/` | No offline corpus — live web search instead |
| `src/harvest/` | No harvester — SerpApi does the discovery |
| `data/corpus/` | No pre-built corpus |

> [!NOTE]
> The generated face images in `data/corpus/images/` and `data/probes/` are still useful as test inputs for the demo.

---

## Open Questions

> [!IMPORTANT]
> **SerpApi API key**: You'll need to sign up at [serpapi.com](https://serpapi.com) and get a free API key. Should I include instructions for that in the README, or do you want to sign up now so I can wire it in?

> [!IMPORTANT]  
> **Blockchain choice**: The simulated Python blockchain is fastest to demo. Do you want to stick with that for tonight's deadline, or try to get Anvil (local Ethereum) working too?

---

## Verification Plan

### Demo Recording Flow
1. Run `python src/demo_pipeline.py data/probes/probe_match.jpg`
2. Show: face detected → SerpApi search results → best match found → MatchRecord created → anchored on chain → verified → tamper detection
3. Run with a non-matching image to show the no-match path
4. Record screen → upload → submit

### Automated Tests
```bash
python -m pytest tests/ -v
```
