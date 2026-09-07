# Face-Match Provenance Pipeline

A pipeline that takes a face scan as input, identifies matching content on the web using reverse image search, and verifies the discovered data using blockchain anchoring — end to end.

```
Face scan input → Web search (find matching post) → Blockchain verification
```

---

## How it works

### Stage 1 — Face Identification
Detects and encodes a face from the input image using **InsightFace** (SCRFD detection + ArcFace embedding), producing a 512-dimensional vector. Quality gates reject blurry, tiny, or low-confidence detections. Falls back to a deterministic PIL-based encoder on machines without ONNX/InsightFace.

### Stage 2 — Web Search
Performs a **genuine reverse image search** using [SerpApi Google Lens](https://serpapi.com/google-lens-api). The probe face is uploaded and matched against publicly available web content. Candidate images are downloaded and compared against the probe embedding via cosine similarity. Falls back to DuckDuckGo text search if SerpApi is unavailable.

### Stage 3 — Blockchain Verification
The match record (containing only URLs, hashes, and similarity scores — **no names or embeddings**) is canonically serialised, SHA-256 hashed, and anchored on a simulated blockchain. A tamper-detection demo proves that changing even one field by 0.0001 produces a completely different digest that fails chain verification.

---

## Setup

### Prerequisites

- **Python 3.10 or 3.11** (onnxruntime wheels may not be available on 3.12+)
- **pip** (Python package manager)
- **Git** (for cloning and version control)
- A **SerpApi account** (free tier: 100 searches/month) — [sign up here](https://serpapi.com/users/sign_up)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/face-recognition-blockchain-verification.git
cd face-recognition-blockchain-verification
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `google-search-results` (SerpApi), `Pillow`, `numpy`, `requests`, `beautifulsoup4`, `python-dotenv`, `insightface`, `onnxruntime`, `opencv-python-headless`, `scikit-learn`, `matplotlib`, `pytest`.

If InsightFace installation fails (common on some systems), the pipeline will automatically use the PIL fallback encoder — everything still works.

### 4. Set up your API key

Copy the example environment file:

```bash
cp .env.example .env
```

Open `.env` in any text editor and replace the placeholder with your actual SerpApi key:

```
SERPAPI_KEY=your_actual_serpapi_key_here
```

> **Security:** The `.env` file is listed in `.gitignore` and will never be pushed to GitHub. Your key stays local.

### 5. (Optional) Pre-download InsightFace model weights

First run of InsightFace downloads ~300 MB of model weights. To avoid this during a demo recording:

```bash
python -c "from insightface.app import FaceAnalysis; a = FaceAnalysis(name='buffalo_l'); a.prepare(ctx_id=-1)"
```

Skip this if InsightFace is not installed — the fallback encoder needs no model download.

---

## Running the project

### Run the full pipeline (end-to-end demo)

```bash
python src/demo_pipeline.py <path_to_face_image>
```

**Examples:**

```bash
# Using the included test probe image
python src/demo_pipeline.py data/probes/probe_match.jpg

# Using your own photo (best results with a real public figure or your own face)
python src/demo_pipeline.py path/to/your_photo.jpg

# With a custom similarity threshold
python src/demo_pipeline.py data/probes/probe_match.jpg --threshold 0.3

# Passing the API key directly (instead of using .env)
python src/demo_pipeline.py data/probes/probe_match.jpg --serpapi-key YOUR_KEY
```

### Run the unit tests

```bash
python -m pytest tests/ -v
```

All 17 tests run locally without any API key or InsightFace. They cover:
- Canonical JSON serialisation (determinism, compactness)
- SHA-256 hashing (round-trip stability, key-order invariance, tiny-change divergence)
- Tamper detection (untampered passes, tampered fails)
- Blockchain (genesis block, anchoring, verification, chain validity, multi-anchor, block info)
- Match evaluation (above/below/at threshold)

### Run with the non-matching probe (demonstrating the "no" path)

```bash
python src/demo_pipeline.py data/probes/probe_nomatch.jpg
```

This demonstrates that the system can correctly report "no match found" — a system that always finds something is a system nobody should trust.

---

## Pipeline output

The demo prints each stage with clear visual formatting:

```
╔══════════════════════════════════════════════════════════════════╗
║   FACE-MATCH PROVENANCE PIPELINE                                 ║
║   Face scan → Web search → Blockchain verification               ║
╚══════════════════════════════════════════════════════════════════╝

    STAGE 1 — Face detected, embedding computed
    STAGE 2 — SerpApi Google Lens results with candidate URLs
    STAGE 3 — Match record built, anchored on blockchain

    VERIFY  — Recomputed digest matches chain → untampered ✓
    TAMPER  — Mutate similarity by 0.0001 → different digest → fails ✗

    INTEGRITY ≠ CORRECTNESS
```

Full JSON output is saved to `data/pipeline_result.json` after each run.

---

## Project structure

```
face-recognition-blockchain-verification/
├── .env                     ← Your SerpApi key (gitignored, never pushed)
├── .env.example             ← Template showing required env vars
├── .gitignore               ← Protects .env and generated files
├── README.md                ← This file
├── requirements.txt         ← Python dependencies
├── task #3.pdf              ← Original problem statement
├── data/
│   └── probes/
│       ├── probe_match.jpg  ← Test face image
│       └── probe_nomatch.jpg← Test non-matching image
├── src/
│   ├── __init__.py
│   ├── face_utils.py        ← Face detection & encoding (InsightFace + fallback)
│   ├── web_search.py        ← SerpApi Google Lens reverse image search
│   ├── pipeline.py          ← Core pipeline orchestration & rich output
│   ├── blockchain.py        ← Simulated blockchain with blocks & PoW
│   └── demo_pipeline.py     ← CLI entry point
└── tests/
    └── test_pipeline.py     ← 17 unit tests (no API key needed)
```

---

## Blockchain

This project uses a **simulated local blockchain** implemented in Python. The task specification permits this:

> *"Any blockchain may be used — public testnet, mainnet, or a local/simulated chain — as long as you can demonstrate re-verifying the data against the on-chain record."*

The simulated chain implements:
- **Genesis block** initialising the chain
- **Block structure** with index, timestamp, data digest, previous block hash, and nonce
- **Proof of work** (low difficulty for speed, but demonstrates the concept)
- **Chain integrity verification** walking every block to verify hash linkage
- **Tamper detection** showing that any field mutation produces a different digest

**Only the SHA-256 digest goes on-chain** — no images, names, embeddings, or URLs. This satisfies data erasure requirements since the off-chain record can be deleted independently.

---

## Known limitations

- **Demographic bias** — Face recognition models (including ArcFace) exhibit error rates varying by 10–100× across demographic groups (NIST FRVT Part 3). A single global threshold does not perform uniformly.

- **Integrity ≠ Correctness** — The blockchain proves the match record has not been altered since anchoring. It does **not** prove the facial match is factually correct. An anchored false positive is a permanently recorded false positive.

- **No liveness detection** — A printed photograph will pass face detection. Anti-spoofing (ISO/IEC 30107-3) is out of scope.

- **Search depends on web availability** — If the person's face doesn't appear on publicly accessible web pages, reverse image search returns no matches. This is by design.

- **Simulated blockchain** — The chain is an in-memory Python ledger. It demonstrates tamper-evidence but does not provide the persistence or decentralisation of a real blockchain.

- **API rate limits** — SerpApi free tier provides 100 searches/month. The pipeline gracefully falls back to DuckDuckGo text search if the key is missing or quota is exhausted.

- **Scraped names are heuristic** — Any names extracted from web results are best-effort metadata, never asserted as verified identity.

- **Key custody** — The simulated chain attests to a record digest, not to any real-world identity behind the submission.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `InsightFace not available, using PIL fallback` | InsightFace/ONNX not installed | Install with `pip install insightface onnxruntime` or accept the fallback |
| `No SerpApi key` | `.env` missing or key not set | Create `.env` with your `SERPAPI_KEY` |
| `SerpApi returned status 401` | Invalid API key | Check your key at [serpapi.com/dashboard](https://serpapi.com/dashboard) |
| `SerpApi returned status 429` | Rate limit exceeded | Wait or upgrade your SerpApi plan |
| `No search results found` | Face not on the web / API issue | Try a photo of someone with a public web presence |
| `All faces rejected by quality gate` | Image too blurry/small | Use a clearer, larger face photo (min 60px face width) |
| `onnxruntime install fails` | Python 3.12+ | Use Python 3.10 or 3.11 |

---

## License

MIT
