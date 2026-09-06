# Build Plan — Face-Match Provenance Pipeline

A single self-contained plan. Scope, stack, phased implementation with code,
demo runbook, and deliverables. Follow the phases in order; each ends in
something runnable.

---

## 1. What is being built

A three-stage pipeline:

```
probe photo → face detection + embedding
            → search over a corpus harvested from public institutional
              profile pages (live web crawl)
            → SHA-256 digest of the match record anchored on a blockchain
            → re-verification and tamper detection
```

Every requirement of the original brief is met: a real face-encoding stage, a
genuine (non-hardcoded) web search stage, and blockchain anchoring with
demonstrable re-verification.

### Scope boundary and its justification

The corpus is harvested from **public institutional profile pages** — lab team
pages, faculty directories, conference speaker listings. Not social media.

Reasons, in the order a judge will care about them:

1. **Contractual.** Instagram, Facebook, X, LinkedIn and TikTok prohibit
   automated collection and biometric processing in their terms of service.
   Public visibility is not a defence to a terms violation. Institutional pages
   generally carry no equivalent restriction.
2. **Legal.** DPDP Act 2023 §3(c)(ii) exempts personal data the data principal
   has themselves made publicly available. An academic profile page fits. SPDI
   Rules 2011 under IT Act §43A classify biometric data as sensitive, so the
   exemption matters.
3. **Technical.** Precision degrades with corpus size. At a false accept rate of
   1e-6 against a billion faces, a single query returns roughly 1,000 false
   matches before finding the true one. Corpus-bounded search is a sounder
   design, not merely a safer one — present it as engineering judgement.

**Lookup, not enrolment.** Using a team page to work out who is who is close to
that page's purpose. Building an arbitrary-probe biometric database is not. This
project stays on the lookup side, enforced by: a fixed corpus of 20–60
identities, no open query endpoint, and a scripted removal procedure.

---

## 2. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10 / 3.11 | onnxruntime wheels lag on 3.12+ |
| Face detection | InsightFace SCRFD | handles pose; ships with alignment |
| Face embedding | ArcFace `buffalo_l` | 512-d, strong open baseline |
| Vector search | FAISS `IndexFlatIP` | exact, no training, right at this scale |
| Crawling | `requests` + `beautifulsoup4` + `urllib.robotparser` | no JS needed on most institutional pages |
| Contract | Solidity 0.8.24 | current stable |
| Chain tooling | Foundry (`forge`, `anvil`, `cast`) | fast, no Node dependency |
| Chain (dev) | Anvil local | instant, free |
| Chain (demo) | Polygon Amoy testnet | ~2s blocks — matters on video |
| Web3 client | `web3.py` 6.x | mature |
| Evaluation | scikit-learn + matplotlib | ROC curves |
| Tests | pytest + `forge test` | |

---

## 3. Deliverables

| # | Deliverable | Phase |
|---|---|---|
| D1 | `corpus/MANIFEST.md` — targets, robots status, removal contact | 1 |
| D2 | Harvester producing images + provenance JSONL from live pages | 2 |
| D3 | Face encoder with quality gating and rejection logging | 3 |
| D4 | FAISS index + build script | 4 |
| D5 | ROC plot, EER, FAR/FRR at a stated operating threshold | 5 |
| D6 | Search module emitting MatchRecord / no-match | 6 |
| D7 | Canonicalisation + hashing with passing round-trip tests | 7 |
| D8 | `MatchAnchor.sol` deployed to Amoy, verified, explorer link | 8 |
| D9 | End-to-end CLI demo incl. tamper detection | 9 |
| D10 | README with limitations section | 10 |
| D11 | Demo video | 10 |

---

## Phase 0 — Environment

```bash
mkdir face-provenance && cd face-provenance
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
curl -L https://foundry.paradigm.xyz | bash && foundryup
```

`requirements.txt`:

```
insightface==0.7.3
onnxruntime==1.17.3
faiss-cpu==1.8.0
numpy==1.26.4
opencv-python-headless==4.9.0.80
requests==2.31.0
beautifulsoup4==4.12.3
web3==6.15.1
python-dotenv==1.0.1
scikit-learn==1.4.1.post1
matplotlib==3.8.3
pytest==8.1.1
```

```bash
pip install -r requirements.txt
mkdir -p src/{harvest,ingest,index,search,anchor,eval} corpus/adapters contracts cli tests data docs
find src -type d -exec touch {}/__init__.py \;
printf "data/\n.venv/\n.env\n*.faiss\n*.pkl\n" > .gitignore
```

**Pre-download model weights** — first InsightFace run pulls ~300 MB and
will stall video recording.

```bash
python -c "from insightface.app import FaceAnalysis; a=FaceAnalysis(name='buffalo_l'); a.prepare(ctx_id=-1)"
```

---

## Phase 1 — Corpus manifest (D1)

Write this **before** fetching anything. `corpus/MANIFEST.md`:

```markdown
# Corpus Manifest

Purpose: educational face-matching pipeline, coursework/hackathon.
Scope: public institutional profile pages only. No social media.

| # | Target URL | Type | robots.txt | Fetched | Images |
|---|---|---|---|---|---|
| 1 | https://example.edu/lab/team | lab team page | allowed | 2026-09-06 | 14 |
| 2 | … | | | | |

## Removal
Contact: <your email>. Any individual may request removal. Procedure:
`python -m cli.remove_identity --name "<name>"` — deletes source images,
provenance rows, and rebuilds the index.

On-chain digests cannot be removed. This is why anchored records contain no
names, images, URLs, or embeddings — only a SHA-256 digest.

## Rate limiting
2s minimum between requests. User-Agent identifies this project and carries the
contact address above.
```

---

## Phase 2 — Harvester (D2)

### `src/harvest/robots.py`

```python
import urllib.robotparser as urobot
from urllib.parse import urlparse

UA = "FaceProvenanceEdu/1.0 (coursework; contact: you@example.com)"

_cache = {}

def allowed(url: str) -> bool:
    parts = urlparse(url)
    root = f"{parts.scheme}://{parts.netloc}"
    if root not in _cache:
        rp = urobot.RobotFileParser()
        rp.set_url(root + "/robots.txt")
        try:
            rp.read()
        except Exception:
            return False          # fail closed
        _cache[root] = rp
    return _cache[root].can_fetch(UA, url)
```

Fail closed. If robots.txt is unreachable, do not fetch.

### `src/harvest/fetch.py`

```python
import time, hashlib, json, re
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from src.harvest.robots import allowed, UA

DELAY = 2.0
NAME_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z.'-]+){1,3}$")


def _get(url):
    time.sleep(DELAY)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    return r


def extract_people(page_url: str, html: str):
    """Generic heuristics. Add per-site adapters in corpus/adapters/ as needed."""
    soup = BeautifulSoup(html, "html.parser")
    out = []

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if not src:
            continue
        name = None

        alt = (img.get("alt") or "").strip()
        if NAME_RE.match(alt):
            name = alt

        if not name:
            fig = img.find_parent("figure")
            if fig and fig.figcaption:
                t = fig.figcaption.get_text(strip=True)
                if NAME_RE.match(t):
                    name = t

        if not name:
            parent = img.parent
            for _ in range(3):
                if parent is None:
                    break
                for tag in parent.find_all(["h2", "h3", "h4", "strong", "b", "p"]):
                    t = tag.get_text(strip=True)
                    if NAME_RE.match(t):
                        name = t
                        break
                if name:
                    break
                parent = parent.parent

        if name:
            out.append({"image_url": urljoin(page_url, src),
                        "extracted_name": name})
    return out


def harvest(targets, out_dir="data/corpus"):
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows, n = [], 0

    for page_url in targets:
        if not allowed(page_url):
            print(f"SKIP (robots): {page_url}")
            continue
        print(f"FETCH {page_url}")
        try:
            people = extract_people(page_url, _get(page_url).text)
        except Exception as e:
            print(f"  failed: {e}")
            continue
        print(f"  {len(people)} candidates")

        for p in people:
            if not allowed(p["image_url"]):
                continue
            try:
                blob = _get(p["image_url"]).content
            except Exception:
                continue
            n += 1
            iid = f"img_{n:04d}"
            ext = Path(p["image_url"].split("?")[0]).suffix or ".jpg"
            fn = f"{iid}{ext}"
            (out / "images" / fn).write_bytes(blob)
            rows.append({
                "image_id": iid,
                "image_file": fn,
                "source_page_url": page_url,
                "image_url": p["image_url"],
                "extracted_name": p["extracted_name"],
                "fetched_at": datetime.now(timezone.utc)
                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
                "robots_allowed": True,
                "image_sha256": hashlib.sha256(blob).hexdigest(),
            })

    with open(out / "provenance.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nHarvested {len(rows)} images from {len(targets)} pages")


if __name__ == "__main__":
    import sys
    harvest([l.strip() for l in open(sys.argv[1]) if l.strip()])
```

```bash
cat > corpus/targets.txt <<'EOF'
https://example.edu/lab/team
https://example.edu/faculty
EOF
python -m src.harvest.fetch corpus/targets.txt
```

Expect the generic extractor to work on some sites and not others. Write a small
adapter in `corpus/adapters/` per stubborn site rather than generalising — three
or four adapters is the right amount of work here and is more honest about what
the system does.

**Target selection tip:** prefer people who appear on *two* pages (a lab page and
a conference speaker page). Each such overlap gives you a genuine pair for free,
which Phase 5 needs.

---

## Phase 3 — Face encoder (D3)

### `src/ingest/encoder.py`

```python
import cv2, numpy as np
from dataclasses import dataclass
from insightface.app import FaceAnalysis

MIN_FACE_WIDTH, MIN_BLUR_VAR, MIN_DET_SCORE = 60, 50.0, 0.6


@dataclass
class EncodedFace:
    embedding: np.ndarray
    bbox: tuple
    det_score: float
    blur_var: float


class FaceEncoder:
    def __init__(self, model_pack="buffalo_l", ctx_id=-1):
        self.app = FaceAnalysis(name=model_pack)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def encode(self, image_path, largest_only=True):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"unreadable: {image_path}")

        kept, rejected = [], []
        for f in self.app.get(img):
            x1, y1, x2, y2 = map(int, f.bbox)
            if f.det_score < MIN_DET_SCORE:
                rejected.append(("det_score", round(float(f.det_score), 3))); continue
            if x2 - x1 < MIN_FACE_WIDTH:
                rejected.append(("face_width", x2 - x1)); continue
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                rejected.append(("empty_crop", None)); continue
            blur = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                                 cv2.CV_64F).var()
            if blur < MIN_BLUR_VAR:
                rejected.append(("blur", round(float(blur), 1))); continue

            emb = f.normed_embedding.astype(np.float32)
            emb = emb / np.linalg.norm(emb)     # defensive renormalisation
            kept.append(EncodedFace(emb, (x1, y1, x2, y2),
                                    float(f.det_score), float(blur)))

        kept.sort(key=lambda r: r.bbox[2] - r.bbox[0], reverse=True)
        return (kept[:1] if largest_only else kept), rejected
```

**Checkpoint before proceeding.** Two photos of the same person must give
noticeably higher cosine similarity than two photos of different people. Verify
by hand. If this fails, nothing downstream works.

```bash
python -c "
import numpy as np
from src.ingest.encoder import FaceEncoder
e = FaceEncoder()
a = e.encode('data/test_a.jpg')[0][0].embedding
b = e.encode('data/test_b.jpg')[0][0].embedding
print('similarity:', float(np.dot(a,b)))
"
```

---

## Phase 4 — Index (D4)

### `src/index/builder.py`

```python
import json, pickle
from pathlib import Path
import numpy as np, faiss
from src.ingest.encoder import FaceEncoder


def build(corpus_dir="data/corpus", out_dir="data/index",
          index_version="corpus-v1"):
    c, o = Path(corpus_dir), Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    enc = FaceEncoder()

    rows = [json.loads(l) for l in open(c / "provenance.jsonl") if l.strip()]
    embs, meta, skipped = [], [], 0

    for r in rows:
        try:
            faces, rej = enc.encode(str(c / "images" / r["image_file"]),
                                    largest_only=True)
        except ValueError:
            skipped += 1; continue
        if not faces:
            print(f"  no usable face: {r['image_id']} {rej}")
            skipped += 1; continue

        embs.append(faces[0].embedding)
        meta.append({**r, "bbox": faces[0].bbox})

    if not embs:
        raise RuntimeError("nothing indexed — check harvest and quality gates")

    mat = np.vstack(embs).astype(np.float32)
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat)
    faiss.write_index(idx, str(o / "corpus.faiss"))
    np.save(o / "embeddings.npy", mat)
    with open(o / "metadata.pkl", "wb") as fh:
        pickle.dump({"index_version": index_version, "records": meta}, fh)

    print(f"Indexed {len(meta)} faces, skipped {skipped}, version {index_version}")


if __name__ == "__main__":
    build()
```

```bash
python -m src.index.builder
```

`embeddings.npy` is saved separately so Phase 5 can read vectors without
depending on FAISS reconstruction support.

---

## Phase 5 — Calibration (D5)

This phase is where the marks are. Do not skip it.

### `src/eval/calibrate.py`

```python
import pickle, itertools
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def calibrate(index_dir="data/index", out_png="docs/roc.png"):
    d = Path(index_dir)
    embs = np.load(d / "embeddings.npy")
    recs = pickle.load(open(d / "metadata.pkl", "rb"))["records"]
    names = [r["extracted_name"] for r in recs]

    scores, truth = [], []
    for i, j in itertools.combinations(range(len(names)), 2):
        scores.append(float(np.dot(embs[i], embs[j])))
        truth.append(1 if names[i] == names[j] else 0)
    scores, truth = np.array(scores), np.array(truth)

    print(f"{truth.sum()} genuine / {len(truth)-truth.sum()} impostor pairs")
    if truth.sum() < 5:
        print("WARNING: too few genuine pairs. Add people appearing on 2+ pages.")

    fpr, tpr, thr = roc_curve(truth, scores)
    fnr = 1 - tpr
    e = int(np.nanargmin(np.abs(fnr - fpr)))
    print(f"EER {fpr[e]:.4f} @ threshold {thr[e]:.4f}")
    for t in (0.05, 0.01):
        i = int(np.argmin(np.abs(fpr - t)))
        print(f"FAR {fpr[i]:.4f}  FRR {fnr[i]:.4f}  threshold {thr[i]:.4f}")

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC {auc(fpr,tpr):.4f}")
    plt.plot([0,1],[0,1],"k--",lw=.8)
    plt.scatter([fpr[e]],[tpr[e]],c="red",zorder=5,label=f"EER thr={thr[e]:.3f}")
    plt.xlabel("False Accept Rate"); plt.ylabel("True Accept Rate")
    plt.title("Face matching ROC"); plt.legend(); plt.grid(alpha=.3)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    calibrate()
```

Pick your operating threshold from this output and put it in `.env`. Put the
number, the ROC plot, and your justification in the report.

**Do not take a threshold from a blog post.** ArcFace `buffalo_l` usually lands
near 0.35–0.45 cosine, but it is dataset-dependent, and measuring it yourself is
the entire point of this phase.

---

## Phase 6 — Search (D6)

### `src/search/searcher.py`

```python
import pickle, hashlib
from pathlib import Path
from datetime import datetime, timezone
import faiss, numpy as np
from src.ingest.encoder import FaceEncoder


class Searcher:
    def __init__(self, index_dir="data/index", threshold=0.38):
        d = Path(index_dir)
        self.index = faiss.read_index(str(d / "corpus.faiss"))
        blob = pickle.load(open(d / "metadata.pkl", "rb"))
        self.records = blob["records"]
        self.index_version = blob["index_version"]
        self.threshold = threshold
        self.encoder = FaceEncoder()

    def search(self, probe_path, k=5):
        faces, rej = self.encoder.encode(probe_path, largest_only=True)
        probe_hash = hashlib.sha256(open(probe_path, "rb").read()).hexdigest()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if not faces:
            return {"status": "no_face", "rejections": rej}

        q = faces[0].embedding.reshape(1, -1).astype(np.float32)
        scores, idxs = self.index.search(q, min(k, self.index.ntotal))
        cands = [{**self.records[i], "similarity": round(float(s), 4)}
                 for s, i in zip(scores[0], idxs[0]) if i != -1]
        top = cands[0] if cands else None

        if top is None or top["similarity"] < self.threshold:
            return {"status": "no_match", "probe_sha256": probe_hash,
                    "best_similarity": top["similarity"] if top else None,
                    "threshold": self.threshold, "searched_at": now,
                    "candidates": cands}

        return {
            "status": "match",
            "record": {
                "schema_version": "1.0",
                "probe_sha256": probe_hash,
                "matched_image_id": top["image_id"],
                "matched_source_url": top["source_page_url"],
                "matched_image_sha256": top["image_sha256"],
                "similarity": top["similarity"],
                "threshold": self.threshold,
                "model": "insightface/buffalo_l",
                "index_version": self.index_version,
                "matched_at": now,
            },
            "candidates": cands,
        }
```

The record carries a URL and hashes — **no name, no embedding**. Similarity is
rounded when the record is built, not at serialisation, which Phase 7 depends on.

---

## Phase 7 — Canonicalisation and hashing (D7)

### `src/anchor/canonical.py`

```python
import json, hashlib

def canonical_json(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def digest(record: dict) -> bytes:
    return hashlib.sha256(canonical_json(record)).digest()

def digest_hex(record: dict) -> str:
    return "0x" + digest(record).hex()
```

### `tests/test_canonical.py`

```python
import json
from src.anchor.canonical import digest_hex, canonical_json

def test_key_order_irrelevant():
    assert digest_hex({"b":2,"a":1}) == digest_hex({"a":1,"b":2})

def test_round_trip_stable():
    r = {"similarity":0.4127,"id":"img_0007","ts":"2026-09-06T11:00:00Z"}
    assert digest_hex(r) == digest_hex(json.loads(canonical_json(r).decode()))

def test_tiny_change_diverges():
    assert digest_hex({"s":0.4127}) != digest_hex({"s":0.4128})
```

```bash
pytest tests/ -v
```

All three must pass before Phase 8. Hash instability discovered after anchoring
is unrecoverable.

---

## Phase 8 — Contract (D8)

```bash
cd contracts && forge init --no-git --no-commit . && cd ..
```

### `contracts/src/MatchAnchor.sol`

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract MatchAnchor {
    struct Record { uint64 timestamp; address submitter; bool exists; }

    mapping(bytes32 => Record) private records;

    event Anchored(bytes32 indexed digest, address indexed submitter, uint64 timestamp);
    error AlreadyAnchored(bytes32 digest);

    function anchor(bytes32 digest) external {
        if (records[digest].exists) revert AlreadyAnchored(digest);
        records[digest] = Record(uint64(block.timestamp), msg.sender, true);
        emit Anchored(digest, msg.sender, uint64(block.timestamp));
    }

    function verify(bytes32 digest)
        external view
        returns (bool exists, uint64 timestamp, address submitter)
    {
        Record memory r = records[digest];
        return (r.exists, r.timestamp, r.submitter);
    }
}
```

### `contracts/test/MatchAnchor.t.sol`

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "../src/MatchAnchor.sol";

contract MatchAnchorTest is Test {
    MatchAnchor a;
    bytes32 constant D = keccak256("rec");

    function setUp() public { a = new MatchAnchor(); }

    function test_AnchorThenVerify() public {
        a.anchor(D);
        (bool e, uint64 t, address s) = a.verify(D);
        assertTrue(e); assertEq(s, address(this)); assertGt(t, 0);
    }

    function test_UnknownFails() public {
        (bool e,,) = a.verify(keccak256("nope"));
        assertFalse(e);
    }

    function test_DuplicateReverts() public {
        a.anchor(D);
        vm.expectRevert(abi.encodeWithSelector(MatchAnchor.AlreadyAnchored.selector, D));
        a.anchor(D);
    }
}
```

```bash
cd contracts && forge test -vv && cd ..
```

### Deploy

Local first:

```bash
anvil                                    # terminal 1, note a printed key
cd contracts
forge create src/MatchAnchor.sol:MatchAnchor \
  --rpc-url http://127.0.0.1:8545 --private-key <ANVIL_KEY> --broadcast
```

Then Amoy for the demo:

```bash
forge create src/MatchAnchor.sol:MatchAnchor \
  --rpc-url $AMOY_RPC_URL --private-key $PRIVATE_KEY --broadcast
```

Fund from a Polygon Amoy faucet. Use a **throwaway key created for this
project** — never a wallet holding real funds.

`.env`:

```
RPC_URL=https://rpc-amoy.polygon.technology
CONTRACT_ADDRESS=0x…
PRIVATE_KEY=0x…
MATCH_THRESHOLD=0.38
INDEX_DIR=data/index
```

### `src/anchor/chain.py`

```python
import json, os
from web3 import Web3

ABI = json.loads("""[
 {"inputs":[{"internalType":"bytes32","name":"digest","type":"bytes32"}],
  "name":"anchor","outputs":[],"stateMutability":"nonpayable","type":"function"},
 {"inputs":[{"internalType":"bytes32","name":"digest","type":"bytes32"}],
  "name":"verify","outputs":[
    {"internalType":"bool","name":"exists","type":"bool"},
    {"internalType":"uint64","name":"timestamp","type":"uint64"},
    {"internalType":"address","name":"submitter","type":"address"}],
  "stateMutability":"view","type":"function"}]""")


class ChainClient:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(os.environ["RPC_URL"]))
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(os.environ["CONTRACT_ADDRESS"]),
            abi=ABI)
        self.acct = self.w3.eth.account.from_key(os.environ["PRIVATE_KEY"])

    def anchor(self, d: bytes):
        tx = self.contract.functions.anchor(d).build_transaction({
            "from": self.acct.address,
            "nonce": self.w3.eth.get_transaction_count(self.acct.address),
            "gas": 120_000,
            "gasPrice": self.w3.eth.gas_price,
        })
        signed = self.acct.sign_transaction(tx)
        h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        r = self.w3.eth.wait_for_transaction_receipt(h, timeout=180)
        return {"tx_hash": r.transactionHash.hex(), "block": r.blockNumber,
                "gas_used": r.gasUsed, "status": r.status}

    def verify(self, d: bytes):
        e, t, s = self.contract.functions.verify(d).call()
        return {"exists": e, "timestamp": t, "submitter": s}
```

> `signed.raw_transaction` is web3.py v6+. On v5 it is `signed.rawTransaction`.

---

## Phase 9 — End-to-end demo (D9)

### `cli/demo.py`

```python
import json, sys, os, copy
from dotenv import load_dotenv
from src.search.searcher import Searcher
from src.anchor.canonical import digest, digest_hex
from src.anchor.chain import ChainClient

load_dotenv()
BAR = "=" * 64


def main(probe):
    print(BAR); print("STAGE 1+2 — encode probe, search corpus"); print(BAR)
    s = Searcher(os.environ["INDEX_DIR"], float(os.environ["MATCH_THRESHOLD"]))
    res = s.search(probe)

    if res["status"] != "match":
        print(f"status: {res['status']}")
        print(json.dumps(res, indent=2)[:800])
        print("\nNothing anchored. This is a correct outcome, not a failure.")
        return

    rec = res["record"]
    print(json.dumps(rec, indent=2))
    print("\nrunners-up:")
    for c in res["candidates"][1:4]:
        print(f"  {c['image_id']:<10} sim={c['similarity']}  {c['source_page_url']}")

    print("\n" + BAR); print("STAGE 3 — anchor on chain"); print(BAR)
    d = digest(rec)
    print("digest:", digest_hex(rec))
    chain = ChainClient()
    r = chain.anchor(d)
    print(f"tx:     {r['tx_hash']}")
    print(f"block:  {r['block']}   gas: {r['gas_used']}")
    json.dump(rec, open("data/last_record.json", "w"), indent=2)

    print("\n" + BAR); print("VERIFY — untampered"); print(BAR)
    v = chain.verify(d)
    print(f"exists={v['exists']}  ts={v['timestamp']}  by={v['submitter']}")

    print("\n" + BAR); print("VERIFY — tampered"); print(BAR)
    t = copy.deepcopy(rec)
    t["similarity"] = round(rec["similarity"] + 0.0001, 4)
    print(f"original  {rec['similarity']}  ->  {digest_hex(rec)}")
    print(f"tampered  {t['similarity']}  ->  {digest_hex(t)}")
    print(f"exists={chain.verify(digest(t))['exists']}   <-- tamper detected")

    print("\nThis proves the record is unchanged since anchoring.")
    print("It does NOT prove the match itself is correct.")


if __name__ == "__main__":
    main(sys.argv[1])
```

```bash
python -m cli.demo data/probes/probe_match.jpg
python -m cli.demo data/probes/probe_nomatch.jpg
```

---

## Phase 10 — Video and README (D10, D11)

### Recording runbook

Rehearse this end to end at least once before recording.

| # | Action | Say |
|---|---|---|
| 1 | Show `corpus/MANIFEST.md` | scope, robots, removal policy |
| 2 | Run harvester on one live page | "genuine crawl, nothing hardcoded" |
| 3 | Show `provenance.jsonl` | source URL + hash per image |
| 4 | Run index builder | detection rate, rejections logged |
| 5 | Show `docs/roc.png` | EER, chosen threshold at stated FAR |
| 6 | `cli/demo.py` on a matching probe | record, similarity, runners-up |
| 7 | Show tx on Amoy block explorer | independently checkable |
| 8 | Tamper section of the same output | both digests, `exists=False` |
| 9 | `cli/demo.py` on a non-corpus probe | "it can say no" |
| 10 | README limitations | bias, base rates, integrity ≠ correctness |

**Pre-flight:** weights downloaded, wallet funded, explorer tab open, `.env`
correct, both probes tested, terminal font large enough to read.

Steps 8 and 9 are what distinguish this from an average submission. Do not cut
them for time.

### README limitations section

Include, in your own words:

- **Demographic bias** — NIST FRVT Part 3 documents error rates varying by
  10–100× across demographic groups depending on algorithm. A single global
  threshold does not perform uniformly. State your corpus composition.
- **Base rates** — precision degrades as corpus size grows; at FAR 1e-6 against
  a billion faces you get ~1,000 false matches per query. Corpus-bounded search
  is sounder engineering, not just safer.
- **Integrity ≠ correctness** — the chain protects the record, not the claim.
- **No liveness** — a printed photo passes Stage 1. ISO/IEC 30107-3 defines the
  standard countermeasure family; out of scope here, named as future work.
- **Scraped names are heuristic** — used only for evaluation labelling, never
  asserted as verified identity.
- **Key custody** — the chain attests to a submitting address, not to any
  real-world identity behind it.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| onnxruntime install fails | Python 3.12+ | use 3.10/3.11 |
| Harvester finds 0 people | site structure unmatched | write a per-site adapter |
| All similarities ≈ 0.99 | probe is the indexed image | use a different photo |
| All similarities ≈ 0 | wrong index type or unnormalised vectors | confirm `IndexFlatIP` + unit norm |
| Too few genuine pairs | one image per person | add targets where people appear twice |
| verify() false right after anchoring | canonicalisation drift | rerun Phase 7 tests; check float precision |
| `nonce too low` | concurrent txs | serialise anchoring |
| `insufficient funds` | empty wallet | refill from faucet |
| `raw_transaction` AttributeError | web3.py v5 | use `rawTransaction` |

---

## Optional extension

If you want a second phase after submission: the same primitives — canonical
serialisation, hashing, signatures — are what **C2PA / Content Credentials**
uses for media provenance. Anchoring "this image is what it claims to be, with
this edit history" is the defensive application of everything built here, and it
reuses Stages 1 and 3 almost unchanged.
