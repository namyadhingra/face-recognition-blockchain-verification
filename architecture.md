# Face-Match Provenance Pipeline — Architecture

## 1. What this system does

Given a probe photograph of a person, the system searches a corpus harvested
from **public institutional profile pages** for other appearances of the same
person, and anchors any match onto a blockchain as a tamper-evident record.

Output claim:

> "At time T, this system asserted that the face in probe P matched a face at
> URL Q with similarity S, using index version V."

That claim is hashed and written on-chain. Anyone can recompute the hash later
and confirm the record is unaltered.

**State this distinction explicitly in your demo and report:** the blockchain
proves the record has not changed since anchoring. It does not prove the match
was correct. An anchored false positive is a permanently recorded false
positive. Conflating integrity with correctness is the most common conceptual
error in projects of this shape, and calling it out yourself is a strong signal
to a grader.

---

## 2. Corpus scope

### In scope

Public institutional profile pages where a photograph is published alongside a
name, by an institution, for the express purpose of being found:

- University lab and research group "Our Team" pages
- Faculty and departmental directories
- Conference speaker and programme committee listings
- Public organisation staff pages
- Personal academic homepages

These are published deliberately, with names attached, to be discovered. That is
what makes them defensible as a corpus.

### Out of scope, and why

**Social media accounts, including public ones and business accounts.** The
argument that a business account "wants to be discovered" is true but describes
a different capability. Such an account seeks discovery by name, product,
location, or hashtag. Face-searchability is not that. Consent to be findable by
name is not consent to be enrolled in a biometric index.

Independently of that reasoning, the binding constraint is contractual:
Instagram, Facebook, X, LinkedIn and TikTok all prohibit automated collection
and biometric processing in their terms of service. Public visibility is not a
defence to a terms violation.

Such accounts also contain many faces other than the account holder's —
customers, staff, family — who have consented under no reading at all.

### The lookup / enrolment distinction

This is the actual line, and it is worth understanding:

- **Lookup** — using a team page to work out who is who in a group photo. Close
  to the page's stated purpose.
- **Enrolment** — indexing everyone from many such pages into a biometric
  database queryable with arbitrary probes. This converts "I agreed to be listed
  on my lab's site" into "I am in a face search system," which nobody agreed to.

The technical difference is a loop bound. The difference in kind is large. This
project sits deliberately on the lookup side, enforced by three design choices:
a small fixed corpus, no open query endpoint, and a scripted removal procedure.

### Legal position (India)

DPDP Act 2023 §3(c)(ii) exempts personal data the data principal has themselves
made publicly available. An academic's institutional profile page is a
reasonable fit. SPDI Rules 2011 under IT Act §43A classify biometric information
as sensitive personal data, so the exemption matters.

This is jurisdiction-specific. GDPR Art. 9 offers no equivalent "but it was
public" defence for biometric data — publicness was never what saved Clearview
AI in any European enforcement action. Do not generalise this design outside
India without rechecking.

### Operating constraints

| Constraint | Rule |
|---|---|
| Corpus size | 20–60 identities. Class-project scale, not database scale. |
| robots.txt | Check and honour before fetching any domain. |
| Rate limiting | ≥2s between requests; identify your bot in User-Agent. |
| Provenance | Source URL + fetch timestamp for every image. |
| Removal | Publish a contact; honour requests; script the deletion. |
| Query surface | No open endpoint. Probes come from the corpus population. |

---

## 3. Pipeline

```
┌──────────────────────────────────────────────┐
│ PHASE A — Corpus harvest (offline, one-off)   │
│  robots.txt check → fetch profile pages       │
│  → extract (image, name, source URL)          │
│  → provenance manifest                        │
└───────────────────┬──────────────────────────┘
                    ▼
┌──────────────────────────────────────────────┐
│ PHASE B — Index build (offline)               │
│  detect → align → quality gate → embed        │
│  → FAISS IndexFlatIP                          │
└───────────────────┬──────────────────────────┘
                    ▼
   probe image
        │
        ▼
┌──────────────────────────────────────────────┐
│ STAGE 1 — Encode probe                        │
│  SCRFD detect → 5-pt align → ArcFace 512-d    │
└───────────────────┬──────────────────────────┘
                    ▼
┌──────────────────────────────────────────────┐
│ STAGE 2 — Search                              │
│  ANN top-k → calibrated threshold             │
│  → MatchRecord or explicit no-match           │
└───────────────────┬──────────────────────────┘
                    ▼
┌──────────────────────────────────────────────┐
│ STAGE 3 — Anchor                              │
│  canonical JSON → SHA-256 → anchor(digest)    │
└───────────────────┬──────────────────────────┘
                    ▼
┌──────────────────────────────────────────────┐
│ VERIFY — recompute digest → verify() on chain │
└──────────────────────────────────────────────┘
```

---

## 4. Phase A — Corpus harvest

This is the web-search component. It is a genuine crawl of live public pages;
nothing is pre-selected or hardcoded.

**Per target domain:**

1. Fetch and parse `/robots.txt`. If the profile path is disallowed, drop the
   domain and record why in the manifest.
2. Fetch the profile listing page.
3. Extract candidate entries — image URL, associated name text, page URL.
4. Download images, respecting rate limits.
5. Write one provenance row per image.

**Extraction heuristics.** Team pages follow no standard schema, so expect a
small adapter per site. Patterns worth trying in order:

- schema.org `Person` structured data, where present — cleanest by far
- `<figure>` / `<figcaption>` pairs
- `alt` attribute containing the person's name
- `<img>` inside a card whose sibling text holds a name

Do not over-engineer a general extractor. Three or four site-specific adapters
is the right amount of work at this scale, and it is more honest about what the
system actually does.

**Provenance record:**

```json
{
  "image_id": "img_0007",
  "source_page_url": "https://example.edu/lab/team",
  "image_url": "https://example.edu/static/people/asha.jpg",
  "extracted_name": "Asha Verma",
  "fetched_at": "2026-09-06T10:12:03Z",
  "robots_allowed": true,
  "image_sha256": "…"
}
```

---

## 5. Stages 1–2 — Encode and search

### Encoding

- **Detect** — SCRFD via InsightFace; handles pose far better than HOG/dlib.
- **Align** — 5-point landmark affine warp to 112×112. Skipping this measurably
  degrades ArcFace accuracy. Not optional.
- **Quality gate** — reject and log:

| Check | Threshold | Reason |
|---|---|---|
| bbox width | < 60 px | insufficient detail |
| Laplacian variance | < 50 | blur |
| detection score | < 0.6 | probably not a face |

- **Embed** — ArcFace `buffalo_l` → 512-d, L2-normalised. Once normalised, inner
  product equals cosine similarity, which is what `IndexFlatIP` computes.

Log rejections rather than dropping silently. "Why did it not find my face" is
the most common demo failure; the log answers it instantly.

### Index

`IndexFlatIP` — exact, no training, trivially fast at this scale. Do not reach
for HNSW at 200 vectors; you would add tuning knobs that accomplish nothing.

### Threshold calibration

A top-1 result is meaningless without a calibrated threshold. This section
carries the marks.

1. Compute all pairwise similarities within the corpus.
2. Label pairs genuine (same extracted name) or impostor (different).
3. Sweep threshold, plot ROC, report FAR / FRR / EER.
4. Choose an operating point at a stated FAR and justify it.

ArcFace `buffalo_l` typically lands near 0.35–0.45 cosine, but **this is
dataset-dependent**. Quoting a number from a blog post rather than measuring it
on your corpus is precisely the error calibration exists to prevent.

You need 2+ images per identity for genuine pairs. Where a person appears on
both a lab page and a conference page you get one free — prefer targets with
that overlap.

### MatchRecord

```json
{
  "schema_version": "1.0",
  "probe_sha256": "…",
  "matched_image_id": "img_0007",
  "matched_source_url": "https://example.edu/lab/team",
  "matched_image_sha256": "…",
  "similarity": 0.4127,
  "threshold": 0.38,
  "model": "insightface/buffalo_l",
  "index_version": "corpus-2026-09-06",
  "matched_at": "2026-09-06T11:04:22Z"
}
```

The record carries a URL and hashes — not a name, not an embedding. Below
threshold, emit an explicit no-match record, and **demonstrate that path**. A
system that always finds something is a system nobody should trust.

---

## 6. Stage 3 — Blockchain anchoring

### Rule: only the digest goes on-chain

No images, names, embeddings, or URLs.

- **Legal.** On-chain data is immutable. Personal data on an immutable ledger
  creates an erasure obligation you cannot satisfy — DPDP §12 grants a right to
  erasure, as does GDPR Art. 17.
- **A face embedding is not anonymous.** It is a biometric identifier. Do not
  put embeddings on-chain on the theory that "it's just numbers."
- **Cost and privacy.** 32 bytes is cheap; public chains are permanently
  readable by everyone.

### Canonicalisation

Hash reproducibility depends entirely on byte-identical serialisation:

- UTF-8, keys sorted lexicographically
- No insignificant whitespace — `separators=(',', ':')`
- Floats rounded to fixed precision **when building the record**, not at
  serialisation time
- Timestamps RFC 3339 UTC with `Z`

`digest = sha256(canonical_json_bytes)`. Getting this wrong produces
"verification fails and I don't know why." Unit-test the round trip.

### Contract

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

Reverting on duplicates makes replay explicit rather than silently overwriting a
timestamp. `block.timestamp` is validator-influenced within seconds — fine for
provenance ordering, not for precise timing.

### Chain choice

Develop against **Anvil** locally. Demo against **Polygon Amoy** — roughly 2s
blocks versus Sepolia's ~12s, which matters a lot on a recorded video. Put the
contract address and block explorer link in your README so the on-chain claim is
checkable in one click.

---

## 7. Tamper-evidence demo

The payoff. Script it.

1. Load stored MatchRecord, recompute digest.
2. `verify(digest)` → `(true, ts, submitter)`. Show the explorer link.
3. Mutate one field — `similarity` 0.4127 → 0.4128.
4. Recompute → an entirely different digest. Print both side by side.
5. `verify(new_digest)` → `(false, 0, 0x0)`.

Then say plainly: this demonstrates integrity, not correctness.

---

## 8. Repository layout

```
face-provenance/
├── corpus/
│   ├── MANIFEST.md          # targets, robots status, removal contact
│   └── adapters/            # per-site extractors
├── contracts/
│   ├── src/MatchAnchor.sol
│   └── test/MatchAnchor.t.sol
├── src/
│   ├── harvest/             # robots, fetch, extract, provenance
│   ├── ingest/              # detect, align, gate, embed
│   ├── index/               # FAISS build
│   ├── search/              # query, threshold, MatchRecord
│   ├── anchor/              # canonicalise, hash, web3
│   └── eval/                # ROC, FAR/FRR
├── cli/
├── tests/
├── data/                    # gitignored
└── docs/
```

---

## 9. Evaluation

| Metric | Source |
|---|---|
| Pages crawled / images harvested | harvest log |
| Detection rate | % harvested images yielding a usable face |
| EER | pairwise score sweep |
| FAR / FRR @ operating threshold | ROC |
| Rank-1 accuracy | correct identity is top-1 |
| End-to-end latency | per stage |
| Gas per anchor | testnet receipt |

Include **failure cases with images** — a rejected blurry crop, a
near-threshold ambiguous pair, a false positive if you find one. A report
showing only successes reads as untested.

---

## 10. Limitations to state in the writeup

- **Demographic bias.** NIST FRVT Part 3 documents error rates varying by
  10–100× across demographic groups depending on algorithm. A single global
  threshold does not perform uniformly. If your corpus is demographically
  narrow, say so.
- **Base rates.** Precision degrades as corpus size grows. At FAR = 1e-6 against
  a billion faces you would see ~1,000 false matches per query. This is the
  technical reason corpus-bounded search is a sounder design, not merely a safer
  one — worth stating, since it reframes scope as engineering judgement rather
  than as a limitation you are apologising for.
- **Integrity ≠ correctness.**
- **Corpus-bounded recall.** Deliberate.
- **No liveness.** A printed photo passes Stage 1.
- **Key custody.** The chain attests to a submitting address, not to any
  real-world identity behind it.
- **Name extraction is heuristic.** Scraped names are best-effort, used only for
  evaluation labelling, never asserted as verified identity.
