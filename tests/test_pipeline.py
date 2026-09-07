"""Tests for the face-match provenance pipeline.

Covers: canonical serialisation, hashing, tamper detection, blockchain
anchoring, chain integrity, and match evaluation.
No external API keys or InsightFace required.
"""

import hashlib
import json

from src.blockchain import SimulatedBlockchain
from src.pipeline import (
    build_match_record,
    canonicalize_record,
    evaluate_match,
    hash_record,
    verify_record_digest,
)


# ── Canonical serialisation ──────────────────────────────────────

def test_canonicalize_deterministic():
    """Key insertion order does not affect canonical output."""
    assert canonicalize_record({"b": 2, "a": 1}) == canonicalize_record({"a": 1, "b": 2})


def test_canonicalize_compact():
    """No insignificant whitespace in canonical JSON."""
    c = canonicalize_record({"key": "value", "num": 42})
    assert " " not in c


# ── Hashing ──────────────────────────────────────────────────────

def test_hash_equals_sha256_of_canonical():
    record = build_match_record(
        probe_sha256="abc", matched_url="https://example.edu/team",
        matched_title="Lab", similarity=0.4127, threshold=0.38,
        matched_at="2026-09-06T11:04:22Z",
    )
    canonical = canonicalize_record(record)
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert hash_record(record) == expected


def test_key_order_irrelevant():
    assert hash_record({"b": 2, "a": 1}) == hash_record({"a": 1, "b": 2})


def test_round_trip_stable():
    r = {"similarity": 0.4127, "id": "img_0007", "ts": "2026-09-06T11:00:00Z"}
    canonical = canonicalize_record(r)
    reparsed = json.loads(canonical)
    assert hash_record(r) == hash_record(reparsed)


def test_tiny_change_diverges():
    assert hash_record({"s": 0.4127}) != hash_record({"s": 0.4128})


# ── Tamper detection ─────────────────────────────────────────────

def test_verify_untampered():
    record = build_match_record(
        probe_sha256="abc", matched_url="https://example.edu/team",
        matched_title="Team", similarity=0.4127, threshold=0.38,
        matched_at="2026-09-06T11:04:22Z",
    )
    assert verify_record_digest(record, hash_record(record)) is True


def test_verify_tampered():
    record = build_match_record(
        probe_sha256="abc", matched_url="https://example.edu/team",
        matched_title="Team", similarity=0.4127, threshold=0.38,
        matched_at="2026-09-06T11:04:22Z",
    )
    digest = hash_record(record)
    tampered = dict(record, similarity=0.999)
    assert verify_record_digest(tampered, digest) is False


# ── Blockchain ───────────────────────────────────────────────────

def test_blockchain_has_genesis():
    chain = SimulatedBlockchain()
    assert len(chain.chain) == 1
    assert chain.chain[0].index == 0


def test_blockchain_anchor_and_verify():
    chain = SimulatedBlockchain()
    record = build_match_record(
        probe_sha256="abc", matched_url="https://example.edu/lab",
        matched_title="Vision Lab", similarity=0.44, threshold=0.38,
        matched_at="2026-09-06T11:00:00Z",
    )
    digest = hash_record(record)
    tx = chain.anchor_record(digest, record)

    assert chain.verify_digest(tx, digest) is True
    assert chain.verify_record(tx, record, hash_record) is True


def test_blockchain_tamper_rejected():
    chain = SimulatedBlockchain()
    record = build_match_record(
        probe_sha256="abc", matched_url="https://example.edu/lab",
        matched_title="Lab", similarity=0.44, threshold=0.38,
        matched_at="2026-09-06T11:00:00Z",
    )
    tx = chain.anchor_record(hash_record(record), record)
    tampered = dict(record, similarity=0.11)
    assert chain.verify_record(tx, tampered, hash_record) is False


def test_blockchain_unknown_tx():
    chain = SimulatedBlockchain()
    assert chain.verify_digest("0xnonexistent", "abc") is False


def test_blockchain_chain_valid():
    chain = SimulatedBlockchain()
    record = build_match_record(
        probe_sha256="test", matched_url="https://x.com/post",
        matched_title="Post", similarity=0.5, threshold=0.38,
        matched_at="2026-09-06T12:00:00Z",
    )
    chain.anchor_record(hash_record(record), record)
    assert chain.is_chain_valid() is True


def test_blockchain_multiple_anchors():
    chain = SimulatedBlockchain()
    for i in range(5):
        r = {"id": i, "data": f"record_{i}"}
        chain.anchor_record(hash_record(r), r)
    assert len(chain.chain) == 6  # genesis + 5
    assert chain.is_chain_valid() is True


def test_blockchain_block_info():
    chain = SimulatedBlockchain()
    record = build_match_record(
        probe_sha256="xyz", matched_url="https://example.com",
        matched_title="Page", similarity=0.45, threshold=0.38,
        matched_at="2026-09-06T13:00:00Z",
    )
    tx = chain.anchor_record(hash_record(record), record)
    info = chain.get_block_info(tx)
    assert info is not None
    assert info["block_index"] == 1
    assert info["chain_valid"] is True
    assert info["chain_length"] == 2


# ── Match evaluation ─────────────────────────────────────────────

def test_evaluate_above_threshold():
    assert evaluate_match({"similarity": 0.5, "threshold": 0.38}) is True

def test_evaluate_below_threshold():
    assert evaluate_match({"similarity": 0.2, "threshold": 0.38}) is False

def test_evaluate_at_threshold():
    assert evaluate_match({"similarity": 0.38, "threshold": 0.38}) is True
