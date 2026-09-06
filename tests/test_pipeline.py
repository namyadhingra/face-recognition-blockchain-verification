import hashlib
import json

from src.blockchain import SimulatedBlockchain
from src.pipeline import canonicalize_record, evaluate_match, hash_record, verify_record_digest


def test_canonicalize_and_hash_roundtrip():
    record = {
        "schema_version": "1.0",
        "probe_sha256": "abc",
        "matched_image_id": "img_001",
        "matched_source_url": "https://example.edu/team/1",
        "matched_image_sha256": "def",
        "similarity": 0.4127,
        "threshold": 0.38,
        "model": "insightface/buffalo_l",
        "index_version": "corpus-2026-09-06",
        "matched_at": "2026-09-06T11:04:22Z",
    }

    canonical = canonicalize_record(record)
    assert isinstance(canonical, str)
    assert canonical == json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    digest = hash_record(record)
    assert len(digest) == 64
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert verify_record_digest(record, digest)

    tampered = dict(record, similarity=0.999)
    assert not verify_record_digest(tampered, digest)
    assert evaluate_match(record) is True
    assert evaluate_match(tampered) is True


def test_blockchain_anchor_verification():
    chain = SimulatedBlockchain()
    record = {
        "schema_version": "1.0",
        "probe_sha256": "abc",
        "matched_image_id": "img_002",
        "matched_source_url": "https://example.edu/lab/team",
        "matched_image_sha256": "xyz",
        "similarity": 0.44,
        "threshold": 0.38,
        "model": "insightface/buffalo_l",
        "index_version": "corpus-2026-09-06",
        "matched_at": "2026-09-06T11:00:00Z",
    }
    digest = hash_record(record)
    tx_hash = chain.anchor_record(digest, record)
    assert chain.verify_digest(tx_hash, digest) is True
    assert chain.verify_record(tx_hash, record, hash_record) is True
    tampered = dict(record, similarity=0.11)
    assert chain.verify_record(tx_hash, tampered, hash_record) is False
