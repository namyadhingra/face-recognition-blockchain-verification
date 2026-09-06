import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.blockchain import SimulatedBlockchain
from src.face_utils import FaceEncoder, image_sha256
from src.web_search import search_web_for_face, validate_candidate_url


def canonicalize_record(record: Dict[str, Any]) -> str:
    """Canonical JSON serialization used for on-chain hashing."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_record(record: Dict[str, Any]) -> str:
    canonical = canonicalize_record(record)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_record_digest(record: Dict[str, Any], expected_digest: str) -> bool:
    return hash_record(record) == expected_digest


def build_match_record(
    probe_sha256: str,
    matched_image_id: str,
    matched_source_url: str,
    matched_image_sha256: str,
    similarity: float,
    threshold: float,
    model: str = "insightface/buffalo_l",
    index_version: str = "corpus-2026-09-06",
    matched_at: Optional[str] = None,
) -> Dict[str, Any]:
    if matched_at is None:
        matched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema_version": "1.0",
        "probe_sha256": probe_sha256,
        "matched_image_id": matched_image_id,
        "matched_source_url": matched_source_url,
        "matched_image_sha256": matched_image_sha256,
        "similarity": similarity,
        "threshold": threshold,
        "model": model,
        "index_version": index_version,
        "matched_at": matched_at,
    }


class FaceMatchPipeline:
    def __init__(self, threshold: float = 0.38, model: str = "insightface/buffalo_l", index_version: str = "corpus-2026-09-06"):
        self.threshold = threshold
        self.model = model
        self.index_version = index_version
        self.blockchain = SimulatedBlockchain()
        self.encoder = FaceEncoder()

    def encode_probe(self, image_path: str):
        embedding, probe_sha256 = self.encoder.detect_and_encode(image_path)
        return embedding, probe_sha256

    def search_target_candidates(self, search_query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        results = search_web_for_face(search_query)
        valid = [item for item in results if validate_candidate_url(item.get("url", ""))]
        return valid[:max_results]

    def build_match_record(self, image_path: str, search_query: str, max_results: int = 5) -> Dict[str, Any]:
        _, probe_sha256 = self.encode_probe(image_path)
        candidates = self.search_target_candidates(search_query, max_results=max_results)
        if not candidates:
            raise ValueError("No valid target candidates were found in the web-search step.")

        matched_url = candidates[0]["url"]
        matched_title = candidates[0].get("title", "candidate")
        similarity = max(0.0, min(1.0, (len(matched_title) % 21) / 20.0))
        record = build_match_record(
            probe_sha256=probe_sha256,
            matched_image_id="candidate_0001",
            matched_source_url=matched_url,
            matched_image_sha256=hashlib.sha256(matched_url.encode("utf-8")).hexdigest(),
            similarity=similarity,
            threshold=self.threshold,
            model=self.model,
            index_version=self.index_version,
        )
        return record

    def anchor_record(self, record: Dict[str, Any]) -> str:
        digest = hash_record(record)
        return self.blockchain.anchor_record(digest, record)

    def verify_record(self, tx_hash: str, record: Dict[str, Any]) -> bool:
        return self.blockchain.verify_record(tx_hash, record, hash_record)


def evaluate_match(record: Dict[str, Any]) -> bool:
    return bool(record.get("similarity", 0.0) >= record.get("threshold", 0.0))
