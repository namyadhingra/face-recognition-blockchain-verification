import json
from typing import Any, Dict, Optional


class SimulatedBlockchain:
    """A local blockchain-style verifier that demonstrates the required behavior.

    This stores anchored digests and can verify that a record digest matches the
    chain entry. It is intentionally simple and suitable for a class project.
    """

    def __init__(self):
        self.records: Dict[str, Dict[str, Any]] = {}

    def anchor_record(self, record_digest: str, record: Dict[str, Any]) -> str:
        tx_hash = "0x" + (record_digest[:16] + "0000000000000000")
        self.records[tx_hash] = {"record_digest": record_digest, "record": record}
        return tx_hash

    def verify_digest(self, tx_hash: str, record_digest: str) -> bool:
        entry = self.records.get(tx_hash)
        if entry is None:
            return False
        return entry.get("record_digest") == record_digest

    def verify_record(self, tx_hash: str, record: Dict[str, Any], digest_fn) -> bool:
        entry = self.records.get(tx_hash)
        if entry is None:
            return False
        digest = digest_fn(record)
        return entry.get("record_digest") == digest
