"""Simulated Blockchain Ledger for tamper-evident record anchoring.

Implements a local blockchain with proper block structure:
- Genesis block with chain initialisation
- Each block contains: index, timestamp, data hash, previous block hash, nonce
- Blocks are chained via SHA-256 of the previous block
- Chain integrity verification walks the entire chain

This is a simulated chain — it runs locally and demonstrates the
tamper-evidence concept without requiring a real network.

Key insight demonstrated here: if you change even one byte of the
anchored record, its SHA-256 digest changes, and verify() returns False.
The chain is immutable *within* this process. A real blockchain extends
this guarantee across untrusted parties.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional


class Block:
    """A single block in the simulated chain."""

    def __init__(
        self,
        index: int,
        timestamp: float,
        data_digest: str,
        previous_hash: str,
        nonce: int = 0,
    ):
        self.index = index
        self.timestamp = timestamp
        self.data_digest = data_digest
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        block_string = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "data_digest": self.data_digest,
                "previous_hash": self.previous_hash,
                "nonce": self.nonce,
            },
            sort_keys=True,
        )
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data_digest": self.data_digest,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash,
        }


class SimulatedBlockchain:
    """A local blockchain ledger for anchoring match record digests.

    Usage::

        chain = SimulatedBlockchain()
        tx = chain.anchor_record(digest_hex, record_dict)
        assert chain.verify_digest(tx, digest_hex)
        assert chain.is_chain_valid()
    """

    DIFFICULTY = 2  # number of leading zeros required (kept low for speed)

    def __init__(self):
        self.chain: List[Block] = []
        self.records: Dict[str, Dict[str, Any]] = {}
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis = Block(
            index=0,
            timestamp=time.time(),
            data_digest="0" * 64,
            previous_hash="0" * 64,
        )
        self.chain.append(genesis)

    def _proof_of_work(self, block: Block) -> Block:
        """Simple proof of work: find a nonce that gives a hash with
        DIFFICULTY leading zeros. Kept computationally trivial."""
        prefix = "0" * self.DIFFICULTY
        while not block.hash.startswith(prefix):
            block.nonce += 1
            block.hash = block.compute_hash()
        return block

    def _add_block(self, data_digest: str) -> Block:
        previous = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            data_digest=data_digest,
            previous_hash=previous.hash,
        )
        new_block = self._proof_of_work(new_block)
        self.chain.append(new_block)
        return new_block

    # ── Public API ──────────────────────────────────────────────

    def anchor_record(self, record_digest: str, record: Dict[str, Any]) -> str:
        """Anchor a record digest on the chain.

        Returns a transaction hash (the block hash) for later verification.
        """
        block = self._add_block(record_digest)
        tx_hash = "0x" + block.hash
        self.records[tx_hash] = {
            "record_digest": record_digest,
            "record": record,
            "block_index": block.index,
            "block_hash": block.hash,
            "timestamp": block.timestamp,
        }
        return tx_hash

    def verify_digest(self, tx_hash: str, record_digest: str) -> bool:
        """Check if a digest matches the one stored at tx_hash."""
        entry = self.records.get(tx_hash)
        if entry is None:
            return False
        return entry["record_digest"] == record_digest

    def verify_record(self, tx_hash: str, record: Dict[str, Any], digest_fn) -> bool:
        """Recompute the digest of a record and verify against the chain."""
        entry = self.records.get(tx_hash)
        if entry is None:
            return False
        return entry["record_digest"] == digest_fn(record)

    def is_chain_valid(self) -> bool:
        """Walk the entire chain and verify hash linkage + proof of work."""
        prefix = "0" * self.DIFFICULTY
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Verify current block's hash
            if current.hash != current.compute_hash():
                return False

            # Verify chain linkage
            if current.previous_hash != previous.hash:
                return False

            # Verify proof of work
            if not current.hash.startswith(prefix):
                return False

        return True

    def get_block_info(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Return block details for a given transaction hash."""
        entry = self.records.get(tx_hash)
        if entry is None:
            return None
        block = self.chain[entry["block_index"]]
        return {
            "block_index": block.index,
            "block_hash": block.hash,
            "previous_hash": block.previous_hash,
            "timestamp": block.timestamp,
            "nonce": block.nonce,
            "data_digest": block.data_digest,
            "chain_length": len(self.chain),
            "chain_valid": self.is_chain_valid(),
        }

    def get_chain_summary(self) -> List[Dict[str, Any]]:
        """Return a summary of the entire chain (for demo display)."""
        return [block.to_dict() for block in self.chain]
