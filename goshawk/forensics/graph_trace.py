"""Directed financial transaction graph analyzer for money laundering and fraud detection."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .ledger_parser import TransactionRecord


class TransactionFlowGraph:
    """Constructs and queries directed transaction flow networks across accounts and mobile wallets."""

    def __init__(self):
        self.adjacency: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)
        self.node_totals_in: Dict[str, int] = defaultdict(int)
        self.node_totals_out: Dict[str, int] = defaultdict(int)

    def add_transaction(self, sender: str, receiver: str, amount: int, reference: str) -> None:
        """Add directed financial edge from sender to receiver."""
        self.adjacency[sender].append((receiver, amount, reference))
        self.node_totals_out[sender] += amount
        self.node_totals_in[receiver] += amount

    def build_from_records(self, records: List[TransactionRecord], primary_account: str) -> None:
        """Populate graph from a parsed list of TransactionRecords."""
        for rec in records:
            if rec.direction == "DEBIT":
                # Money moved from primary account to counterparty
                self.add_transaction(
                    sender=primary_account,
                    receiver=rec.counterparty_identifier or rec.counterparty_name,
                    amount=rec.amount,
                    reference=rec.reference,
                )
            else:
                # Money moved from counterparty to primary account
                self.add_transaction(
                    sender=rec.counterparty_identifier or rec.counterparty_name,
                    receiver=primary_account,
                    amount=rec.amount,
                    reference=rec.reference,
                )

    def detect_rapid_dispersal(self, target_node: str, threshold: float = 0.85) -> bool:
        """
        Detects potential structuring or funneling:
        If total outflows roughly equal total inflows within a narrow window.
        """
        in_amt = self.node_totals_in.get(target_node, 0)
        out_amt = self.node_totals_out.get(target_node, 0)
        if in_amt == 0 or out_amt == 0:
            return False
        ratio = min(in_amt, out_amt) / max(in_amt, out_amt)
        return ratio >= threshold

    def get_summary(self) -> Dict[str, Any]:
        """Summarize total nodes, edges, and high-volume participants."""
        nodes = set(self.node_totals_in.keys()).union(set(self.node_totals_out.keys()))
        return {
            "total_parties": len(nodes),
            "total_transfers": sum(len(edges) for edges in self.adjacency.values()),
            "top_senders": sorted(self.node_totals_out.items(), key=lambda x: x[1], reverse=True)[:5],
            "top_receivers": sorted(self.node_totals_in.items(), key=lambda x: x[1], reverse=True)[:5],
        }
