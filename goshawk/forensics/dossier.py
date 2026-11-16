"""Automated legal and digital forensics dossier generator."""

from dataclasses import dataclass, field
from typing import Dict, List

from .graph_trace import TransactionFlowGraph
from .ledger_parser import ForensicLedgerParser, TransactionRecord
from .statute_verifier import LegalStatuteVerifier


@dataclass
class ForensicDossier:
    """Structured investigation dossier for East African legal and financial proceedings."""

    primary_account: str
    channel: str
    total_records: int
    total_inflow: int
    total_outflow: int
    rapid_dispersal_flagged: bool
    applicable_statutes: List[Dict[str, str]] = field(default_factory=list)
    admissibility_checklist: List[str] = field(default_factory=list)
    records: List[TransactionRecord] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render investigation dossier as a structured Markdown report."""
        statute_lines = "\n".join(
            f"- **{s['act']} ({s['section']})**: {s['title']}"
            for s in self.applicable_statutes
        )
        checklist_lines = "\n".join(f"- [ ] {item}" for item in self.admissibility_checklist)
        status = "CRITICAL - RAPID DISPERSAL / SMURFING DETECTED" if self.rapid_dispersal_flagged else "NORMAL"

        return (
            f"# Forensic Investigation Dossier: {self.primary_account}\n\n"
            f"- **Channel**: {self.channel}\n"
            f"- **Transactions Parsed**: {self.total_records}\n"
            f"- **Total Inflow**: {self.total_inflow:,}\n"
            f"- **Total Outflow**: {self.total_outflow:,}\n"
            f"- **Risk Assessment**: {status}\n\n"
            f"## Applicable Statutory Framework\n{statute_lines}\n\n"
            f"## Section 40A / 40B Electronic Evidence Checklist\n{checklist_lines}\n"
        )


def build_forensic_dossier(
    statement_text: str,
    primary_account: str = "PRIMARY_ACCOUNT",
    channel: str = "MPESA",
    dispersal_threshold: float = 0.85,
) -> ForensicDossier:
    """Parse ledger text, construct transaction graph, and generate legal dossier."""
    parser = ForensicLedgerParser()
    records = parser.parse_statement_text(statement_text, default_channel=channel)

    graph = TransactionFlowGraph()
    graph.build_from_records(records, primary_account=primary_account)

    inflow = graph.node_totals_in.get(primary_account, 0)
    outflow = graph.node_totals_out.get(primary_account, 0)
    flagged = graph.detect_rapid_dispersal(primary_account, threshold=dispersal_threshold)

    statute_codes = ["TEA_40A", "PC_302", "CYBER_14"] if flagged else ["TEA_40A"]
    statutes: List[Dict[str, str]] = []
    for code in statute_codes:
        st = LegalStatuteVerifier.get_statute(code)
        if st:
            statutes.append({"act": st.act, "section": st.section, "title": st.title})

    checklist = [
        "Verify mobile money / bank server operational integrity during transaction window.",
        "Execute Certificate of Electronic Evidence under Tanzania Evidence Act Section 40B.",
        "Preserve cryptographic hash (SHA-256) of raw SMS / CSV statement export.",
    ]

    return ForensicDossier(
        primary_account=primary_account,
        channel=channel,
        total_records=len(records),
        total_inflow=inflow,
        total_outflow=outflow,
        rapid_dispersal_flagged=flagged,
        applicable_statutes=statutes,
        admissibility_checklist=checklist,
        records=records,
    )
