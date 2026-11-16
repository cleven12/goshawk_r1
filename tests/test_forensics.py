"""Tests for East African forensic ledger extraction, graph analysis, and statutory verification."""


from goshawk.forensics.graph_trace import TransactionFlowGraph
from goshawk.forensics.ledger_parser import ForensicLedgerParser
from goshawk.forensics.statute_verifier import LegalStatuteVerifier


def test_mpesa_parser():
    raw_sms = (
        "QJ94KD810Z Confirmed. You have received TZS 1,500,000 from JOHN KWEKA +255754123456 "
        "on 14/11/2026 at 10:14 PM. New M-Pesa balance is TZS 2,850,000."
    )
    parser = ForensicLedgerParser()
    records = parser.parse_statement_text(raw_sms)

    assert len(records) == 1
    rec = records[0]
    assert rec.reference == "QJ94KD810Z"
    assert rec.amount == 1500000
    assert rec.currency == "TZS"
    assert rec.direction == "CREDIT"
    assert "JOHN KWEKA" in rec.counterparty_name
    assert rec.counterparty_identifier == "+255754123456"
    assert rec.balance == 2850000


def test_transaction_graph_and_rapid_dispersal():
    graph = TransactionFlowGraph()
    primary = "+255700111222"

    # Inflow: 10,000,000
    graph.add_transaction("SOURCE_A", primary, 10000000, "REF001")

    # Immediate outflows of 9,500,000
    graph.add_transaction(primary, "TARGET_1", 3000000, "REF002")
    graph.add_transaction(primary, "TARGET_2", 3000000, "REF003")
    graph.add_transaction(primary, "TARGET_3", 3500000, "REF004")

    # Ratio is 9,500,000 / 10,000,000 = 0.95 >= 0.85
    assert graph.detect_rapid_dispersal(primary, threshold=0.85) is True

    summary = graph.get_summary()
    assert summary["total_parties"] == 5
    assert summary["total_transfers"] == 4


def test_statute_verifier():
    # Test valid statute retrieval
    statute_pc = LegalStatuteVerifier.get_statute("PC_302")
    assert statute_pc is not None
    assert "False Pretences" in statute_pc.title
    assert "Cap 16" in statute_pc.act

    # Test Section 40A electronic evidence
    statute_tea = LegalStatuteVerifier.get_statute("TEA_40A")
    assert statute_tea is not None
    assert "Electronic and Digital Records" in statute_tea.title

    # Test citation verification
    assert LegalStatuteVerifier.verify_citation("Penal Code", "Section 265") is True
    assert LegalStatuteVerifier.verify_citation("Evidence Act", "Section 40A") is True
    assert LegalStatuteVerifier.verify_citation("Unknown Act", "Section 999") is False
