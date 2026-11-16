"""Tests for automated forensic dossier generation."""

from goshawk.forensics.dossier import build_forensic_dossier


def test_build_forensic_dossier_flags_rapid_dispersal():
    statements = "\n".join(
        [
            "QJ94KD810Z Confirmed. You have received TZS 2,000,000 from SOURCE A +255754000001 on 16/11/2026. New M-Pesa balance is TZS 2,000,000.",
            "QJ94KD811A Confirmed. sent to TZS 950,000 to MULE ONE +255754000002 on 16/11/2026. New M-Pesa balance is TZS 1,050,000.",
            "QJ94KD811B Confirmed. sent to TZS 950,000 to MULE TWO +255754000003 on 16/11/2026. New M-Pesa balance is TZS 100,000.",
        ]
    )

    dossier = build_forensic_dossier(statements, primary_account="+255754999999")
    assert dossier.total_records == 3
    assert dossier.total_inflow == 2000000
    assert dossier.total_outflow == 1900000
    assert dossier.rapid_dispersal_flagged is True

    md = dossier.to_markdown()
    assert "RAPID DISPERSAL" in md
    assert "Section 40A" in md
    assert "Section 40B" in md
