"""Statutory compliance and legal citation verification for East African criminal forensics."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class LegalStatute:
    act: str
    section: str
    title: str
    description: str
    elements_of_proof: List[str]


# Key Tanzanian criminal and forensic statutes
TANZANIA_STATUTES: Dict[str, LegalStatute] = {
    "PC_265": LegalStatute(
        act="Penal Code Cap 16 R.E. 2019",
        section="Section 265",
        title="General Punishment for Theft",
        description="Any person who steals anything capable of being stolen is guilty of a felony.",
        elements_of_proof=[
            "Taking of property without claim of right",
            "Intent to permanently deprive the owner",
            "Property capable of being stolen",
        ],
    ),
    "PC_271": LegalStatute(
        act="Penal Code Cap 16 R.E. 2019",
        section="Section 271",
        title="Stealing by Servants, Agents or Clerks",
        description="Embezzlement or conversion of property received by clerk or servant in trust.",
        elements_of_proof=[
            "Fiduciary or employment relationship existed",
            "Funds received on behalf of employer",
            "Fraudulent conversion or diversion",
        ],
    ),
    "PC_302": LegalStatute(
        act="Penal Code Cap 16 R.E. 2019",
        section="Section 302",
        title="Obtaining by False Pretences",
        description="Any person who by any false pretence obtains from any other person anything capable of being stolen.",
        elements_of_proof=[
            "False representation of past or present fact",
            "Inducement of victim to part with property",
            "Fraudulent intent",
        ],
    ),
    "TEA_40A": LegalStatute(
        act="Tanzania Evidence Act Cap 6 R.E. 2019",
        section="Section 40A",
        title="Admissibility of Electronic and Digital Records",
        description="Electronic statements, mobile money records, and computer outputs are admissible as primary evidence.",
        elements_of_proof=[
            "Computer or server operated properly during period",
            "Certificate of electronic evidence executed under section 40B",
            "Integrity of data stream preserved without tampering",
        ],
    ),
    "CYBER_14": LegalStatute(
        act="Cybercrimes Act No. 14 of 2015",
        section="Section 14",
        title="Computer Related Fraud",
        description="Causing unlawful financial loss or gain through input, alteration, or deletion of computer data.",
        elements_of_proof=[
            "Unauthorized electronic manipulation",
            "Resulting in financial gain or wrongful loss",
            "Intent to defraud",
        ],
    ),
}


class LegalStatuteVerifier:
    """Verifies that legal citations and penal code references are valid and complete."""

    @staticmethod
    def get_statute(code: str) -> Optional[LegalStatute]:
        return TANZANIA_STATUTES.get(code.upper())

    @staticmethod
    def verify_citation(act_substring: str, section_number: str) -> bool:
        """Return True if the citation matches official statutory records."""
        for statute in TANZANIA_STATUTES.values():
            if act_substring.lower() in statute.act.lower():
                if section_number.lower() in statute.section.lower():
                    return True
        return False
