"""East African mobile money and bank statement forensic ledger extractor."""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class TransactionRecord:
    reference: str
    timestamp: str
    channel: str  # MPESA, AIRTEL_MONEY, CRDB, NMB, HALOPESA
    direction: str  # DEBIT, CREDIT
    amount: int
    currency: str
    counterparty_name: str
    counterparty_identifier: str  # Phone or Bank Account
    balance: Optional[int] = None
    fee: int = 0
    raw_text: str = ""


class ForensicLedgerParser:
    """Parses raw statement text from East African banks and mobile money into structured audit records."""

    # M-Pesa SMS / Statement regex pattern
    MPESA_PATTERN = re.compile(
        r"([A-Z0-9]{10,12})\s+Confirmed\.\s+(?:You have received|sent to|paid to|withdrawn)\s+([A-Z]+)\s*([\d,]+)\s+(?:from|to)\s+([^0-9\.\+]+)\s*(\+?255\d{9}|0\d{9})?.*?balance is\s+([A-Z]+)\s*([\d,]+)",
        re.IGNORECASE,
    )

    # Simplified general statement line regex
    LINE_PATTERN = re.compile(
        r"([A-Z0-9]{8,15})\s+([\d\-\:\s]{10,19})\s+([A-Z\_\s]+?)\s+([0-9,]+)\s+([A-Z]{3})\s+([A-Z\s]+?)\s+(\d{9,16})",
        re.IGNORECASE,
    )

    def parse_statement_text(self, text: str, default_channel: str = "MPESA") -> List[TransactionRecord]:
        records: List[TransactionRecord] = []
        lines = text.strip().splitlines()

        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue

            # Check explicit M-Pesa format
            m = self.MPESA_PATTERN.search(line_clean)
            if m:
                ref = m.group(1).upper()
                curr = m.group(2).upper()
                amt = int(m.group(3).replace(",", ""))
                name = m.group(4).strip()
                phone = m.group(5) or ""
                direction = "CREDIT" if "received" in line_clean.lower() else "DEBIT"
                bal = int(m.group(7).replace(",", ""))

                records.append(
                    TransactionRecord(
                        reference=ref,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        channel=default_channel,
                        direction=direction,
                        amount=amt,
                        currency=curr,
                        counterparty_name=name,
                        counterparty_identifier=phone,
                        balance=bal,
                        raw_text=line_clean,
                    )
                )
                continue

            # Fallback line pattern
            m_line = self.LINE_PATTERN.search(line_clean)
            if m_line:
                ref = m_line.group(1).upper()
                ts = m_line.group(2)
                direction_str = m_line.group(3).upper()
                amt = int(m_line.group(4).replace(",", ""))
                curr = m_line.group(5).upper()
                name = m_line.group(6).strip()
                ident = m_line.group(7).strip()
                direction = "CREDIT" if "IN" in direction_str or "CR" in direction_str else "DEBIT"

                records.append(
                    TransactionRecord(
                        reference=ref,
                        timestamp=ts,
                        channel=default_channel,
                        direction=direction,
                        amount=amt,
                        currency=curr,
                        counterparty_name=name,
                        counterparty_identifier=ident,
                        raw_text=line_clean,
                    )
                )

        return records
