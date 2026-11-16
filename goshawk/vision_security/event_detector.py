"""Symbolic CCTV stream event parser and real-time security anomaly detector."""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


class ThreatLevel:
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class SecurityAlert:
    alert_id: str
    camera_id: str
    timestamp: str
    threat_level: str
    event_type: str
    description: str
    evidence_tokens: List[str] = field(default_factory=list)
    confidence: float = 0.95
    recommended_action: str = "Dispatch security personnel"


# Threat patterns based on surveillance frame annotations
SUSPICIOUS_PATTERNS = [
    (
        r"(crowbar|pry\s*bar|lockpick|break-in|forcing\s*door|broken\s*glass)",
        ThreatLevel.CRITICAL,
        "forced_entry",
        "Immediate perimeter intrusion with burglary tool detected.",
    ),
    (
        r"(gun|pistol|firearm|rifle|knife|machete|panga|weapon)",
        ThreatLevel.CRITICAL,
        "armed_threat",
        "Lethal weapon or edged instrument visible in frame.",
    ),
    (
        r"(altercation|fighting|assault|punching|struggling|shoving)",
        ThreatLevel.HIGH,
        "physical_violence",
        "Violent confrontation or physical altercation observed.",
    ),
    (
        r"(scaling\s*wall|jumping\s*fence|over\s*fence|perimeter\s*climb)",
        ThreatLevel.HIGH,
        "perimeter_breach",
        "Unauthorized scaling of perimeter fence or security wall.",
    ),
    (
        r"(loitering|lingering|concealing\s*face|mask\s*at\s*night|casing)",
        ThreatLevel.MEDIUM,
        "suspicious_surveillance",
        "Subject loitering with obscured face outside operating hours.",
    ),
]


class SecurityEventDetector:
    """Parses sequential CCTV/vision metadata and classifies immediate security threats."""

    def __init__(self, high_alert_after_hours: bool = True):
        self.high_alert_after_hours = high_alert_after_hours

    def analyze_event_sequence(
        self,
        camera_id: str,
        events: List[str],
        timestamp: Optional[str] = None,
    ) -> Optional[SecurityAlert]:
        """
        Evaluate temporal vision descriptions from camera stream.
        Returns a SecurityAlert if an anomalous or criminal threat is detected.
        """
        combined_text = " ".join(events).lower()
        now_ts = timestamp or datetime.now(timezone.utc).isoformat()

        highest_alert: Optional[SecurityAlert] = None

        for pattern, threat, ev_type, desc in SUSPICIOUS_PATTERNS:
            match = re.search(pattern, combined_text)
            if match:
                matched_token = match.group(1)
                alert = SecurityAlert(
                    alert_id=f"ALT-{camera_id}-{int(datetime.now(timezone.utc).timestamp())}",
                    camera_id=camera_id,
                    timestamp=now_ts,
                    threat_level=threat,
                    event_type=ev_type,
                    description=desc,
                    evidence_tokens=[matched_token],
                    confidence=0.96 if threat == ThreatLevel.CRITICAL else 0.88,
                )
                if highest_alert is None or threat == ThreatLevel.CRITICAL:
                    highest_alert = alert
                if threat == ThreatLevel.CRITICAL:
                    break

        return highest_alert

    def format_llm_event_prompt(self, alert: SecurityAlert) -> str:
        """Format a structured security event prompt for Goshawk R1 reasoning."""
        return (
            "<security_event>\n"
            f"Camera: {alert.camera_id}\n"
            f"Timestamp: {alert.timestamp}\n"
            f"Threat Level: {alert.threat_level}\n"
            f"Event Type: {alert.event_type}\n"
            f"Evidence: {', '.join(alert.evidence_tokens)}\n"
            "</security_event>\n"
            "<think>\n"
        )
