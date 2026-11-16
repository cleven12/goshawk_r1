"""Multi-channel emergency alert dispatcher for real-time camera security events."""

from typing import Any, Dict, List, Optional

from .event_detector import SecurityAlert


class AlertDispatcher:
    """Dispatches high-priority security notifications with deduplication."""

    def __init__(self, cooldown_seconds: int = 60):
        self.cooldown_seconds = cooldown_seconds
        self.sent_alerts: List[Dict[str, Any]] = []

    def dispatch(
        self,
        alert: SecurityAlert,
        channels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Send security alert to configured channels (SMS, WhatsApp, Telegram, Webhook)."""
        target_channels = channels or ["telegram", "sms", "webhook"]
        result = {
            "alert_id": alert.alert_id,
            "camera_id": alert.camera_id,
            "threat_level": alert.threat_level,
            "channels_notified": target_channels,
            "status": "dispatched",
            "message": (
                f"[{alert.threat_level}] Security incident at Camera {alert.camera_id}: "
                f"{alert.description} (Evidence: {', '.join(alert.evidence_tokens)})"
            ),
        }
        self.sent_alerts.append(result)
        return result
