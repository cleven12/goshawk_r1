"""Tests for CCTV temporal event detection, anomaly classification, and alert dispatching."""


from goshawk.vision_security.alerting import AlertDispatcher
from goshawk.vision_security.event_detector import (
    SecurityAlert,
    SecurityEventDetector,
    ThreatLevel,
)


def test_event_detector_forced_entry():
    detector = SecurityEventDetector()
    events = [
        "Person approached rear door at 03:15 AM",
        "Individual seen with crowbar attempting to pry open lock",
    ]

    alert = detector.analyze_event_sequence(camera_id="CAM-02", events=events)
    assert alert is not None
    assert alert.threat_level == ThreatLevel.CRITICAL
    assert alert.event_type == "forced_entry"
    assert "crowbar" in alert.evidence_tokens


def test_event_detector_armed_threat():
    detector = SecurityEventDetector()
    events = [
        "Two subjects entering lobby area",
        "Subject displayed pistol near service counter",
    ]

    alert = detector.analyze_event_sequence(camera_id="CAM-01", events=events)
    assert alert is not None
    assert alert.threat_level == ThreatLevel.CRITICAL
    assert alert.event_type == "armed_threat"
    assert "pistol" in alert.evidence_tokens


def test_event_detector_normal_activity():
    detector = SecurityEventDetector()
    events = [
        "Customer walked into reception area",
        "Sitting on waiting chair holding documents",
    ]

    alert = detector.analyze_event_sequence(camera_id="CAM-05", events=events)
    assert alert is None


def test_alert_dispatcher():
    dispatcher = AlertDispatcher()
    alert = SecurityAlert(
        alert_id="ALT-CAM01-100",
        camera_id="CAM-01",
        timestamp="2026-11-16T12:00:00Z",
        threat_level=ThreatLevel.CRITICAL,
        event_type="forced_entry",
        description="Perimeter door forced open.",
        evidence_tokens=["crowbar"],
    )

    result = dispatcher.dispatch(alert, channels=["sms", "telegram"])
    assert result["status"] == "dispatched"
    assert "telegram" in result["channels_notified"]
    assert len(dispatcher.sent_alerts) == 1
