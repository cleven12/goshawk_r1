"""Sliding-window CCTV stream watcher with cooldown alert deduplication."""

from collections import deque
from typing import Any, Deque, Dict, Optional

from .alerting import AlertDispatcher
from .event_detector import SecurityAlert, SecurityEventDetector


class CameraStreamWatcher:
    """
    Maintains a rolling temporal buffer of camera frame descriptions per camera_id
    and dispatches deduplicated emergency alerts when criminal anomalies emerge.
    """

    def __init__(
        self,
        window_size: int = 5,
        cooldown_seconds: int = 60,
        detector: Optional[SecurityEventDetector] = None,
        dispatcher: Optional[AlertDispatcher] = None,
    ):
        self.window_size = window_size
        self.cooldown_seconds = cooldown_seconds
        self.detector = detector or SecurityEventDetector()
        self.dispatcher = dispatcher or AlertDispatcher(cooldown_seconds=cooldown_seconds)
        self._buffers: Dict[str, Deque[str]] = {}
        self._last_alert_epoch: Dict[str, float] = {}

    def ingest_frame_event(
        self,
        camera_id: str,
        description: str,
        epoch_seconds: float,
        timestamp: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Add a frame description to the camera's rolling window and evaluate threat level.
        Suppresses duplicate dispatches within `cooldown_seconds` per (camera_id, event_type).
        """
        if camera_id not in self._buffers:
            self._buffers[camera_id] = deque(maxlen=self.window_size)

        self._buffers[camera_id].append(description)
        alert: Optional[SecurityAlert] = self.detector.analyze_event_sequence(
            camera_id=camera_id,
            events=list(self._buffers[camera_id]),
            timestamp=timestamp,
        )
        if alert is None:
            return None

        dedup_key = f"{camera_id}:{alert.event_type}"
        last_sent = self._last_alert_epoch.get(dedup_key)
        if last_sent is not None and (epoch_seconds - last_sent) < self.cooldown_seconds:
            return None

        self._last_alert_epoch[dedup_key] = epoch_seconds
        return self.dispatcher.dispatch(alert)
