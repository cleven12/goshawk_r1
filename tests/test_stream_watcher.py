"""Tests for sliding-window CCTV CameraStreamWatcher and cooldown deduplication."""

from goshawk.vision_security.stream_watcher import CameraStreamWatcher


def test_stream_watcher_cooldown_deduplication():
    watcher = CameraStreamWatcher(window_size=4, cooldown_seconds=60)

    # Normal frame -> no alert
    res0 = watcher.ingest_frame_event("CAM-01", "Empty hallway at 01:00 AM", epoch_seconds=1000.0)
    assert res0 is None

    # Forced entry detected -> dispatches alert
    res1 = watcher.ingest_frame_event("CAM-01", "Person holding crowbar at door", epoch_seconds=1010.0)
    assert res1 is not None
    assert res1["threat_level"] == "CRITICAL"

    # Same camera within cooldown (1030 - 1010 = 20 < 60) -> suppressed
    res2 = watcher.ingest_frame_event("CAM-01", "Person still forcing lock", epoch_seconds=1030.0)
    assert res2 is None

    # After cooldown expires (1080 - 1010 = 70 >= 60) -> dispatches again
    res3 = watcher.ingest_frame_event("CAM-01", "Crowbar still visible", epoch_seconds=1080.0)
    assert res3 is not None
    assert len(watcher.dispatcher.sent_alerts) == 2
