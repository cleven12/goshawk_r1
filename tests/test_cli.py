"""Tests for Goshawk R1 CLI commands."""

import argparse
import os
import tempfile

from goshawk.cli import cmd_audit, cmd_export, cmd_watch


def test_cli_watch_and_audit(capsys):
    watch_args = argparse.Namespace(
        camera_id="CAM-07",
        events=["Person carrying knife near vault"],
    )
    cmd_watch(watch_args)
    captured = capsys.readouterr()
    assert "ALERT DETECTED: [CRITICAL] armed_threat" in captured.out

    audit_args = argparse.Namespace(
        file=None,
        text="QJ94KD810Z Confirmed. You have received TZS 1,500,000 from JOHN KWEKA +255754123456 on 14/11/2026. New M-Pesa balance is TZS 2,850,000.",
        channel="MPESA",
        primary="PRIMARY",
        threshold=0.85,
    )
    cmd_audit(audit_args)
    captured_audit = capsys.readouterr()
    assert "Parsed 1 transaction records." in captured_audit.out


def test_cli_export_gshk(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "nano.gshk")
        export_args = argparse.Namespace(
            profile="nano",
            bits=4,
            output=out_path,
            hf_dir="",
            hf_repo="cleven12/goshawk-r1-nano",
        )
        cmd_export(export_args)
        captured = capsys.readouterr()
        assert "Quantized (4-bit):" in captured.out
        assert os.path.exists(out_path)
