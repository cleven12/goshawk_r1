import { describe, expect, it } from "bun:test";
import { spawnSync } from "bun";

const PYTHON_BIN = "/home/cleven/Private/cleven-opensource/goshawk_r1/.venv/bin/python";
const WORKDIR = "/home/cleven/Private/cleven-opensource/goshawk_r1";

describe("Python Engine CLI Bridge", () => {
  it("runs goshawk watch and detects forced entry alert", () => {
    const proc = spawnSync({
      cmd: [
        PYTHON_BIN,
        "-m",
        "goshawk.cli",
        "watch",
        "--camera-id",
        "CAM-09",
        "--events",
        "Subject carrying crowbar near rear entrance",
      ],
      cwd: WORKDIR,
      stdout: "pipe",
      stderr: "pipe",
    });

    expect(proc.exitCode).toBe(0);
    const stdout = proc.stdout.toString();
    expect(stdout).toContain("ALERT DETECTED: [CRITICAL] forced_entry");
    expect(stdout).toContain("dispatched");
  });

  it("runs goshawk audit on sample M-Pesa text", () => {
    const sample =
      "QJ94KD810Z Confirmed. You have received TZS 1,500,000 from JOHN KWEKA +255754123456 on 14/11/2026 at 10:14 PM. New M-Pesa balance is TZS 2,850,000.";

    const proc = spawnSync({
      cmd: [PYTHON_BIN, "-m", "goshawk.cli", "audit", "--text", sample],
      cwd: WORKDIR,
      stdout: "pipe",
      stderr: "pipe",
    });

    expect(proc.exitCode).toBe(0);
    const stdout = proc.stdout.toString();
    expect(stdout).toContain("Parsed 1 transaction records.");
    expect(stdout).toContain("FORENSIC SUMMARY");
  });
});
