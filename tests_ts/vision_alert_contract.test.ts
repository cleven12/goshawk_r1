import { describe, expect, it } from "bun:test";

type ThreatLevel = "INFO" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

interface VisionAlertPayload {
  alertId: string;
  cameraId: string;
  timestamp: string;
  threatLevel: ThreatLevel;
  eventType: "forced_entry" | "armed_threat" | "physical_violence" | "perimeter_breach" | "suspicious_surveillance";
  description: string;
  evidenceTokens: string[];
  confidence: number;
}

function shouldDispatchEmergency(alert: VisionAlertPayload): boolean {
  return alert.threatLevel === "CRITICAL" || alert.threatLevel === "HIGH";
}

describe("Vision Alert & Emergency Dispatch Contracts", () => {
  it("enforces security alert contract structure", () => {
    const alert: VisionAlertPayload = {
      alertId: "ALT-CAM04-1700000000",
      cameraId: "CAM-04",
      timestamp: new Date().toISOString(),
      threatLevel: "CRITICAL",
      eventType: "forced_entry",
      description: "Subject with crowbar detected forcing lock.",
      evidenceTokens: ["crowbar", "forcing door"],
      confidence: 0.96,
    };

    expect(alert.alertId.startsWith("ALT-")).toBeTrue();
    expect(alert.evidenceTokens.length).toBeGreaterThan(0);
    expect(alert.confidence).toBeGreaterThanOrEqual(0.9);
    expect(shouldDispatchEmergency(alert)).toBeTrue();
  });

  it("filters low-priority ambient events from emergency channel dispatch", () => {
    const ambientAlert: VisionAlertPayload = {
      alertId: "ALT-CAM01-1700000001",
      cameraId: "CAM-01",
      timestamp: new Date().toISOString(),
      threatLevel: "MEDIUM",
      eventType: "suspicious_surveillance",
      description: "Subject lingering outside.",
      evidenceTokens: ["loitering"],
      confidence: 0.85,
    };

    expect(shouldDispatchEmergency(ambientAlert)).toBeFalse();
  });
});
