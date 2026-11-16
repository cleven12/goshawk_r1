import { describe, expect, it } from "bun:test";

interface TransactionContract {
  reference: string;
  timestamp: string;
  channel: "MPESA" | "AIRTEL_MONEY" | "HALOPESA" | "CRDB" | "NMB";
  direction: "DEBIT" | "CREDIT";
  amount: number;
  currency: "TZS" | "USD";
  counterpartyName: string;
  counterpartyIdentifier: string;
  balance?: number;
}

interface DispersalAnalysis {
  totalInflow: number;
  totalOutflow: number;
  ratio: number;
  isRapidDispersal: boolean;
}

function analyzeDispersal(inflow: number, outflow: number, threshold = 0.85): DispersalAnalysis {
  if (inflow <= 0 || outflow <= 0) {
    return {
      totalInflow: inflow,
      totalOutflow: outflow,
      ratio: 0,
      isRapidDispersal: false,
    };
  }
  const ratio = Math.min(inflow, outflow) / Math.max(inflow, outflow);
  return {
    totalInflow: inflow,
    totalOutflow: outflow,
    ratio: Math.round(ratio * 1000) / 1000,
    isRapidDispersal: ratio >= threshold,
  };
}

describe("Forensic Transaction Contracts & Dispersal Logic", () => {
  it("validates transaction schema for M-Pesa statements", () => {
    const tx: TransactionContract = {
      reference: "QJ94KD810Z",
      timestamp: "2026-11-16T10:14:00Z",
      channel: "MPESA",
      direction: "CREDIT",
      amount: 1500000,
      currency: "TZS",
      counterpartyName: "JOHN KWEKA",
      counterpartyIdentifier: "+255754123456",
      balance: 2850000,
    };

    expect(tx.reference).toMatch(/^[A-Z0-9]{8,15}$/);
    expect(tx.amount).toBeGreaterThan(0);
    expect(tx.currency).toBe("TZS");
    expect(tx.counterpartyIdentifier).toMatch(/^\+?255\d{9}$/);
  });

  it("identifies high-probability rapid dispersal / smurfing", () => {
    const inflow = 5000000;
    const outflow = 4900000;
    const analysis = analyzeDispersal(inflow, outflow, 0.85);

    expect(analysis.isRapidDispersal).toBeTrue();
    expect(analysis.ratio).toBe(0.98);
  });

  it("handles normal legitimate transaction disparity", () => {
    const inflow = 10000000;
    const outflow = 1200000;
    const analysis = analyzeDispersal(inflow, outflow, 0.85);

    expect(analysis.isRapidDispersal).toBeFalse();
    expect(analysis.ratio).toBe(0.12);
  });
});
