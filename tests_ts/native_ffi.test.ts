import { afterAll, describe, expect, it } from "bun:test";
import { spawnSync } from "bun";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { NativeGoshawkKernel } from "../native/runner";

const kernel = new NativeGoshawkKernel();

afterAll(() => {
  kernel.close();
});

describe("Native C99 + Bun FFI Inference Kernels", () => {
  it("computes RMSNorm accurately", () => {
    const x = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const weight = new Float32Array([1.0, 1.0, 1.0, 1.0]);
    const out = kernel.rmsnorm(x, weight, 1e-6);

    // mean(x^2) = (1 + 4 + 9 + 16)/4 = 7.5; rms = sqrt(7.5) ~= 2.7386128
    const invRms = 1.0 / Math.sqrt(7.5 + 1e-6);
    for (let i = 0; i < x.length; i++) {
      expect(out[i]).toBeCloseTo(x[i]! * invRms, 4);
    }
  });

  it("computes FP32 GEMV accurately", () => {
    // 2x3 matrix
    const W = new Float32Array([1, 2, 3, 4, 5, 6]);
    const x = new Float32Array([10, 20, 30]);
    const out = kernel.gemvF32(W, x, 2, 3);

    expect(out[0]).toBeCloseTo(140, 4);
    expect(out[1]).toBeCloseTo(320, 4);
  });

  it("computes fused packed INT4 GEMV accurately including odd dimensions", () => {
    // 2x3 matrix packed into 2x2 bytes:
    // Row 0: weights [2, -3, 1] -> +8 => [10, 5, 9] -> bytes [10 | (5 << 4), 9 | (8 << 4)] = [0x5A, 0x89]
    // Row 1: weights [-1, 4, -2] -> +8 => [7, 12, 6] -> bytes [7 | (12 << 4), 6 | (8 << 4)] = [0xC7, 0x86]
    const WPacked = new Uint8Array([0x5a, 0x89, 0xc7, 0x86]);
    const scales = new Float32Array([0.5, 2.0]);
    const x = new Float32Array([2.0, 4.0, 6.0]);

    const out = kernel.gemvQ4(WPacked, scales, x, 2, 3);
    // Row 0: (2*2 + (-3)*4 + 1*6) * 0.5 = (4 - 12 + 6) * 0.5 = -1.0
    // Row 1: ((-1)*2 + 4*4 + (-2)*6) * 2.0 = (-2 + 16 - 12) * 2.0 = 4.0
    expect(out[0]).toBeCloseTo(-1.0, 4);
    expect(out[1]).toBeCloseTo(4.0, 4);
  });

  it("computes SwiGLU activation accurately", () => {
    const gate = new Float32Array([0.0, 2.0]);
    const up = new Float32Array([5.0, 3.0]);
    const out = kernel.swiglu(gate, up);

    // silu(0) = 0; silu(2) = 2 / (1 + exp(-2)) ~= 1.761594
    expect(out[0]).toBeCloseTo(0.0, 4);
    expect(out[1]).toBeCloseTo((2.0 / (1.0 + Math.exp(-2.0))) * 3.0, 4);
  });

  it("inspects exported .gshk binary file directly in C without Python", () => {
    const tmp = mkdtempSync(join(tmpdir(), "gshk-ffi-"));
    const gshkPath = join(tmp, "mini.gshk");

    const script = `
from goshawk.config import GoshawkConfig
from goshawk.export.gguf import GoshawkBinaryExporter
from goshawk.model.transformer import GoshawkForCausalLM

cfg = GoshawkConfig(
    vocab_size=64, d_model=32, n_layers=1, n_heads=2, v_head_dim=16,
    max_seq_len=32, q_lora_rank=16, kv_lora_rank=8, qk_rope_head_dim=8,
    n_routed_experts=2, n_shared_experts=1, num_experts_per_tok=1,
    moe_intermediate_dim=32, dense_intermediate_dim=32, mtp_depth=0,
)
m = GoshawkForCausalLM(cfg)
GoshawkBinaryExporter.export(m, cfg, ${JSON.stringify(gshkPath)})
`;
    const pyProc = spawnSync({
      cmd: ["/home/cleven/Private/cleven-opensource/goshawk_r1/.venv/bin/python", "-c", script],
      cwd: "/home/cleven/Private/cleven-opensource/goshawk_r1",
      stdout: "pipe",
      stderr: "pipe",
    });
    expect(pyProc.exitCode).toBe(0);

    const info = kernel.inspectGshk(gshkPath);
    expect(info.version).toBe(1);
    expect(info.numTensors).toBeGreaterThan(10);

    rmSync(tmp, { recursive: true, force: true });
  });
});
