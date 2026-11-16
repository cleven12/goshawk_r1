import { describe, expect, it } from "bun:test";
import { spawnSync } from "bun";

const PYTHON_BIN = "/home/cleven/Private/cleven-opensource/goshawk_r1/.venv/bin/python";
const WORKDIR = "/home/cleven/Private/cleven-opensource/goshawk_r1";

describe("Goshawk R1 Edge Latency Contracts", () => {
  it("completes nano forward pass within edge latency budget", () => {
    const script = `
import time, torch
from goshawk.config import GoshawkConfig
from goshawk.model.transformer import GoshawkForCausalLM

cfg = GoshawkConfig(
    vocab_size=256, d_model=64, n_layers=2, n_heads=4, v_head_dim=16,
    max_seq_len=64, q_lora_rank=16, kv_lora_rank=8, qk_rope_head_dim=8,
    n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
    moe_intermediate_dim=64, dense_intermediate_dim=64,
)
model = GoshawkForCausalLM(cfg).eval()
x = torch.randint(0, 256, (1, 16))
with torch.no_grad():
    _ = model(x)
    t0 = time.perf_counter()
    for _ in range(5):
        _ = model(x)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / 5.0
print(f"{elapsed_ms:.2f}")
`;
    const proc = spawnSync({
      cmd: [PYTHON_BIN, "-c", script],
      cwd: WORKDIR,
      stdout: "pipe",
      stderr: "pipe",
    });

    expect(proc.exitCode).toBe(0);
    const ms = parseFloat(proc.stdout.toString().trim());
    expect(ms).toBeGreaterThan(0);
    expect(ms).toBeLessThan(2000);
  });
});
