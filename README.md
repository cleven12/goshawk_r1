# Goshawk R1

Goshawk R1 is an ultra-fast, edge-capable reasoning language model architecture built from scratch following DeepSeek's modern breakthroughs (Multi-Head Latent Attention, DeepSeekMoE, and Group Relative Policy Optimization).

Designed for high-throughput, offline execution on low-cost hardware in Africa, Goshawk R1 specializes in:
1. **Edge Computer Vision & CCTV Event Surveillance**: Translating continuous physical security and camera event streams into immediate structured alerts (forced entry, weapon display, perimeter intrusion, physical altercation).
2. **Legal & Digital Forensics Reasoning**: Automated ingestion of East African bank ledgers, M-Pesa / Airtel Money transaction statements, and communication logs to reconstruct verifiable money trails and statutory violations under East African penal codes.

---

## Architectural Highlights

```text
Input Tokens -> RMSNorm -> Multi-Head Latent Attention (MLA) -> RMSNorm -> DeepSeekMoE -> Output Logits
                                    |                                           |
                    Low-Rank Latent KV Compression              Shared + Fine-Grained Routed Experts
                    (80%+ KV Cache Memory Reduction)            (High Capacity with Low Compute/Token)
```

### 1. Multi-Head Latent Attention (MLA)
Standard multi-head or grouped-query attention requires storing full Key and Value tensors for every attention head at every step, quickly exhausting RAM on edge devices.

Goshawk R1 projects Keys and Values into a single compressed latent vector:

$$c_t^{KV} = W^{DKV} h_t$$

Only $c_t^{KV}$ is retained in the inference KV cache. At attention time, $c_t^{KV}$ is projected back into decoupled key-value representations with Rotary Position Embeddings (RoPE), achieving up to an 80% reduction in KV cache memory footprint.

### 2. DeepSeekMoE (Fine-Grained + Shared Experts)
Goshawk R1 isolates domain knowledge into:
- **Shared Experts**: Fixed experts that are always activated for universal syntactic, logical, and grammatical coherence.
- **Fine-Grained Routed Experts**: Specialized sub-networks dynamically selected via top-k gating with auxiliary-loss-free bias routing, specializing in:
  - Security event anomaly detection
  - Financial ledger balance reconciliation
  - Legal statute cross-referencing (Tanzania Evidence Act, Penal Code, Cybercrimes Act)

### 3. DeepSeek-R1 GRPO (Group Relative Policy Optimization)
Traditional Reinforcement Learning from Human Feedback (PPO) requires training a separate, memory-intensive Critic (Value) model. 

Goshawk R1 implements **Group Relative Policy Optimization (GRPO)**:
- Generates a group of $G$ outputs per prompt.
- Evaluates outputs using deterministic, rule-based reward verifiers (e.g. mathematical balance reconciliation, correct citation of legal statutes, and `<think>...</think>` chain-of-thought formatting).
- Computes relative advantage:

$$A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$$

- Optimizes policy weights directly without a value model, slashing training memory overhead by 50%.

---

## Model Profiles

| Profile | Total Parameters | Active Parameters | Target Hardware | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Goshawk-Nano** | 180M | 45M | Raspberry Pi 5, NPU, CCTV DVR | Real-time security camera event parsing & alerting |
| **Goshawk-Micro** | 850M | 210M | Standard laptops (Intel i5/AMD Ryzen, 4GB RAM) | Mobile money audit trail & forensic statement analysis |
| **Goshawk-Edge** | 2.8B | 650M | Workstations & local law firm servers (8GB RAM) | Full legal brief synthesis & multi-account fraud tracing |

---

## Quickstart

### Python Environment

```bash
git clone https://github.com/cleven12/goshawk_r1.git
cd goshawk_r1
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run model architecture tests
pytest
```

### Bun + TypeScript Suite

```bash
# Run fast contract and tokenizer tests in under 1 second
bun test
```

### Example: Multi-Step Forensic Reasoning

```python
from goshawk.config import GoshawkConfig
from goshawk.model.transformer import GoshawkForCausalLM
from goshawk.tokenizer.bpe import GoshawkTokenizer

config = GoshawkConfig.micro()
model = GoshawkForCausalLM(config)
tokenizer = GoshawkTokenizer.default()

prompt = (
    "<forensic>\n"
    "Statement 1: A sent 5,000,000 TZS to B via M-Pesa (Ref: QX1029).\n"
    "Statement 2: B deposited 4,950,000 TZS to CRDB Bank Account 0150... within 12 minutes.\n"
    "Identify discrepancy and applicable section under Tanzania Penal Code Cap 16.\n"
    "</forensic>\n"
    "<think>\n"
)

tokens = tokenizer.encode(prompt)
output_tokens = model.generate(tokens, max_new_tokens=256)
print(tokenizer.decode(output_tokens))
```

---

## License

Apache-2.0 License. Copyright (c) Cleven.
