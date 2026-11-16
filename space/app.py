"""Gradio Hugging Face Space application for Goshawk R1."""

import json
import time

import gradio as gr
import torch

from goshawk.config import GoshawkConfig
from goshawk.forensics.dossier import build_forensic_dossier
from goshawk.model.generation import LatentPrefixCache, generate
from goshawk.model.transformer import GoshawkForCausalLM
from goshawk.tokenizer.bpe import GoshawkTokenizer
from goshawk.vision_security.alerting import AlertDispatcher
from goshawk.vision_security.event_detector import SecurityEventDetector

# Initialize lightweight Nano engine and prefix cache on Space startup
_CONFIG = GoshawkConfig.nano()
_TOKENIZER = GoshawkTokenizer.default()
_MODEL = GoshawkForCausalLM(_CONFIG).eval()
_PREFIX_CACHE = LatentPrefixCache(max_entries=16)

_DETECTOR = SecurityEventDetector()
_DISPATCHER = AlertDispatcher()


def run_cctv_watch(camera_id: str, events_text: str) -> str:
    events = [line.strip() for line in events_text.splitlines() if line.strip()]
    if not events:
        return "Provide at least one camera frame description."
    alert = _DETECTOR.analyze_event_sequence(camera_id=camera_id or "CAM-01", events=events)
    if alert is None:
        return json.dumps({"status": "NORMAL", "camera_id": camera_id, "threat_level": "NONE"}, indent=2)
    dispatch = _DISPATCHER.dispatch(alert)
    return json.dumps(
        {
            "alert_id": alert.alert_id,
            "threat_level": alert.threat_level,
            "event_type": alert.event_type,
            "evidence_tokens": alert.evidence_tokens,
            "confidence": alert.confidence,
            "dispatch": dispatch,
        },
        indent=2,
    )


def run_forensic_audit(statement_text: str, primary_account: str, threshold: float) -> str:
    if not statement_text.strip():
        return "Provide M-Pesa or bank statement lines to audit."
    dossier = build_forensic_dossier(
        statement_text=statement_text,
        primary_account=primary_account or "PRIMARY_ACCOUNT",
        dispersal_threshold=threshold,
    )
    return dossier.to_markdown()


def run_nano_reasoning(prompt: str, max_new_tokens: int, temperature: float) -> str:
    if not prompt.strip():
        return "Enter a prompt."
    prompt_ids = _TOKENIZER.encode(prompt, add_special_tokens=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = generate(
            _MODEL,
            prompt_ids,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            prefix_cache=_PREFIX_CACHE,
        )
    elapsed = time.perf_counter() - t0
    new_count = len(out_ids) - len(prompt_ids)
    decoded = _TOKENIZER.decode(out_ids[len(prompt_ids) :])
    tok_s = new_count / max(elapsed, 1e-5)
    return f"{decoded}\n\n---\nLatency: {elapsed * 1000:.1f} ms | {new_count} tokens ({tok_s:.1f} tok/s) | Cache hits: {_PREFIX_CACHE.stats['hits']}"


with gr.Blocks(title="Goshawk R1") as demo:
    gr.Markdown(
        "# Goshawk R1\n"
        "Edge Security Surveillance, East African Digital Forensics, and DeepSeek-style MLA + MoE Reasoning."
    )

    with gr.Tab("CCTV Security Watch"):
        cam_in = gr.Textbox(label="Camera ID", value="CAM-04")
        events_in = gr.Textbox(
            label="Frame Event Stream (one per line)",
            lines=4,
            value="02:41 AM Subject approached rear gate\n02:41 AM Individual carrying crowbar forcing door lock",
        )
        watch_btn = gr.Button("Analyze Stream")
        watch_out = gr.Code(label="Security Alert & Dispatch", language="json")
        watch_btn.click(run_cctv_watch, inputs=[cam_in, events_in], outputs=watch_out)

    with gr.Tab("Forensic Ledger Auditor"):
        acct_in = gr.Textbox(label="Primary Account / MSISDN", value="+255754999999")
        thresh_in = gr.Slider(label="Rapid Dispersal Threshold", minimum=0.5, maximum=1.0, value=0.85, step=0.05)
        stmt_in = gr.Textbox(
            label="M-Pesa / Bank Statements",
            lines=5,
            value=(
                "QJ94KD810Z Confirmed. You have received TZS 5,000,000 from SOURCE A +255754000001 on 16/11/2026. New M-Pesa balance is TZS 5,000,000.\n"
                "QJ94KD811A Confirmed. sent to TZS 4,800,000 to MULE ONE +255754000002 on 16/11/2026. New M-Pesa balance is TZS 200,000."
            ),
        )
        audit_btn = gr.Button("Generate Forensic Dossier")
        audit_out = gr.Markdown(label="Investigation Dossier")
        audit_btn.click(run_forensic_audit, inputs=[stmt_in, acct_in, thresh_in], outputs=audit_out)

    with gr.Tab("Goshawk-Nano MLA+MoE Engine"):
        prompt_in = gr.Textbox(
            label="Prompt",
            lines=3,
            value="<forensic> Analyze rapid dispersal under Tanzania Evidence Act Section 40A. </forensic>",
        )
        max_tok_in = gr.Slider(label="Max New Tokens", minimum=8, maximum=128, value=32, step=8)
        temp_in = gr.Slider(label="Temperature", minimum=0.0, maximum=1.2, value=0.7, step=0.1)
        gen_btn = gr.Button("Run Reasoning")
        gen_out = gr.Textbox(label="Model Output & Telemetry", lines=8)
        gen_btn.click(run_nano_reasoning, inputs=[prompt_in, max_tok_in, temp_in], outputs=gen_out)

if __name__ == "__main__":
    demo.launch()
