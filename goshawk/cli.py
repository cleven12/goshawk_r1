"""Command-line interface for Goshawk R1 edge surveillance and forensic intelligence."""

import argparse
import sys
from typing import List

from .forensics.graph_trace import TransactionFlowGraph
from .forensics.ledger_parser import ForensicLedgerParser
from .forensics.statute_verifier import LegalStatuteVerifier
from .vision_security.alerting import AlertDispatcher
from .vision_security.event_detector import SecurityEventDetector


def cmd_audit(args: argparse.Namespace) -> None:
    """Audit mobile money and bank statements for rapid dispersal or smurfing."""
    parser = ForensicLedgerParser()
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
    elif args.text:
        content = args.text
    else:
        print("Error: Provide --file or --text for audit.")
        sys.exit(1)

    records = parser.parse_statement_text(content, default_channel=args.channel)
    print(f"Parsed {len(records)} transaction records.")

    graph = TransactionFlowGraph()
    primary = args.primary or "PRIMARY_ACCOUNT"
    graph.build_from_records(records, primary_account=primary)

    summary = graph.get_summary()
    print("\n--- FORENSIC SUMMARY ---")
    print(f"Total Participants: {summary['total_parties']}")
    print(f"Total Transfers: {summary['total_transfers']}")

    is_dispersal = graph.detect_rapid_dispersal(primary, threshold=args.threshold)
    if is_dispersal:
        print("\nWARNING: High probability of rapid dispersal or smurfing detected!")
        statute = LegalStatuteVerifier.get_statute("PC_302")
        if statute:
            print(f"Relevant Statute: {statute.act} {statute.section} - {statute.title}")
    else:
        print("\nFlow balance within normal operating threshold.")


def cmd_watch(args: argparse.Namespace) -> None:
    """Watch camera stream annotations and detect immediate security threats."""
    detector = SecurityEventDetector()
    dispatcher = AlertDispatcher()

    events: List[str] = args.events
    print(f"Analyzing {len(events)} vision event descriptions for Camera {args.camera_id}...")

    alert = detector.analyze_event_sequence(
        camera_id=args.camera_id,
        events=events,
    )
    if alert:
        print(f"\nALERT DETECTED: [{alert.threat_level}] {alert.event_type}")
        print(f"Description: {alert.description}")
        print(f"Evidence: {', '.join(alert.evidence_tokens)}")
        dispatch_res = dispatcher.dispatch(alert)
        print(f"Notification status: {dispatch_res['status']} via {dispatch_res['channels_notified']}")
    else:
        print("Surveillance stream normal: No criminal or security anomalies detected.")


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive reasoning session demonstrating <think>...</think> DeepSeek-R1 style generation."""
    from .config import GoshawkConfig
    from .model.generation import generate
    from .model.transformer import GoshawkForCausalLM
    from .tokenizer.bpe import GoshawkTokenizer

    print(f"Loading Goshawk R1 (Profile: {args.profile})...")
    config = getattr(GoshawkConfig, args.profile, GoshawkConfig.nano)()
    tokenizer = GoshawkTokenizer.default()
    model = GoshawkForCausalLM(config)
    model.eval()

    print("Goshawk R1 Ready. Type 'exit' to quit.")
    while True:
        try:
            prompt = input("\nUser > ").strip()
            if not prompt or prompt.lower() in ("exit", "quit"):
                break

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            output_ids = generate(
                model,
                prompt_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            response = tokenizer.decode(output_ids[len(prompt_ids) :])
            print(f"Goshawk R1 > {response}")
        except (KeyboardInterrupt, EOFError):
            break


def cmd_train(args: argparse.Namespace) -> None:
    """Run SFT or GRPO training on Goshawk R1 reasoning trajectories."""
    from .config import GoshawkConfig
    from .model.transformer import GoshawkForCausalLM
    from .tokenizer.bpe import GoshawkTokenizer
    from .train.dataset import GoshawkReasoningDataset
    from .train.pretrain import GoshawkTrainer, PretrainConfig

    config = getattr(GoshawkConfig, args.profile, GoshawkConfig.nano)()
    tokenizer = GoshawkTokenizer.default()
    model = GoshawkForCausalLM(config)
    dataset = GoshawkReasoningDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len)

    trainer = GoshawkTrainer(
        model=model,
        train_dataset=dataset,
        config=PretrainConfig(batch_size=args.batch_size, max_epochs=args.epochs),
    )
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(epoch)
        print(f"Epoch {epoch}: train_loss={metrics['train_loss']:.4f}, ppl={metrics['perplexity']:.2f}")

    if args.output:
        trainer.save_checkpoint(args.output)
        print(f"Saved checkpoint to {args.output}")


def cmd_export(args: argparse.Namespace) -> None:
    """Quantize and export Goshawk R1 to .gshk binary or Hugging Face bundle."""
    from .config import GoshawkConfig
    from .export.gguf import GoshawkBinaryExporter
    from .export.hf_hub import export_hf_bundle, upload_hf_bundle
    from .export.quantize import quantize_model
    from .model.transformer import GoshawkForCausalLM

    config = getattr(GoshawkConfig, args.profile, GoshawkConfig.nano)()
    model = GoshawkForCausalLM(config)

    if args.bits in (4, 8):
        model, stats = quantize_model(model, bits=args.bits)
        print(f"Quantized ({args.bits}-bit): compression_ratio={stats['compression_ratio']:.2f}x")

    if args.hf_dir:
        res = export_hf_bundle(model, config, args.hf_dir, model_name=args.hf_repo)
        print(f"Exported Hugging Face bundle to {res['output_dir']} ({len(res['files'])} files)")
        if getattr(args, "hf_push", False):
            up = upload_hf_bundle(
                bundle_dir=args.hf_dir,
                repo_id=args.hf_repo if args.hf_repo else None,
                env_path=getattr(args, "env_file", ".env"),
            )
            print(f"Uploaded to Hugging Face Hub: {up['url']}")
    else:
        res = GoshawkBinaryExporter.export(model, config, args.output)
        print(f"Exported {res['tensors_exported']} tensors to {res['output_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="goshawk",
        description="Goshawk R1 - Edge Security & Legal Forensics Reasoning LLM",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Audit subcommand
    audit_parser = subparsers.add_parser("audit", help="Audit financial ledgers")
    audit_parser.add_argument("--file", type=str, help="Path to statement file")
    audit_parser.add_argument("--text", type=str, help="Raw statement text string")
    audit_parser.add_argument("--channel", type=str, default="MPESA", help="Payment channel")
    audit_parser.add_argument("--primary", type=str, default="PRIMARY_ACCOUNT", help="Primary account ID")
    audit_parser.add_argument("--threshold", type=float, default=0.85, help="Dispersal ratio threshold")
    audit_parser.set_defaults(func=cmd_audit)

    # Watch subcommand
    watch_parser = subparsers.add_parser("watch", help="Watch CCTV annotations for security threats")
    watch_parser.add_argument("--camera-id", type=str, default="CAM-01", help="Camera identifier")
    watch_parser.add_argument("--events", nargs="+", required=True, help="List of visual event tokens or descriptions")
    watch_parser.set_defaults(func=cmd_watch)

    # Chat subcommand
    chat_parser = subparsers.add_parser("chat", help="Run interactive reasoning session")
    chat_parser.add_argument("--profile", type=str, default="nano", choices=["nano", "micro", "edge"])
    chat_parser.add_argument("--max-tokens", type=int, default=128)
    chat_parser.add_argument("--temperature", type=float, default=0.7)
    chat_parser.set_defaults(func=cmd_chat)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train Goshawk R1 on reasoning trajectories")
    train_parser.add_argument("--profile", type=str, default="nano", choices=["nano", "micro", "edge"])
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--max-seq-len", type=int, default=256)
    train_parser.add_argument("--output", type=str, default="", help="Optional checkpoint output path")
    train_parser.set_defaults(func=cmd_train)

    # Export subcommand
    export_parser = subparsers.add_parser("export", help="Quantize and export Goshawk R1 weights")
    export_parser.add_argument("--profile", type=str, default="nano", choices=["nano", "micro", "edge"])
    export_parser.add_argument("--bits", type=int, default=4, choices=[0, 4, 8], help="0 for FP32, 4 or 8 for quantized")
    export_parser.add_argument("--output", type=str, default="goshawk.gshk", help="Output .gshk binary path")
    export_parser.add_argument("--hf-dir", type=str, default="", help="Export full Hugging Face bundle directory")
    export_parser.add_argument("--hf-repo", type=str, default="", help="Hugging Face repo ID (overrides HF_REPO_ID in .env)")
    export_parser.add_argument("--hf-push", action="store_true", help="Upload exported Hugging Face bundle using .env HF_TOKEN")
    export_parser.add_argument("--env-file", type=str, default=".env", help="Path to .env file with HF_TOKEN")
    export_parser.set_defaults(func=cmd_export)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
