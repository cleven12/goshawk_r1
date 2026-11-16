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

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
