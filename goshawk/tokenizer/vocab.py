"""Special tokens and vocabulary mappings for Goshawk R1."""

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "<think>",
    "</think>",
    "<forensic>",
    "</forensic>",
    "<security_event>",
    "<alert>",
    "<evidence>",
    "<tanzania_law>",
    "<ledger>",
    "<ussd_flow>",
    "<cdr_log>",
]

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
THINK_START_ID = 4
THINK_END_ID = 5
FORENSIC_START_ID = 6
FORENSIC_END_ID = 7
SECURITY_EVENT_ID = 8
ALERT_ID = 9
EVIDENCE_ID = 10
TANZANIA_LAW_ID = 11
LEDGER_ID = 12
USSD_FLOW_ID = 13
CDR_LOG_ID = 14
