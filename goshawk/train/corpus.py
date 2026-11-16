"""Curated East African forensic, legal, and CCTV security reasoning corpus for Goshawk R1."""

from typing import Dict, List

TRAINING_CORPUS: List[Dict[str, str]] = [
    {
        "prompt": "<forensic> M-Pesa inflow TZS 5,000,000, five debits of TZS 980,000 in 12 min. </forensic>",
        "thinking": "<think>Inflow 5,000,000. Outflow 5 * 980,000 = 4,900,000. Ratio 0.98 >= 0.85. Rapid dispersal and smurfing under Cybercrimes Act s.14 and Penal Code s.302.</think>",
        "response": "<alert> [CRITICAL] Rapid dispersal detected (98%). Certify records under Evidence Act Section 40A/40B.",
        "task_type": "forensics",
    },
    {
        "prompt": "<forensic> M-Pesa inflow TZS 2,000,000, two transfers of TZS 950,000 to mule accounts. </forensic>",
        "thinking": "<think>Inflow 2,000,000. Outflow 1,900,000. Dispersal ratio 0.95 >= 0.85. Structuring pattern under Penal Code Cap 16 s.302.</think>",
        "response": "<alert> [CRITICAL] Structuring flagged (95% outflow). Preserve SHA-256 hash and Section 40A certificate.",
        "task_type": "forensics",
    },
    {
        "prompt": "<forensic> Airtel Money inflow TZS 3,000,000, rent payment TZS 400,000. </forensic>",
        "thinking": "<think>Inflow 3,000,000. Outflow 400,000. Ratio 0.133 < 0.85. Normal household expenditure.</think>",
        "response": "NORMAL: Flow ratio 13.3% within normal operating threshold.",
        "task_type": "forensics",
    },
    {
        "prompt": "<forensic> CRDB bank deposit TZS 10,000,000, immediate wire TZS 9,500,000 to 3 accounts. </forensic>",
        "thinking": "<think>Inflow 10,000,000. Outflow 9,500,000. Ratio 0.95 >= 0.85. High-velocity layering under Penal Code s.271.</think>",
        "response": "<alert> [CRITICAL] Layering and rapid dispersal (95%). Freeze accounts and issue Section 40A dossier.",
        "task_type": "forensics",
    },
    {
        "prompt": "<security_event> CAM-04 Rear Gate 02:41 AM: subject carrying crowbar forcing door lock. </security_event>",
        "thinking": "<think>After-hours 02:41 AM. Token 'crowbar' and 'forcing door' match forced_entry pattern.</think>",
        "response": "<alert> [CRITICAL] forced_entry at CAM-04. Dispatch security patrol immediately.",
        "task_type": "security_event",
    },
    {
        "prompt": "<security_event> CAM-01 Lobby: subject displayed pistol near service counter. </security_event>",
        "thinking": "<think>Token 'pistol' matches armed_threat lethal weapon pattern. Immediate life-safety risk.</think>",
        "response": "<alert> [CRITICAL] armed_threat at CAM-01. Trigger silent alarm and notify police.",
        "task_type": "security_event",
    },
    {
        "prompt": "<security_event> CAM-02 Perimeter: individual scaling wall near warehouse. </security_event>",
        "thinking": "<think>Token 'scaling wall' matches perimeter_breach pattern.</think>",
        "response": "<alert> [HIGH] perimeter_breach at CAM-02. Dispatch guard unit to warehouse wall.",
        "task_type": "security_event",
    },
    {
        "prompt": "<security_event> CAM-03 Gate: physical altercation and punching between two subjects. </security_event>",
        "thinking": "<think>Tokens 'altercation' and 'punching' match physical_violence pattern.</think>",
        "response": "<alert> [HIGH] physical_violence at CAM-03. Deploy response team.",
        "task_type": "security_event",
    },
    {
        "prompt": "<security_event> CAM-05 Reception: customer sitting quietly holding documents. </security_event>",
        "thinking": "<think>No burglary tools, weapons, or violence tokens detected. Routine activity.</think>",
        "response": "INFO: Normal activity at CAM-05. No threat detected.",
        "task_type": "security_event",
    },
    {
        "prompt": "<tanzania_law> Is an M-Pesa SMS admissible without a Section 40B certificate? </tanzania_law>",
        "thinking": "<think>Tanzania Evidence Act Cap 6 s.40A admits electronic records only when authenticated by a Section 40B certificate.</think>",
        "response": "No. Under Tanzania Evidence Act Cap 6 Section 40A and 40B, a Certificate of Electronic Evidence is required.",
        "task_type": "statute_verification",
    },
    {
        "prompt": "<tanzania_law> Which Penal Code section covers stealing by clerks or servants? </tanzania_law>",
        "thinking": "<think>Penal Code Cap 16 R.E. 2019 Section 271 governs stealing by servants, agents, or clerks.</think>",
        "response": "Penal Code Cap 16 R.E. 2019 Section 271 (Stealing by Servants, Agents or Clerks).",
        "task_type": "statute_verification",
    },
    {
        "prompt": "<tanzania_law> What statute governs computer-related financial fraud in Tanzania? </tanzania_law>",
        "thinking": "<think>Cybercrimes Act No. 14 of 2015 Section 14 criminalizes computer-related fraud causing unlawful loss.</think>",
        "response": "Cybercrimes Act No. 14 of 2015 Section 14 (Computer Related Fraud).",
        "task_type": "statute_verification",
    },
]
