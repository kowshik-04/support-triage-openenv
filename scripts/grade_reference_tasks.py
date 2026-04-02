#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import SupportTriageAction
from server.support_triage_environment import SupportTriageEnvironment


REFERENCE_PLANS = {
    "easy_refund_window": [
        SupportTriageAction(action_type="search_policy", query="refund_policy"),
        SupportTriageAction(action_type="set_classification", classification="billing_refund"),
        SupportTriageAction(action_type="set_priority", priority="normal"),
        SupportTriageAction(action_type="route_ticket", route="billing_ops"),
        SupportTriageAction(action_type="add_tag", tag="refund"),
        SupportTriageAction(action_type="add_tag", tag="duplicate_charge"),
        SupportTriageAction(
            action_type="draft_reply",
            reply_text=(
                "Hi Jamie, I reviewed order BIL-1042 and billing can process the refund within the 30-day window. "
                "We are routing this to billing to reverse the duplicate charge."
            ),
        ),
        SupportTriageAction(action_type="resolve_ticket"),
    ],
    "medium_account_takeover": [
        SupportTriageAction(action_type="search_policy", query="account_security"),
        SupportTriageAction(action_type="search_policy", query="enterprise_sla"),
        SupportTriageAction(action_type="set_classification", classification="security_incident"),
        SupportTriageAction(action_type="set_priority", priority="urgent"),
        SupportTriageAction(action_type="route_ticket", route="trust_and_safety"),
        SupportTriageAction(action_type="add_tag", tag="account_takeover"),
        SupportTriageAction(action_type="add_tag", tag="identity_verification"),
        SupportTriageAction(
            action_type="draft_reply",
            reply_text=(
                "Hi Avery, this is being treated as an urgent trust case. Please secure your email, rotate any reused password, "
                "and verify your identity so trust and safety can recover the account safely."
            ),
        ),
        SupportTriageAction(action_type="resolve_ticket"),
    ],
    "hard_privacy_billing_mix": [
        SupportTriageAction(action_type="search_policy", query="privacy_deletion"),
        SupportTriageAction(action_type="search_policy", query="billing_disputes"),
        SupportTriageAction(action_type="search_policy", query="enterprise_sla"),
        SupportTriageAction(action_type="set_classification", classification="privacy_request"),
        SupportTriageAction(action_type="set_priority", priority="high"),
        SupportTriageAction(action_type="route_ticket", route="privacy_legal"),
        SupportTriageAction(action_type="add_tag", tag="privacy"),
        SupportTriageAction(action_type="add_tag", tag="billing_dispute"),
        SupportTriageAction(action_type="add_tag", tag="retention_review"),
        SupportTriageAction(
            action_type="draft_reply",
            reply_text=(
                "Hi Morgan, we will process your request to delete account data and send it to privacy legal for review. "
                "We also opened a billing investigation for the March charges. Some records may be retained for legal obligations while we investigate."
            ),
        ),
        SupportTriageAction(action_type="resolve_ticket"),
    ],
}


def main() -> None:
    env = SupportTriageEnvironment()
    for task_id, actions in REFERENCE_PLANS.items():
        score = env.task_grade(task_id, actions)
        print(f"{task_id}: {score:.4f}")


if __name__ == "__main__":
    main()
