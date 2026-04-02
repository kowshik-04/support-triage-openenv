from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


TaskArchetype = Literal[
    "duplicate_charge",
    "refund_outside_window",
    "account_takeover",
    "sla_breach_escalation",
    "gdpr_billing_mix",
]


class TicketTask(BaseModel):
    task_id: str
    task_type: TaskArchetype
    difficulty: Literal["easy", "medium", "hard"]
    title: str
    objective: str
    customer_name: str
    subject: str
    body: str
    region: str
    account_tier: str
    sla_hours_remaining: int
    ideal_classification: str
    ideal_priority: str
    ideal_route: str
    required_tags: List[str]
    optional_tags: List[str] = Field(default_factory=list)
    required_reply_keywords: List[str]
    forbidden_reply_keywords: List[str] = Field(default_factory=list)
    recommended_policies: List[str]
    resolve_threshold: float = 0.75


POLICIES: Dict[str, str] = {
    "refund_policy": (
        "Billing operations can reverse an accidental or duplicate charge after confirming the order id, "
        "charge date, and that the request falls inside the 30-day refund window. Requests outside the "
        "window should be declined politely unless a billing defect is proven."
    ),
    "enterprise_sla": (
        "Enterprise accounts should be prioritized when SLA remaining time is low. "
        "Escalate before breach when under 8 hours remain, and mark breached enterprise tickets as urgent."
    ),
    "account_security": (
        "Suspected account takeovers require urgent routing to trust_and_safety. "
        "Advise the customer to secure email access, rotate passwords, and complete identity verification."
    ),
    "privacy_deletion": (
        "Privacy deletion requests are owned by privacy_legal. Agents should acknowledge the request, "
        "set expectations for processing, and explain that some records may be retained for legal obligations."
    ),
    "billing_disputes": (
        "Billing disputes should be investigated with invoice or card references. "
        "Do not promise an immediate guaranteed refund before review."
    ),
    "tone_guide": (
        "Write calm, concise, action-oriented responses. Avoid guarantees that exceed policy or request secrets."
    ),
}


TASKS: List[TicketTask] = [
    TicketTask(
        task_id="duplicate_charge_easy",
        task_type="duplicate_charge",
        difficulty="easy",
        title="Duplicate Charge",
        objective="Process a straightforward duplicate-charge refund request inside the refund window.",
        customer_name="Jamie Rivera",
        subject="Need refund for duplicate team subscription charge",
        body=(
            "Hi support, I was charged twice for our April team subscription yesterday. "
            "Order #BIL-1042 shows the duplicate charge. Please help reverse the extra payment."
        ),
        region="US",
        account_tier="growth",
        sla_hours_remaining=19,
        ideal_classification="billing_refund",
        ideal_priority="normal",
        ideal_route="billing_ops",
        required_tags=["refund", "duplicate_charge"],
        required_reply_keywords=["refund", "duplicate", "billing", "order", "30-day"],
        recommended_policies=["refund_policy"],
        resolve_threshold=0.75,
    ),
    TicketTask(
        task_id="refund_outside_window_easy_plus",
        task_type="refund_outside_window",
        difficulty="easy",
        title="Refund Outside Window",
        objective="Politely decline a refund request that falls outside the refund window while explaining the reason.",
        customer_name="Nina Brooks",
        subject="Refund request for January annual plan charge",
        body=(
            "Hello, I am asking for a refund on our January annual plan renewal because we stopped using the product. "
            "The renewal was 47 days ago on invoice INV-7781. Please refund the charge today."
        ),
        region="US",
        account_tier="starter",
        sla_hours_remaining=22,
        ideal_classification="billing_refund",
        ideal_priority="normal",
        ideal_route="billing_ops",
        required_tags=["refund"],
        optional_tags=["duplicate_charge"],
        required_reply_keywords=["refund", "window", "47", "cannot", "billing"],
        forbidden_reply_keywords=["approved", "refund will be processed", "guaranteed refund"],
        recommended_policies=["refund_policy"],
        resolve_threshold=0.74,
    ),
    TicketTask(
        task_id="account_takeover_medium",
        task_type="account_takeover",
        difficulty="medium",
        title="Account Takeover",
        objective="Urgently route a suspected account takeover and guide the customer through safe next steps.",
        customer_name="Avery Chen",
        subject="Someone changed my password and I am locked out",
        body=(
            "I think my account was hacked overnight. My password no longer works, a new device appeared in login history, "
            "and I am seeing failed MFA texts. Please lock the account and tell me what to do next."
        ),
        region="US",
        account_tier="enterprise",
        sla_hours_remaining=4,
        ideal_classification="security_incident",
        ideal_priority="urgent",
        ideal_route="trust_and_safety",
        required_tags=["account_takeover", "identity_verification"],
        required_reply_keywords=["urgent", "secure", "password", "verify", "trust"],
        forbidden_reply_keywords=["share your password", "ignore this"],
        recommended_policies=["account_security", "enterprise_sla"],
        resolve_threshold=0.82,
    ),
    TicketTask(
        task_id="sla_breach_escalation_medium_plus",
        task_type="sla_breach_escalation",
        difficulty="medium",
        title="SLA Breach Escalation",
        objective="Escalate an enterprise ticket that is close to or past SLA breach and communicate ownership clearly.",
        customer_name="Priya Raman",
        subject="Production export failure still unresolved",
        body=(
            "We opened this incident yesterday and our data export is still failing for our finance team. "
            "We are an enterprise customer and have not received a meaningful update. Our deadline is in two hours."
        ),
        region="IN",
        account_tier="enterprise",
        sla_hours_remaining=1,
        ideal_classification="security_incident",
        ideal_priority="urgent",
        ideal_route="trust_and_safety",
        required_tags=["identity_verification"],
        optional_tags=["account_takeover"],
        required_reply_keywords=["escalate", "enterprise", "urgent", "update", "ownership"],
        recommended_policies=["enterprise_sla"],
        resolve_threshold=0.78,
    ),
    TicketTask(
        task_id="gdpr_billing_mix_hard",
        task_type="gdpr_billing_mix",
        difficulty="hard",
        title="GDPR + Billing Mix",
        objective=(
            "Handle a GDPR deletion request mixed with a billing dispute while preserving correct privacy, legal, and billing language."
        ),
        customer_name="Morgan Patel",
        subject="Delete my account and explain two unknown charges",
        body=(
            "I am based in Germany and want my account deleted under GDPR. I also found two charges on card ending 9912 "
            "that I do not recognize from March. Please erase my data and refund those charges immediately."
        ),
        region="DE",
        account_tier="enterprise",
        sla_hours_remaining=6,
        ideal_classification="privacy_request",
        ideal_priority="high",
        ideal_route="privacy_legal",
        required_tags=["privacy", "billing_dispute", "retention_review"],
        required_reply_keywords=["delete", "data", "billing", "investigate", "retain", "legal"],
        forbidden_reply_keywords=["immediate refund guaranteed", "all records deleted today"],
        recommended_policies=["privacy_deletion", "billing_disputes", "enterprise_sla"],
        resolve_threshold=0.86,
    ),
]


TASKS_BY_ID = {task.task_id: task for task in TASKS}
