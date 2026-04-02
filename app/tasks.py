from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class TicketTask(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    title: str
    customer_name: str
    subject: str
    body: str
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
        "Billing agents can issue a refund only after confirming the order number, charge amount, "
        "and that the request falls within the 30-day window."
    ),
    "account_security": (
        "Potential account takeovers must be routed to trust_and_safety with urgent priority. "
        "Agents should instruct the customer to secure email access, rotate passwords, and complete identity verification."
    ),
    "privacy_deletion": (
        "Deletion requests are handled by privacy_legal. Agents may acknowledge the request and explain that "
        "some billing or fraud-prevention records can be retained for legal obligations."
    ),
    "billing_disputes": (
        "When billing issues accompany another request, agents should collect invoice identifiers and promise investigation, "
        "not an immediate guaranteed refund."
    ),
    "tone_guide": (
        "Replies should be calm, concise, and action-oriented. Avoid guarantees that exceed policy."
    ),
}


TASKS: List[TicketTask] = [
    TicketTask(
        task_id="easy_refund_window",
        difficulty="easy",
        title="Refund request within policy window",
        customer_name="Jamie Rivera",
        subject="Need refund for duplicate team subscription charge",
        body=(
            "Hi support, I was charged twice for our April team subscription yesterday. "
            "Order #BIL-1042 shows the duplicate charge. Please help reverse the extra payment."
        ),
        ideal_classification="billing_refund",
        ideal_priority="normal",
        ideal_route="billing_ops",
        required_tags=["refund", "duplicate_charge"],
        required_reply_keywords=["refund", "order", "BIL-1042", "30-day", "billing"],
        recommended_policies=["refund_policy", "tone_guide"],
        resolve_threshold=0.72,
    ),
    TicketTask(
        task_id="medium_account_takeover",
        difficulty="medium",
        title="Possible account takeover with urgent routing",
        customer_name="Avery Chen",
        subject="Someone changed my password and I am locked out",
        body=(
            "I think my account was hacked overnight. My password no longer works, a new device appeared in login history, "
            "and I am seeing failed MFA texts. Please lock the account and tell me what to do next."
        ),
        ideal_classification="security_incident",
        ideal_priority="urgent",
        ideal_route="trust_and_safety",
        required_tags=["account_takeover", "identity_verification"],
        required_reply_keywords=["secure", "password", "verify", "trust", "urgent"],
        forbidden_reply_keywords=["share your password", "ignore this"],
        recommended_policies=["account_security", "tone_guide"],
        resolve_threshold=0.8,
    ),
    TicketTask(
        task_id="hard_privacy_billing_mix",
        difficulty="hard",
        title="GDPR deletion request mixed with billing dispute",
        customer_name="Morgan Patel",
        subject="Delete my account and explain two unknown charges",
        body=(
            "I am based in Germany and want my account deleted under GDPR. I also found two charges on card ending 9912 "
            "that I do not recognize from March. Please erase my data and refund those charges immediately."
        ),
        ideal_classification="privacy_request",
        ideal_priority="high",
        ideal_route="privacy_legal",
        required_tags=["privacy", "billing_dispute", "retention_review"],
        optional_tags=["gdpr"],
        required_reply_keywords=["delete", "data", "billing", "investigate", "retain", "legal"],
        forbidden_reply_keywords=["immediate refund guaranteed", "all records deleted today"],
        recommended_policies=["privacy_deletion", "billing_disputes", "tone_guide"],
        resolve_threshold=0.86,
    ),
]
