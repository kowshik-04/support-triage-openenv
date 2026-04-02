from __future__ import annotations

import os
import sys
import unittest

from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import SupportTriageAction
from server.app import app
from server.support_triage_environment import SupportTriageEnvironment
from tasks import TASKS


class SupportTriageEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SupportTriageEnvironment()

    def test_environment_exposes_exactly_five_tasks(self) -> None:
        self.assertEqual(len(TASKS), 5)

    def test_duplicate_charge_reference_plan_scores_high(self) -> None:
        self.env.reset(task_id="duplicate_charge_easy")
        plan = [
            SupportTriageAction(action_type="search_policy", query="refund_policy"),
            SupportTriageAction(action_type="set_classification", classification="billing_refund"),
            SupportTriageAction(action_type="set_priority", priority="normal"),
            SupportTriageAction(action_type="route_ticket", route="billing_ops"),
            SupportTriageAction(action_type="add_tag", tag="refund"),
            SupportTriageAction(action_type="add_tag", tag="duplicate_charge"),
            SupportTriageAction(
                action_type="draft_reply",
                reply_text=(
                    "Hi Jamie, I reviewed order BIL-1042 and this duplicate charge is eligible for a refund "
                    "within the 30-day billing window. Billing will reverse the duplicate payment."
                ),
            ),
            SupportTriageAction(action_type="resolve_ticket"),
        ]
        for action in plan:
            final_observation = self.env.step(action)

        self.assertTrue(final_observation.done)
        self.assertGreaterEqual(final_observation.grader_score, 0.90)

    def test_refund_outside_window_rewards_correct_rejection(self) -> None:
        self.env.reset(task_id="refund_outside_window_easy_plus")
        partial = self.env.step(SupportTriageAction(action_type="search_policy", query="refund_policy"))
        self.assertGreater(partial.grader_score, 0.0)

        final_observation = self.env.step(
            SupportTriageAction(
                action_type="draft_reply",
                reply_text=(
                    "Hi Nina, I reviewed invoice INV-7781 and this request is outside the refund window at 47 days, "
                    "so we cannot approve a refund under the billing policy."
                ),
            )
        )
        self.assertGreaterEqual(final_observation.component_scores["reply"], 0.5)

    def test_hard_task_is_partial_without_reply_and_resolution(self) -> None:
        self.env.reset(task_id="gdpr_billing_mix_hard")
        for action in [
            SupportTriageAction(action_type="search_policy", query="privacy_deletion"),
            SupportTriageAction(action_type="search_policy", query="billing_disputes"),
            SupportTriageAction(action_type="set_classification", classification="privacy_request"),
            SupportTriageAction(action_type="set_priority", priority="high"),
            SupportTriageAction(action_type="route_ticket", route="privacy_legal"),
        ]:
            observation = self.env.step(action)

        self.assertFalse(observation.done)
        self.assertGreater(observation.grader_score, 0.35)
        self.assertLess(observation.grader_score, 0.80)

    def test_http_endpoints_return_top_level_grader_score(self) -> None:
        client = TestClient(app)
        health = client.get("/health")
        self.assertEqual(health.status_code, 200)

        reset = client.post("/reset", json={"task_id": "account_takeover_medium"})
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(reset.json()["observation"]["task_id"], "account_takeover_medium")
        self.assertIn("grader_score", reset.json())
        self.assertEqual(reset.json()["grader_score"], 0.0)

        step = client.post(
            "/step",
            json={"action_type": "search_policy", "query": "account_security"},
        )
        self.assertEqual(step.status_code, 200)
        payload = step.json()
        self.assertIn("grader_score", payload)
        self.assertGreater(payload["grader_score"], 0.0)

        state = client.get("/state")
        self.assertEqual(state.status_code, 200)
        self.assertEqual(state.json()["task_id"], "account_takeover_medium")


if __name__ == "__main__":
    unittest.main()
