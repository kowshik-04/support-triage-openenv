from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        SupportRewardModel,
        SupportTriageAction,
        SupportTriageObservation,
        SupportTriageState,
        TicketSnapshot,
    )
    from ..tasks import POLICIES, TASKS, TASKS_BY_ID, TicketTask
except ImportError:
    from models import (
        SupportRewardModel,
        SupportTriageAction,
        SupportTriageObservation,
        SupportTriageState,
        TicketSnapshot,
    )
    from tasks import POLICIES, TASKS, TASKS_BY_ID, TicketTask


AVAILABLE_ACTIONS = [
    "read_ticket",
    "search_policy",
    "set_classification",
    "set_priority",
    "route_ticket",
    "add_tag",
    "draft_reply",
    "resolve_ticket",
]
MAX_STEPS = 12


@dataclass
class InternalState:
    task: TicketTask
    episode_id: str
    step_count: int = 0
    done: bool = False
    classification: Optional[str] = None
    priority: Optional[str] = None
    route: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    reply: Optional[str] = None
    last_policy_results: List[str] = field(default_factory=list)
    searched_policies: List[str] = field(default_factory=list)
    reward_total: float = 0.0
    violations: List[str] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    action_fingerprints: Dict[str, int] = field(default_factory=dict)


class SupportTriageEnvironment(
    Environment[SupportTriageAction, SupportTriageObservation, SupportTriageState]
):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self.task_order = [task.task_id for task in TASKS]
        self.task_cursor = 0
        self._state: Optional[InternalState] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: object,
    ) -> SupportTriageObservation:
        del seed
        task_id = kwargs.get("task_id")
        chosen_id = (
            str(task_id)
            if task_id is not None
            else self.task_order[self.task_cursor % len(self.task_order)]
        )
        self.task_cursor += 1
        if chosen_id not in TASKS_BY_ID:
            raise ValueError(f"Unknown task_id: {chosen_id}")
        self._state = InternalState(
            task=TASKS_BY_ID[chosen_id],
            episode_id=episode_id or str(uuid4()),
        )
        return self._build_observation(
            guidance="New ticket opened. Inspect the case, consult policy, and prepare a compliant resolution.",
            reward_model=SupportRewardModel(value=0.0, reasons=["Environment reset."]),
        )

    def step(
        self,
        action: SupportTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: object,
    ) -> SupportTriageObservation:
        del timeout_s, kwargs
        state = self._require_state()
        if state.done:
            reward_model = SupportRewardModel(
                value=-0.05,
                reasons=["Step requested after episode completion."],
                penalties=["step_after_done"],
            )
            return self._build_observation(
                guidance="Episode already completed. Reset before taking more actions.",
                reward_model=reward_model,
            )

        state.step_count += 1
        reasons: List[str] = []
        penalties: List[str] = []
        reward = 0.0

        fingerprint = action.model_dump_json(exclude_none=True)
        seen_count = state.action_fingerprints.get(fingerprint, 0)
        state.action_fingerprints[fingerprint] = seen_count + 1
        if seen_count > 0:
            reward -= 0.03
            penalties.append("repeated_action")
            reasons.append("Repeated the same action without new information.")

        if action.action_type == "read_ticket":
            reward += 0.02
            reasons.append("Reviewed the customer ticket.")
        elif action.action_type == "search_policy":
            delta, step_reasons, step_penalties = self._handle_search_policy(state, action.query)
            reward += delta
            reasons.extend(step_reasons)
            penalties.extend(step_penalties)
        elif action.action_type == "set_classification":
            delta, step_reasons = self._score_exact_field(
                current=state.classification,
                incoming=action.classification,
                expected=state.task.ideal_classification,
                label="classification",
                positive=0.20,
                negative=-0.08,
            )
            state.classification = self._normalize(action.classification)
            reward += delta
            reasons.extend(step_reasons)
        elif action.action_type == "set_priority":
            delta, step_reasons = self._score_exact_field(
                current=state.priority,
                incoming=action.priority,
                expected=state.task.ideal_priority,
                label="priority",
                positive=0.15,
                negative=-0.05,
            )
            state.priority = self._normalize(action.priority)
            reward += delta
            reasons.extend(step_reasons)
        elif action.action_type == "route_ticket":
            delta, step_reasons = self._score_exact_field(
                current=state.route,
                incoming=action.route,
                expected=state.task.ideal_route,
                label="route",
                positive=0.16,
                negative=-0.06,
            )
            state.route = self._normalize(action.route)
            reward += delta
            reasons.extend(step_reasons)
        elif action.action_type == "add_tag":
            delta, step_reasons, step_penalties = self._handle_add_tag(state, action.tag)
            reward += delta
            reasons.extend(step_reasons)
            penalties.extend(step_penalties)
        elif action.action_type == "draft_reply":
            delta, step_reasons, step_penalties = self._handle_draft_reply(state, action.reply_text)
            reward += delta
            reasons.extend(step_reasons)
            penalties.extend(step_penalties)
        elif action.action_type == "resolve_ticket":
            delta, step_reasons, step_penalties, should_finish = self._handle_resolve(state)
            reward += delta
            reasons.extend(step_reasons)
            penalties.extend(step_penalties)
            state.done = should_finish
        else:
            reward -= 0.05
            reasons.append("Unknown action type.")
            penalties.append("unknown_action")

        if state.step_count >= MAX_STEPS and not state.done:
            state.done = True
            reward -= 0.10
            reasons.append("Maximum step limit reached.")
            penalties.append("max_steps_reached")

        reward_model = self._reward_model(reward, reasons, penalties)
        state.reward_total = round(state.reward_total + reward_model.value, 4)
        state.violations = reward_model.penalties
        state.history.append(
            {
                "step": str(state.step_count),
                "action_type": action.action_type,
                "reward": f"{reward_model.value:.4f}",
                "details": self._action_summary(action),
            }
        )
        return self._build_observation(
            guidance=" ".join(reasons) if reasons else "Action applied.",
            reward_model=reward_model,
        )

    @property
    def state(self) -> SupportTriageState:
        state = self._require_state()
        return SupportTriageState(
            episode_id=state.episode_id,
            step_count=state.step_count,
            task_id=state.task.task_id,
            difficulty=state.task.difficulty,
            objective=state.task.objective,
            done=state.done,
            searched_policies=state.searched_policies,
            classification=state.classification,
            priority=state.priority,
            route=state.route,
            tags=state.tags,
            reply=state.reply,
            reward_total=state.reward_total,
            violations=state.violations,
            history=state.history,
        )

    def grade_current_state(self) -> Tuple[float, Dict[str, float], List[str]]:
        state = self._require_state()
        task = state.task
        research_score = self._research_score(state)
        tag_score = self._tag_score(state)
        reply_score = self._reply_score(task, state.reply or "")
        components: Dict[str, float] = {
            "classification": 1.0 if state.classification == task.ideal_classification else 0.0,
            "priority": 1.0 if state.priority == task.ideal_priority else 0.0,
            "route": 1.0 if state.route == task.ideal_route else 0.0,
            "research": research_score,
            "tags": tag_score,
            "reply": reply_score,
            "resolved": 1.0 if state.done else 0.0,
        }
        violations = list(state.violations)
        reply_text = (state.reply or "").lower()
        for phrase in task.forbidden_reply_keywords:
            if phrase.lower() in reply_text:
                violations.append(f"forbidden_reply_phrase:{phrase}")

        score = (
            components["research"] * 0.20
            + components["classification"] * 0.20
            + components["priority"] * 0.10
            + components["route"] * 0.20
            + components["tags"] * 0.10
            + components["reply"] * 0.10
            + components["resolved"] * 0.10
        )
        if violations:
            score = max(0.0, score - 0.05 * len(set(violations)))
        return round(min(score, 1.0), 4), components, sorted(set(violations))

    def task_grade(self, task_id: str, actions: List[SupportTriageAction]) -> float:
        self.reset(task_id=task_id)
        for action in actions:
            observation = self.step(action)
            if observation.done:
                break
        return observation.grader_score if actions else self.grade_current_state()[0]

    def _handle_search_policy(
        self,
        state: InternalState,
        query: Optional[str],
    ) -> Tuple[float, List[str], List[str]]:
        normalized = self._normalize(query)
        if not normalized:
            state.last_policy_results = []
            return -0.03, ["Policy search requires a non-empty query."], ["empty_search_query"]

        results = [
            f"{policy_id}: {text}"
            for policy_id, text in POLICIES.items()
            if normalized in policy_id or normalized in text.lower()
        ]
        state.last_policy_results = results[:3]
        hits = []
        for policy_id in state.task.recommended_policies:
            if normalized in policy_id and policy_id not in state.searched_policies:
                state.searched_policies.append(policy_id)
                hits.append(policy_id)

        if hits:
            reward = 0.12 if len(hits) == 1 else 0.16
            return reward, [f"Found directly relevant policy guidance for {', '.join(hits)}."], []
        if results:
            return 0.03, ["Searched policy knowledge base with partial relevance."], []
        return -0.02, ["No policy results matched the query."], ["irrelevant_search"]

    def _handle_add_tag(
        self,
        state: InternalState,
        tag: Optional[str],
    ) -> Tuple[float, List[str], List[str]]:
        normalized = self._normalize(tag)
        if not normalized:
            return -0.03, ["Tag cannot be empty."], ["empty_tag"]
        if normalized in state.tags:
            return -0.02, [f"Tag '{normalized}' is already present."], ["duplicate_tag"]
        state.tags.append(normalized)
        if normalized in state.task.required_tags or normalized in state.task.optional_tags:
            return 0.05, [f"Added useful tag '{normalized}'."], []
        return -0.01, [f"Added non-essential tag '{normalized}'."], ["noisy_tag"]

    def _handle_draft_reply(
        self,
        state: InternalState,
        reply_text: Optional[str],
    ) -> Tuple[float, List[str], List[str]]:
        reply = (reply_text or "").strip()
        if not reply:
            return -0.04, ["Reply draft cannot be empty."], ["empty_reply"]
        previous_score = self._reply_score(state.task, state.reply or "")
        new_score = self._reply_score(state.task, reply)
        delta = round((new_score - previous_score) * 0.20, 4)
        penalties: List[str] = []
        for bad_phrase in state.task.forbidden_reply_keywords:
            if bad_phrase.lower() in reply.lower():
                penalties.append(f"forbidden_reply_phrase:{bad_phrase}")
                delta -= 0.08
        state.reply = reply
        return delta, [f"Updated customer reply draft (quality {new_score:.2f})."], penalties

    def _handle_resolve(
        self,
        state: InternalState,
    ) -> Tuple[float, List[str], List[str], bool]:
        score, components, violations = self.grade_current_state()
        if score >= state.task.resolve_threshold:
            return 0.10, ["Resolved ticket with acceptable quality."], violations, True
        reasons = [
            "Attempted resolution before the case met the required completion threshold.",
            f"Current quality {score:.2f} is below {state.task.resolve_threshold:.2f}.",
        ]
        reasons.extend([f"Incomplete component: {name}" for name, value in components.items() if value < 0.99])
        return -0.20, reasons, ["premature_resolution", *violations], True

    def _build_observation(
        self,
        guidance: str,
        reward_model: SupportRewardModel,
    ) -> SupportTriageObservation:
        state = self._require_state()
        score, components, violations = self.grade_current_state()
        return SupportTriageObservation(
            done=state.done,
            reward=reward_model.value,
            metadata={
                "reward_model": reward_model.model_dump(),
                "task_title": state.task.title,
                "policy_catalog": sorted(POLICIES.keys()),
            },
            task_id=state.task.task_id,
            difficulty=state.task.difficulty,
            objective=state.task.objective,
            step_count=state.step_count,
            max_steps=MAX_STEPS,
            ticket=TicketSnapshot(
                ticket_id=state.task.task_id,
                customer_name=state.task.customer_name,
                subject=state.task.subject,
                body=state.task.body,
                region=state.task.region,
                account_tier=state.task.account_tier,
                sla_hours_remaining=state.task.sla_hours_remaining,
            ),
            current_classification=state.classification,
            current_priority=state.priority,
            current_route=state.route,
            current_tags=state.tags,
            latest_reply=state.reply,
            last_policy_results=state.last_policy_results,
            completed_actions=[item["action_type"] for item in state.history],
            available_actions=AVAILABLE_ACTIONS,
            guidance=guidance,
            grader_score=score,
            component_scores=components,
            violations=violations,
        )

    def _reward_model(
        self,
        reward: float,
        reasons: List[str],
        penalties: List[str],
    ) -> SupportRewardModel:
        _, components, _ = self.grade_current_state()
        return SupportRewardModel(
            value=round(reward, 4),
            reasons=reasons,
            component_scores=components,
            penalties=sorted(set(penalties)),
        )

    def _require_state(self) -> InternalState:
        if self._state is None:
            raise RuntimeError("Call reset() before step() or state().")
        return self._state

    def _reply_score(self, task: TicketTask, reply: str) -> float:
        text = reply.lower()
        if not text:
            return 0.0
        required_hits = sum(1 for keyword in task.required_reply_keywords if keyword.lower() in text)
        forbidden_hits = sum(1 for keyword in task.forbidden_reply_keywords if keyword.lower() in text)
        raw = required_hits / max(len(task.required_reply_keywords), 1)
        raw *= self._task_reply_multiplier(task, text)
        raw -= forbidden_hits * 0.25
        return round(min(max(raw, 0.0), 1.0), 2)

    def _research_score(self, state: InternalState) -> float:
        recommended = state.task.recommended_policies
        if not recommended:
            return 1.0
        matched = len(set(state.searched_policies) & set(recommended))
        return round(matched / len(recommended), 2)

    def _tag_score(self, state: InternalState) -> float:
        required = state.task.required_tags
        if not required:
            return 1.0
        matched = len(set(state.tags) & set(required))
        return round(matched / len(required), 2)

    def _task_reply_multiplier(self, task: TicketTask, text: str) -> float:
        if task.task_type == "duplicate_charge":
            return 1.0 if "refund" in text and "duplicate" in text else 0.5
        if task.task_type == "refund_outside_window":
            return 1.0 if "cannot" in text and "window" in text else 0.35
        if task.task_type == "account_takeover":
            return 1.0 if "verify" in text and "secure" in text and "urgent" in text else 0.45
        if task.task_type == "sla_breach_escalation":
            return 1.0 if "escalate" in text and "ownership" in text else 0.45
        if task.task_type == "gdpr_billing_mix":
            has_privacy = "delete" in text or "data" in text
            has_billing = "billing" in text or "charges" in text or "investigate" in text
            has_retention = "retain" in text or "legal" in text
            return 1.0 if has_privacy and has_billing and has_retention else 0.4
        return 1.0

    def _score_exact_field(
        self,
        current: Optional[str],
        incoming: Optional[str],
        expected: str,
        label: str,
        positive: float,
        negative: float,
    ) -> Tuple[float, List[str]]:
        normalized = self._normalize(incoming)
        if not normalized:
            return negative, [f"{label.capitalize()} cannot be empty."]
        if normalized == expected:
            if current == expected:
                return -0.01, [f"{label.capitalize()} was already correct."]
            return positive, [f"Set correct {label}."]
        return negative, [f"Set incorrect {label}."]

    def _action_summary(self, action: SupportTriageAction) -> str:
        if action.action_type == "search_policy":
            return action.query or ""
        if action.action_type == "set_classification":
            return action.classification or ""
        if action.action_type == "set_priority":
            return action.priority or ""
        if action.action_type == "route_ticket":
            return action.route or ""
        if action.action_type == "add_tag":
            return action.tag or ""
        if action.action_type == "draft_reply":
            return (action.reply_text or "")[:120]
        return ""

    @staticmethod
    def _normalize(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        return normalized or None
