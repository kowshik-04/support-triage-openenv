from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .models import (
    EnvStateModel,
    ObservationModel,
    RewardModel,
    StepInfoModel,
    StepResultModel,
    SupportAction,
    TicketSnapshot,
)
from .tasks import POLICIES, TASKS, TicketTask


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
    history: List[Dict[str, str]] = field(default_factory=list)
    action_fingerprints: Dict[str, int] = field(default_factory=dict)


class SupportTriageEnv:
    def __init__(self) -> None:
        self.tasks_by_id = {task.task_id: task for task in TASKS}
        self.task_order = [task.task_id for task in TASKS]
        self.task_cursor = 0
        self._state: Optional[InternalState] = None

    def reset(self, task_id: Optional[str] = None) -> ObservationModel:
        chosen_id = task_id or self.task_order[self.task_cursor % len(self.task_order)]
        self.task_cursor += 1
        if chosen_id not in self.tasks_by_id:
            raise ValueError(f"Unknown task_id: {chosen_id}")
        self._state = InternalState(task=self.tasks_by_id[chosen_id])
        return self._observation("Ticket opened. Review the customer issue and prepare the case.")

    def state(self) -> EnvStateModel:
        state = self._require_state()
        return EnvStateModel(
            task_id=state.task.task_id,
            difficulty=state.task.difficulty,
            step_count=state.step_count,
            max_steps=MAX_STEPS,
            done=state.done,
            searched_policies=state.searched_policies,
            classification=state.classification,
            priority=state.priority,
            route=state.route,
            tags=state.tags,
            reply=state.reply,
            reward_total=round(state.reward_total, 4),
            history=state.history,
        )

    def step(self, action: SupportAction) -> StepResultModel:
        state = self._require_state()
        if state.done:
            return self._finalize_step(0.0, ["Episode already completed."], violation="step_after_done")

        state.step_count += 1
        reasons: List[str] = []
        reward = 0.0
        violation: Optional[str] = None

        fingerprint = f"{action.action_type}:{(action.argument or '').strip().lower()}"
        previous_count = state.action_fingerprints.get(fingerprint, 0)
        state.action_fingerprints[fingerprint] = previous_count + 1
        if previous_count >= 1:
            reward -= 0.03
            reasons.append("Repeated the same action without adding new progress.")

        if action.action_type == "read_ticket":
            reward += 0.02
            reasons.append("Read the customer ticket.")
        elif action.action_type == "search_policy":
            delta, step_reasons = self._handle_search_policy(state, action.argument)
            reward += delta
            reasons.extend(step_reasons)
        elif action.action_type == "set_classification":
            delta, step_reasons = self._score_exact_field(
                current=state.classification,
                incoming=action.argument,
                expected=state.task.ideal_classification,
                label="classification",
                positive=0.2,
                negative=-0.08,
            )
            state.classification = self._normalize(action.argument)
            reward += delta
            reasons.extend(step_reasons)
        elif action.action_type == "set_priority":
            delta, step_reasons = self._score_exact_field(
                current=state.priority,
                incoming=action.argument,
                expected=state.task.ideal_priority,
                label="priority",
                positive=0.15,
                negative=-0.05,
            )
            state.priority = self._normalize(action.argument)
            reward += delta
            reasons.extend(step_reasons)
        elif action.action_type == "route_ticket":
            delta, step_reasons = self._score_exact_field(
                current=state.route,
                incoming=action.argument,
                expected=state.task.ideal_route,
                label="route",
                positive=0.15,
                negative=-0.06,
            )
            state.route = self._normalize(action.argument)
            reward += delta
            reasons.extend(step_reasons)
        elif action.action_type == "add_tag":
            delta, step_reasons = self._handle_add_tag(state, action.argument)
            reward += delta
            reasons.extend(step_reasons)
        elif action.action_type == "draft_reply":
            delta, step_reasons = self._handle_draft_reply(state, action.argument)
            reward += delta
            reasons.extend(step_reasons)
        elif action.action_type == "resolve_ticket":
            delta, step_reasons, resolved = self._handle_resolve(state)
            reward += delta
            reasons.extend(step_reasons)
            if resolved:
                state.done = True
        else:
            reward -= 0.05
            reasons.append("Unknown action.")
            violation = "unknown_action"

        if state.step_count >= MAX_STEPS and not state.done:
            state.done = True
            reward -= 0.1
            reasons.append("Episode ended because the maximum step limit was reached.")
            violation = violation or "max_steps_reached"

        state.reward_total += reward
        state.history.append(
            {
                "step": str(state.step_count),
                "action_type": action.action_type,
                "argument": action.argument or "",
                "reward": f"{reward:.4f}",
            }
        )
        return self._finalize_step(reward, reasons, violation=violation)

    def _handle_search_policy(self, state: InternalState, argument: Optional[str]) -> Tuple[float, List[str]]:
        query = self._normalize(argument)
        if not query:
            state.last_policy_results = []
            return -0.03, ["Policy search needs a useful query."]

        results = [
            f"{policy_id}: {text}"
            for policy_id, text in POLICIES.items()
            if query in policy_id or query in text.lower()
        ]
        state.last_policy_results = results[:3]

        matched_recommended = [
            policy_id for policy_id in state.task.recommended_policies if query in policy_id
        ]
        for policy_id in matched_recommended:
            if policy_id not in state.searched_policies:
                state.searched_policies.append(policy_id)

        if state.last_policy_results:
            if matched_recommended:
                return 0.08, [f"Found relevant policy guidance for {', '.join(matched_recommended)}."]
            return 0.03, ["Searched the policy base."]
        return -0.02, ["Policy search returned no useful matches."]

    def _handle_add_tag(self, state: InternalState, argument: Optional[str]) -> Tuple[float, List[str]]:
        tag = self._normalize(argument)
        if not tag:
            return -0.03, ["Tag cannot be empty."]
        if tag in state.tags:
            return -0.02, [f"Tag '{tag}' was already added."]
        state.tags.append(tag)
        if tag in state.task.required_tags or tag in state.task.optional_tags:
            return 0.05, [f"Added useful tag '{tag}'."]
        return -0.01, [f"Added non-essential tag '{tag}'."]

    def _handle_draft_reply(self, state: InternalState, argument: Optional[str]) -> Tuple[float, List[str]]:
        reply = (argument or "").strip()
        if not reply:
            return -0.04, ["Reply draft cannot be empty."]

        previous_score = self._reply_score(state.task, state.reply or "")
        new_score = self._reply_score(state.task, reply)
        delta = round((new_score - previous_score) * 0.25, 4)
        reasons = [f"Updated customer reply draft (quality {new_score:.2f})."]
        state.reply = reply
        return delta, reasons

    def _handle_resolve(self, state: InternalState) -> Tuple[float, List[str], bool]:
        grader_score, components, violations = self.grade_current_state()
        if grader_score >= state.task.resolve_threshold:
            return 0.2, ["Resolved ticket with sufficient completion quality."], True
        reasons = [
            "Attempted to resolve before all required work was complete.",
            f"Current quality {grader_score:.2f} is below threshold {state.task.resolve_threshold:.2f}.",
        ]
        reasons.extend([f"Missing: {name}" for name, score in components.items() if score < 0.99])
        if violations:
            reasons.extend([f"Violation: {violation}" for violation in violations])
        return -0.2, reasons, True

    def _score_exact_field(
        self,
        current: Optional[str],
        incoming: Optional[str],
        expected: str,
        label: str,
        positive: float,
        negative: float,
    ) -> Tuple[float, List[str]]:
        value = self._normalize(incoming)
        if not value:
            return negative, [f"{label.capitalize()} cannot be empty."]
        if value == expected:
            if current == expected:
                return -0.01, [f"{label.capitalize()} was already correct."]
            return positive, [f"Set correct {label}."]
        return negative, [f"Set incorrect {label}."]

    def grade_current_state(self) -> Tuple[float, Dict[str, float], List[str]]:
        state = self._require_state()
        task = state.task
        components: Dict[str, float] = {
            "classification": 1.0 if state.classification == task.ideal_classification else 0.0,
            "priority": 1.0 if state.priority == task.ideal_priority else 0.0,
            "route": 1.0 if state.route == task.ideal_route else 0.0,
            "research": round(
                len(set(state.searched_policies) & set(task.recommended_policies))
                / max(len(task.recommended_policies), 1),
                2,
            ),
            "tags": round(
                len(set(state.tags) & set(task.required_tags)) / max(len(task.required_tags), 1),
                2,
            ),
            "reply": self._reply_score(task, state.reply or ""),
            "resolved": 1.0 if state.done else 0.0,
        }

        violations: List[str] = []
        reply_text = (state.reply or "").lower()
        for bad_phrase in task.forbidden_reply_keywords:
            if bad_phrase.lower() in reply_text:
                violations.append(f"forbidden_reply_phrase:{bad_phrase}")

        score = (
            components["classification"] * 0.18
            + components["priority"] * 0.14
            + components["route"] * 0.14
            + components["research"] * 0.12
            + components["tags"] * 0.14
            + components["reply"] * 0.18
            + components["resolved"] * 0.10
        )

        if violations:
            score = max(0.0, score - 0.12 * len(violations))

        return round(min(score, 1.0), 4), components, violations

    def _reply_score(self, task: TicketTask, reply: str) -> float:
        text = reply.lower()
        if not text:
            return 0.0
        required_hits = sum(1 for keyword in task.required_reply_keywords if keyword.lower() in text)
        forbidden_hits = sum(1 for keyword in task.forbidden_reply_keywords if keyword.lower() in text)
        raw = required_hits / max(len(task.required_reply_keywords), 1)
        raw -= forbidden_hits * 0.25
        return round(min(max(raw, 0.0), 1.0), 2)

    def _finalize_step(
        self,
        reward: float,
        reasons: List[str],
        violation: Optional[str] = None,
    ) -> StepResultModel:
        state = self._require_state()
        grader_score, components, violations = self.grade_current_state()
        if violation:
            violations = [violation, *violations]
        return StepResultModel(
            observation=self._observation(" ".join(reasons)),
            reward=round(reward, 4),
            done=state.done,
            info=StepInfoModel(
                task_id=state.task.task_id,
                grader_score=grader_score,
                component_scores=components,
                violations=violations,
            ),
        )

    def _observation(self, guidance: str) -> ObservationModel:
        state = self._require_state()
        return ObservationModel(
            task_id=state.task.task_id,
            difficulty=state.task.difficulty,
            step_count=state.step_count,
            max_steps=MAX_STEPS,
            ticket=TicketSnapshot(
                ticket_id=state.task.task_id,
                customer_name=state.task.customer_name,
                subject=state.task.subject,
                body=state.task.body,
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
        )

    def _require_state(self) -> InternalState:
        if self._state is None:
            raise RuntimeError("Call reset() before step() or state().")
        return self._state

    @staticmethod
    def _normalize(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip().lower()
        return cleaned or None
