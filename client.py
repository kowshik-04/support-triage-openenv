from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

try:
    from .models import SupportTriageAction, SupportTriageObservation, SupportTriageState
except ImportError:
    from models import SupportTriageAction, SupportTriageObservation, SupportTriageState


class SupportTriageEnv(
    EnvClient[SupportTriageAction, SupportTriageObservation, SupportTriageState]
):
    def _step_payload(self, action: SupportTriageAction | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(action, dict):
            return SupportTriageAction(**action).model_dump(exclude_none=True)
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SupportTriageObservation]:
        observation = SupportTriageObservation(**payload["observation"])
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SupportTriageState:
        return SupportTriageState(**payload)
