from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal[
    "read_ticket",
    "search_policy",
    "set_classification",
    "set_priority",
    "route_ticket",
    "add_tag",
    "draft_reply",
    "resolve_ticket",
]


class SupportAction(BaseModel):
    action_type: ActionType
    argument: Optional[str] = Field(
        default=None,
        description="Single string argument used by policy search, field updates, tags, and replies.",
    )


class TicketSnapshot(BaseModel):
    ticket_id: str
    customer_name: str
    subject: str
    body: str


class ObservationModel(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    step_count: int
    max_steps: int
    ticket: TicketSnapshot
    current_classification: Optional[str] = None
    current_priority: Optional[str] = None
    current_route: Optional[str] = None
    current_tags: List[str] = Field(default_factory=list)
    latest_reply: Optional[str] = None
    last_policy_results: List[str] = Field(default_factory=list)
    completed_actions: List[str] = Field(default_factory=list)
    available_actions: List[ActionType]
    guidance: str


class RewardModel(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)


class StepInfoModel(BaseModel):
    task_id: str
    grader_score: float = Field(ge=0.0, le=1.0)
    component_scores: Dict[str, float] = Field(default_factory=dict)
    violations: List[str] = Field(default_factory=list)


class StepResultModel(BaseModel):
    observation: ObservationModel
    reward: float
    done: bool
    info: StepInfoModel


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default=None, description="Optional task identifier.")


class EnvStateModel(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    step_count: int
    max_steps: int
    done: bool
    searched_policies: List[str] = Field(default_factory=list)
    classification: Optional[str] = None
    priority: Optional[str] = None
    route: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    reply: Optional[str] = None
    reward_total: float = 0.0
    history: List[Dict[str, Any]] = Field(default_factory=list)
