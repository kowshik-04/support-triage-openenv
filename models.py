from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
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


class TicketSnapshot(BaseModel):
    ticket_id: str
    customer_name: str
    subject: str
    body: str
    region: str
    account_tier: str
    sla_hours_remaining: int


class SupportRewardModel(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)
    component_scores: Dict[str, float] = Field(default_factory=dict)
    penalties: List[str] = Field(default_factory=list)


class SupportTriageAction(Action):
    action_type: ActionType
    query: Optional[str] = Field(default=None, description="Policy query for search_policy.")
    classification: Optional[str] = Field(default=None, description="Classification for set_classification.")
    priority: Optional[str] = Field(default=None, description="Priority for set_priority.")
    route: Optional[str] = Field(default=None, description="Queue for route_ticket.")
    tag: Optional[str] = Field(default=None, description="Tag value for add_tag.")
    reply_text: Optional[str] = Field(default=None, description="Reply body for draft_reply.")


class SupportTriageObservation(Observation):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
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
    available_actions: List[ActionType] = Field(default_factory=list)
    guidance: str
    grader_score: float = Field(default=0.0, ge=0.0, le=1.0)
    component_scores: Dict[str, float] = Field(default_factory=dict)
    violations: List[str] = Field(default_factory=list)


class SupportTriageState(State):
    task_id: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    objective: str = ""
    done: bool = False
    searched_policies: List[str] = Field(default_factory=list)
    classification: Optional[str] = None
    priority: Optional[str] = None
    route: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    reply: Optional[str] = None
    reward_total: float = 0.0
    violations: List[str] = Field(default_factory=list)
    history: List[Dict[str, str]] = Field(default_factory=list)
