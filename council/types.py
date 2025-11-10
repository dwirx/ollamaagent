from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Personality(BaseModel):
    name: str
    model: str
    traits: str
    perspective: str
    persistence: float = Field(0.5, ge=0.0, le=1.0)  # resistance to change
    reasoning_depth: int = 1
    truth_seeking: float = Field(0.7, ge=0.0, le=1.0)


class Argument(BaseModel):
    author: str
    content: str
    iteration: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Vote(BaseModel):
    voter: str
    ranking: List[str]  # names of personalities ranked best->worst
    iteration: int


class IterationResult(BaseModel):
    iteration: int
    arguments: List[Argument]
    votes: List[Vote]
    consensus_reached: bool = False
    consensus_candidate: Optional[str] = None


class DebateConfig(BaseModel):
    title: Optional[str] = None
    question: str
    judge_model: str = "kimi-k2:1t-cloud"
    min_iterations: int = 2
    max_iterations: int = 5
    consensus_threshold: float = 0.6  # fraction of first-place votes


class DebateState(BaseModel):
    config: DebateConfig
    personalities: List[Personality]
    iterations: List[IterationResult] = Field(default_factory=list)
    judge_decision: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


