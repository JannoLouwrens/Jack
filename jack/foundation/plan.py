"""
HIERARCHICAL PLAN - GoalAct-Inspired Dynamic Planning

This module implements production-grade hierarchical planning based on SOTA research:

1. GoalAct (NCIIP 2025 Best Paper) - Global planning + hierarchical execution
2. Plan-and-Act - Chain-of-thought before planning
3. HiPlan - Global milestone guides + local stepwise hints
4. HTN Planning - Hierarchical task network decomposition

Core Components:
1. PRIMITIVE       - Executable actions (Jack's 5 primitives + extensions)
2. STEP            - Hierarchical step with decomposition
3. PLAN            - Complete hierarchical plan
4. SKILL           - High-level reusable skill patterns (GoalAct)
5. GLOBAL PLANNER  - LLM-driven global planning with continuous updates
6. REPLANNER       - Dynamic replanning on failure
7. PLAN MEMORY     - Store and retrieve successful plan patterns

Design Principles:
- LLM-First: LLM makes ALL planning decisions
- Dynamic Replanning: Adapt plan on failure or environment change
- Skill-Based Decomposition: Reusable high-level skills
- Global-Local: Global strategy with local tactical steps
- Explanation: Every decomposition is explained

References:
- GoalAct: https://arxiv.org/abs/2504.16563 (NCIIP 2025 Best Paper)
- Plan-and-Act: https://arxiv.org/html/2503.09572v3
- HiPlan: Hierarchical planning with milestone library
- HTN: https://en.wikipedia.org/wiki/Hierarchical_task_network
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    List, Dict, Optional, Any, Callable, Tuple,
    TypeVar, Generic, Protocol, runtime_checkable, Union
)
from enum import Enum, auto
from datetime import datetime
import uuid
import json
import asyncio

from jack.foundation.types import Result, Ok, Err, Option, Some, NONE, Error, ErrorCode
from jack.foundation.state import State, Goal, GoalType


# =============================================================================
# REASONER PROTOCOL (for LLM-driven planning)
# =============================================================================

@runtime_checkable
class Reasoner(Protocol):
    """Protocol for LLM reasoning - used for planning decisions."""

    async def reason(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Make a reasoning call to the LLM."""
        ...


class SimpleReasoner:
    """Simple reasoner for testing."""

    async def reason(self, prompt: str, context: Dict[str, Any] = None) -> str:
        prompt_lower = prompt.lower()

        if "decompose" in prompt_lower or "break down" in prompt_lower:
            return json.dumps({
                "steps": [
                    {"description": "Analyze requirements", "skill": "analysis"},
                    {"description": "Execute main task", "skill": "execution"},
                    {"description": "Verify results", "skill": "verification"},
                ],
                "reasoning": "Standard task decomposition pattern",
            })

        if "replan" in prompt_lower or "alternative" in prompt_lower:
            return json.dumps({
                "new_steps": [
                    {"description": "Try alternative approach", "skill": "fallback"},
                ],
                "reasoning": "Switching to fallback strategy",
            })

        if "select skill" in prompt_lower:
            return json.dumps({"skill": "general", "confidence": 0.8})

        return json.dumps({"result": "ok"})


# =============================================================================
# ENUMS
# =============================================================================

class PlanStatus(Enum):
    """Status of a plan or step."""
    PENDING = auto()        # Not yet started
    IN_PROGRESS = auto()    # Currently executing
    COMPLETED = auto()      # Successfully finished
    FAILED = auto()         # Failed (might retry)
    SKIPPED = auto()        # Skipped (dependency failed)
    CANCELLED = auto()      # Cancelled by user/system
    REPLANNING = auto()     # Being replanned (GoalAct)


class StepType(Enum):
    """Type of step in the plan."""
    INTENT = auto()         # High-level goal (needs decomposition)
    STEP = auto()           # Concrete step (needs primitives)
    PRIMITIVE = auto()      # Executable action
    CHECKPOINT = auto()     # Verification point
    PARALLEL = auto()       # Multiple steps in parallel
    CONDITIONAL = auto()    # Conditional branching
    LOOP = auto()           # Repeated execution
    SKILL = auto()          # High-level skill (GoalAct)


class PrimitiveType(Enum):
    """Types of primitive actions (Jack's 5 primitives + extensions)."""
    SHELL_RUN = auto()
    FILE_READ = auto()
    FILE_WRITE = auto()
    HTTP_REQUEST = auto()
    GET_STATE = auto()
    # Extended primitives
    LLM_CALL = auto()       # Call language model
    WAIT = auto()           # Wait/delay
    USER_INPUT = auto()     # Request user input


# =============================================================================
# PRIMITIVE
# =============================================================================

@dataclass(frozen=True)
class Primitive:
    """
    An executable primitive action.

    This is the lowest level - what actually gets executed.
    Maps directly to Jack's 5 primitives + extensions.
    """
    id: str
    primitive_type: PrimitiveType
    params: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    timeout_seconds: float = 30.0
    retry_count: int = 0
    max_retries: int = 3

    @staticmethod
    def shell(command: str, description: str = None, timeout: float = 30.0) -> Primitive:
        """Create a shell command primitive."""
        return Primitive(
            id=str(uuid.uuid4())[:8],
            primitive_type=PrimitiveType.SHELL_RUN,
            params={"command": command},
            description=description or f"Run: {command[:50]}",
            timeout_seconds=timeout,
        )

    @staticmethod
    def read_file(path: str, description: str = None) -> Primitive:
        """Create a file read primitive."""
        return Primitive(
            id=str(uuid.uuid4())[:8],
            primitive_type=PrimitiveType.FILE_READ,
            params={"path": path},
            description=description or f"Read: {path}",
        )

    @staticmethod
    def write_file(path: str, content: str, description: str = None) -> Primitive:
        """Create a file write primitive."""
        return Primitive(
            id=str(uuid.uuid4())[:8],
            primitive_type=PrimitiveType.FILE_WRITE,
            params={"path": path, "content": content},
            description=description or f"Write: {path}",
        )

    @staticmethod
    def http(
        method: str,
        url: str,
        body: Any = None,
        headers: Dict[str, str] = None,
        description: str = None,
    ) -> Primitive:
        """Create an HTTP request primitive."""
        return Primitive(
            id=str(uuid.uuid4())[:8],
            primitive_type=PrimitiveType.HTTP_REQUEST,
            params={
                "method": method,
                "url": url,
                "body": body,
                "headers": headers or {},
            },
            description=description or f"{method} {url}",
        )

    @staticmethod
    def llm_call(prompt: str, description: str = None) -> Primitive:
        """Create an LLM call primitive."""
        return Primitive(
            id=str(uuid.uuid4())[:8],
            primitive_type=PrimitiveType.LLM_CALL,
            params={"prompt": prompt},
            description=description or "LLM reasoning",
            timeout_seconds=120.0,
        )

    def with_retry(self, max_retries: int) -> Primitive:
        """Create copy with retry configuration."""
        return Primitive(
            id=self.id,
            primitive_type=self.primitive_type,
            params=self.params,
            description=self.description,
            timeout_seconds=self.timeout_seconds,
            retry_count=self.retry_count,
            max_retries=max_retries,
        )

    def can_retry(self) -> bool:
        """Check if this primitive can be retried."""
        return self.retry_count < self.max_retries

    def as_retried(self) -> Primitive:
        """Create a retry copy with incremented count."""
        return Primitive(
            id=self.id,
            primitive_type=self.primitive_type,
            params=self.params,
            description=self.description,
            timeout_seconds=self.timeout_seconds,
            retry_count=self.retry_count + 1,
            max_retries=self.max_retries,
        )


# =============================================================================
# CHECKPOINT
# =============================================================================

@dataclass(frozen=True)
class Checkpoint:
    """
    A verification point in the plan.

    Checkpoints ensure we're on track before proceeding.
    """
    id: str
    description: str
    check_type: str  # "state_check", "output_check", "llm_verify"
    criteria: Dict[str, Any] = field(default_factory=dict)
    required: bool = True  # If False, checkpoint failure is warning only
    on_failure: Optional[str] = None  # Step ID to jump to on failure

    @staticmethod
    def state_check(description: str, conditions: Dict[str, Any]) -> Checkpoint:
        """Create a state-based checkpoint."""
        return Checkpoint(
            id=str(uuid.uuid4())[:8],
            description=description,
            check_type="state_check",
            criteria=conditions,
        )

    @staticmethod
    def output_check(description: str, expected: Any) -> Checkpoint:
        """Create an output-based checkpoint."""
        return Checkpoint(
            id=str(uuid.uuid4())[:8],
            description=description,
            check_type="output_check",
            criteria={"expected": expected},
        )

    @staticmethod
    def llm_verify(description: str, question: str) -> Checkpoint:
        """Create an LLM-verified checkpoint."""
        return Checkpoint(
            id=str(uuid.uuid4())[:8],
            description=description,
            check_type="llm_verify",
            criteria={"question": question},
        )


# =============================================================================
# SKILL (GoalAct Pattern)
# =============================================================================

@dataclass
class Skill:
    """
    A high-level reusable skill (GoalAct pattern).

    Skills are abstractions over common task patterns:
    - searching: Find information
    - coding: Write or modify code
    - writing: Generate text content
    - analysis: Analyze data or information
    - verification: Check results
    - execution: Run commands or actions

    Skills reduce planning complexity by providing reusable patterns.
    """
    name: str
    description: str
    required_capabilities: List[str] = field(default_factory=list)
    typical_steps: List[str] = field(default_factory=list)
    success_patterns: List[Dict[str, Any]] = field(default_factory=list)
    failure_patterns: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def searching() -> Skill:
        return Skill(
            name="searching",
            description="Find information from various sources",
            required_capabilities=["file_read", "http_request", "llm_call"],
            typical_steps=["formulate query", "search sources", "filter results", "extract info"],
        )

    @staticmethod
    def coding() -> Skill:
        return Skill(
            name="coding",
            description="Write or modify code",
            required_capabilities=["file_read", "file_write", "shell_run"],
            typical_steps=["understand requirements", "write code", "test code", "fix issues"],
        )

    @staticmethod
    def analysis() -> Skill:
        return Skill(
            name="analysis",
            description="Analyze data or information",
            required_capabilities=["file_read", "llm_call"],
            typical_steps=["gather data", "process data", "identify patterns", "summarize findings"],
        )

    @staticmethod
    def execution() -> Skill:
        return Skill(
            name="execution",
            description="Execute commands or actions",
            required_capabilities=["shell_run", "file_write"],
            typical_steps=["prepare environment", "execute command", "capture output", "verify result"],
        )

    @staticmethod
    def verification() -> Skill:
        return Skill(
            name="verification",
            description="Verify results and check correctness",
            required_capabilities=["file_read", "llm_call"],
            typical_steps=["define criteria", "check output", "compare expectations", "report status"],
        )


# =============================================================================
# STEP
# =============================================================================

@dataclass
class Step:
    """
    A step in the plan - can be abstract (needs decomposition) or concrete.

    Steps form a tree structure:
    - Intent (root): What we want to achieve
    - Skills (branches): High-level skills to apply (GoalAct)
    - Steps (branches): How we'll do it
    - Primitives (leaves): What we'll execute
    """
    id: str
    step_type: StepType
    description: str
    goal: Optional[Goal] = None  # What this step achieves

    # GoalAct: Skill-based decomposition
    skill: Optional[Skill] = None
    reasoning: str = ""  # Why this step was chosen (Plan-and-Act CoT)

    # Decomposition
    children: List[Step] = field(default_factory=list)
    primitive: Optional[Primitive] = None  # If this is a primitive step

    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Step IDs

    # Verification
    preconditions: List[Checkpoint] = field(default_factory=list)
    postconditions: List[Checkpoint] = field(default_factory=list)

    # Recovery (GoalAct: Alternative plans)
    alternatives: List[Step] = field(default_factory=list)  # Alternative approaches
    fallback: Optional[Step] = None  # Fallback if this fails
    on_failure: Optional[Callable[[Error], Step]] = None  # Dynamic fallback

    # Execution state (mutable for tracking)
    status: PlanStatus = PlanStatus.PENDING
    result: Optional[Any] = None
    error: Optional[Error] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempt_count: int = 0
    max_attempts: int = 3

    # =========================================================================
    # QUERIES
    # =========================================================================

    @property
    def is_leaf(self) -> bool:
        """Is this a leaf node (executable)?"""
        return self.step_type == StepType.PRIMITIVE or (
            not self.children and self.primitive is not None
        )

    @property
    def is_abstract(self) -> bool:
        """Does this step need decomposition?"""
        return self.step_type in (StepType.INTENT, StepType.SKILL) and not self.children

    @property
    def is_complete(self) -> bool:
        """Has this step completed (success or failure)?"""
        return self.status in (PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.SKIPPED)

    @property
    def is_successful(self) -> bool:
        """Did this step complete successfully?"""
        return self.status == PlanStatus.COMPLETED

    @property
    def can_retry(self) -> bool:
        """Can this step be retried?"""
        return self.attempt_count < self.max_attempts

    @property
    def has_alternatives(self) -> bool:
        """Are there alternative approaches?"""
        return len(self.alternatives) > 0 or self.fallback is not None

    @property
    def duration_seconds(self) -> Optional[float]:
        """How long did this step take?"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def get_all_primitives(self) -> List[Primitive]:
        """Get all primitives in this step and children."""
        primitives = []
        if self.primitive:
            primitives.append(self.primitive)
        for child in self.children:
            primitives.extend(child.get_all_primitives())
        return primitives

    def get_pending_steps(self) -> List[Step]:
        """Get all pending steps (DFS order)."""
        pending = []
        if self.status == PlanStatus.PENDING:
            pending.append(self)
        for child in self.children:
            pending.extend(child.get_pending_steps())
        return pending

    def get_next_executable(self) -> Optional[Step]:
        """Get the next step that can be executed."""
        if self.status != PlanStatus.PENDING:
            return None
        if self.is_leaf:
            return self
        for child in self.children:
            if child.status == PlanStatus.PENDING:
                next_step = child.get_next_executable()
                if next_step:
                    return next_step
        return None

    # =========================================================================
    # BUILDERS
    # =========================================================================

    @staticmethod
    def intent(description: str, goal: Goal = None, reasoning: str = "") -> Step:
        """Create an intent step (high-level)."""
        return Step(
            id=str(uuid.uuid4())[:8],
            step_type=StepType.INTENT,
            description=description,
            goal=goal,
            reasoning=reasoning,
        )

    @staticmethod
    def skill_step(skill: Skill, description: str = None, reasoning: str = "") -> Step:
        """Create a skill-based step (GoalAct pattern)."""
        return Step(
            id=str(uuid.uuid4())[:8],
            step_type=StepType.SKILL,
            description=description or skill.description,
            skill=skill,
            reasoning=reasoning,
        )

    @staticmethod
    def action(description: str, primitive: Primitive, reasoning: str = "") -> Step:
        """Create a primitive action step."""
        return Step(
            id=str(uuid.uuid4())[:8],
            step_type=StepType.PRIMITIVE,
            description=description,
            primitive=primitive,
            reasoning=reasoning,
        )

    @staticmethod
    def sequence(description: str, steps: List[Step], reasoning: str = "") -> Step:
        """Create a sequential step group."""
        parent = Step(
            id=str(uuid.uuid4())[:8],
            step_type=StepType.STEP,
            description=description,
            children=steps,
            reasoning=reasoning,
        )
        # Set up dependencies: each step depends on previous
        for i in range(1, len(steps)):
            steps[i].depends_on.append(steps[i-1].id)
        return parent

    @staticmethod
    def parallel(description: str, steps: List[Step], reasoning: str = "") -> Step:
        """Create a parallel step group."""
        return Step(
            id=str(uuid.uuid4())[:8],
            step_type=StepType.PARALLEL,
            description=description,
            children=steps,
            reasoning=reasoning,
        )

    @staticmethod
    def conditional(
        description: str,
        condition: Checkpoint,
        if_true: Step,
        if_false: Optional[Step] = None,
        reasoning: str = "",
    ) -> Step:
        """Create a conditional step."""
        step = Step(
            id=str(uuid.uuid4())[:8],
            step_type=StepType.CONDITIONAL,
            description=description,
            preconditions=[condition],
            children=[if_true] + ([if_false] if if_false else []),
            reasoning=reasoning,
        )
        return step

    def with_precondition(self, checkpoint: Checkpoint) -> Step:
        """Add a precondition checkpoint."""
        self.preconditions.append(checkpoint)
        return self

    def with_postcondition(self, checkpoint: Checkpoint) -> Step:
        """Add a postcondition checkpoint."""
        self.postconditions.append(checkpoint)
        return self

    def with_fallback(self, fallback: Step) -> Step:
        """Add a fallback step for failure recovery."""
        self.fallback = fallback
        return self

    def with_alternative(self, alternative: Step) -> Step:
        """Add an alternative approach (GoalAct)."""
        self.alternatives.append(alternative)
        return self

    def decompose(self, children: List[Step], reasoning: str = "") -> Step:
        """Decompose this step into children."""
        self.children = children
        self.reasoning = reasoning or self.reasoning
        if self.step_type == StepType.INTENT:
            self.step_type = StepType.STEP
        return self

    # =========================================================================
    # EXECUTION STATE
    # =========================================================================

    def mark_started(self) -> None:
        """Mark step as started."""
        self.status = PlanStatus.IN_PROGRESS
        self.started_at = datetime.now()
        self.attempt_count += 1

    def mark_completed(self, result: Any = None) -> None:
        """Mark step as completed."""
        self.status = PlanStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()

    def mark_failed(self, error: Error) -> None:
        """Mark step as failed."""
        self.status = PlanStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()

    def mark_skipped(self) -> None:
        """Mark step as skipped."""
        self.status = PlanStatus.SKIPPED
        self.completed_at = datetime.now()

    def mark_replanning(self) -> None:
        """Mark step as being replanned (GoalAct)."""
        self.status = PlanStatus.REPLANNING

    def reset(self) -> None:
        """Reset step to pending state."""
        self.status = PlanStatus.PENDING
        self.result = None
        self.error = None
        self.started_at = None
        self.completed_at = None


# =============================================================================
# GLOBAL PLANNER (GoalAct Pattern)
# =============================================================================

class GlobalPlanner:
    """
    LLM-driven global planner (GoalAct pattern).

    The global planner:
    1. Creates high-level strategy for the goal
    2. Selects appropriate skills for each phase
    3. Continuously updates the plan based on progress
    4. Generates alternative approaches

    This implements GoalAct's "continuously updated global planning mechanism".
    """

    def __init__(self, reasoner: Reasoner):
        self.reasoner = reasoner
        self.skills = {
            "searching": Skill.searching(),
            "coding": Skill.coding(),
            "analysis": Skill.analysis(),
            "execution": Skill.execution(),
            "verification": Skill.verification(),
        }

    async def create_global_plan(
        self,
        goal: Goal,
        state: State,
        context: Dict[str, Any] = None,
    ) -> List[Step]:
        """
        Create a global plan for a goal.

        Uses Chain-of-Thought (Plan-and-Act pattern) to:
        1. Analyze the goal and available resources
        2. Select appropriate skills
        3. Generate step sequence with reasoning
        """
        context = context or {}

        # Build context string
        entities_str = ", ".join([e.name for e in list(state.entities)[:10]])

        prompt = f"""Global Planning (GoalAct + Plan-and-Act Pattern)

Create a high-level plan for achieving this goal.

GOAL:
Intent: {goal.intent}
Type: {goal.goal_type.name}

AVAILABLE ENTITIES:
{entities_str}

AVAILABLE SKILLS:
- searching: Find information from various sources
- coding: Write or modify code
- analysis: Analyze data or information
- execution: Execute commands or actions
- verification: Verify results and check correctness

CONTEXT:
{context}

Think step by step (Chain-of-Thought):
1. What is the main objective?
2. What skills are needed?
3. What is the logical sequence of steps?
4. What could go wrong? (for alternatives)

Respond in JSON:
{{
    "reasoning": "Your step-by-step analysis",
    "steps": [
        {{
            "description": "Step description",
            "skill": "skill_name",
            "reasoning": "Why this step"
        }}
    ],
    "alternatives": [
        {{
            "description": "Alternative approach if main fails",
            "when_to_use": "Condition for using this alternative"
        }}
    ]
}}"""

        try:
            response = await self.reasoner.reason(prompt, context)
            parsed = json.loads(response)

            steps = []
            for step_data in parsed.get("steps", []):
                skill_name = step_data.get("skill", "execution")
                skill = self.skills.get(skill_name, Skill.execution())

                step = Step.skill_step(
                    skill=skill,
                    description=step_data.get("description", "Execute step"),
                    reasoning=step_data.get("reasoning", ""),
                )
                steps.append(step)

            # Add alternatives to last step if present
            if steps and parsed.get("alternatives"):
                for alt_data in parsed["alternatives"]:
                    alt_step = Step.intent(
                        description=alt_data.get("description", "Alternative approach"),
                        reasoning=alt_data.get("when_to_use", "If main approach fails"),
                    )
                    steps[-1].alternatives.append(alt_step)

            return steps

        except (json.JSONDecodeError, Exception) as e:
            # Fallback: simple 3-step plan
            return [
                Step.skill_step(Skill.analysis(), "Analyze requirements", "Default decomposition"),
                Step.skill_step(Skill.execution(), "Execute main task", "Default decomposition"),
                Step.skill_step(Skill.verification(), "Verify results", "Default decomposition"),
            ]

    async def decompose_skill_step(
        self,
        step: Step,
        state: State,
    ) -> List[Step]:
        """
        Decompose a skill-based step into concrete steps.

        This is the local planning that happens within each skill.
        """
        if not step.skill:
            return []

        prompt = f"""Skill Decomposition

Decompose this high-level skill step into concrete actions.

SKILL: {step.skill.name}
DESCRIPTION: {step.description}
TYPICAL STEPS: {step.skill.typical_steps}
REQUIRED CAPABILITIES: {step.skill.required_capabilities}

STATE:
Goal: {state.goal.intent if state.goal else 'Unknown'}

Generate concrete steps. Respond in JSON:
{{
    "steps": [
        {{
            "description": "Concrete action",
            "primitive_type": "shell_run|file_read|file_write|llm_call",
            "params": {{}},
            "reasoning": "Why this action"
        }}
    ]
}}"""

        try:
            response = await self.reasoner.reason(prompt)
            parsed = json.loads(response)

            steps = []
            for step_data in parsed.get("steps", []):
                prim_type_str = step_data.get("primitive_type", "llm_call")
                prim_type_map = {
                    "shell_run": PrimitiveType.SHELL_RUN,
                    "file_read": PrimitiveType.FILE_READ,
                    "file_write": PrimitiveType.FILE_WRITE,
                    "llm_call": PrimitiveType.LLM_CALL,
                    "http_request": PrimitiveType.HTTP_REQUEST,
                }
                prim_type = prim_type_map.get(prim_type_str, PrimitiveType.LLM_CALL)

                primitive = Primitive(
                    id=str(uuid.uuid4())[:8],
                    primitive_type=prim_type,
                    params=step_data.get("params", {}),
                    description=step_data.get("description", "Execute"),
                )

                concrete_step = Step.action(
                    description=step_data.get("description", "Execute"),
                    primitive=primitive,
                    reasoning=step_data.get("reasoning", ""),
                )
                steps.append(concrete_step)

            return steps

        except (json.JSONDecodeError, Exception):
            # Fallback: single LLM call
            return [
                Step.action(
                    "Execute with LLM",
                    Primitive.llm_call(f"Complete: {step.description}"),
                    "Fallback to LLM execution",
                )
            ]


# =============================================================================
# REPLANNER (Dynamic Replanning)
# =============================================================================

class Replanner:
    """
    Dynamic replanner for handling failures (GoalAct pattern).

    When a step fails, the replanner:
    1. Analyzes the failure
    2. Checks for alternatives
    3. Generates a new plan if needed
    4. Updates the global plan

    This implements GoalAct's adaptive planning capability.
    """

    def __init__(self, reasoner: Reasoner, global_planner: GlobalPlanner):
        self.reasoner = reasoner
        self.global_planner = global_planner

    async def replan_on_failure(
        self,
        failed_step: Step,
        error: Error,
        state: State,
        remaining_steps: List[Step],
    ) -> Result[List[Step], Error]:
        """
        Create a new plan when a step fails.

        Strategy:
        1. Try alternatives if available
        2. Generate new approach using LLM
        3. Adjust remaining steps if needed
        """
        # 1. Check for pre-defined alternatives
        if failed_step.alternatives:
            alt = failed_step.alternatives.pop(0)
            return Ok([alt] + remaining_steps)

        # 2. Check for fallback
        if failed_step.fallback:
            return Ok([failed_step.fallback] + remaining_steps)

        # 3. Ask LLM to generate alternative approach
        prompt = f"""Replanning After Failure

A step in the plan failed. Generate an alternative approach.

FAILED STEP:
Description: {failed_step.description}
Skill: {failed_step.skill.name if failed_step.skill else 'N/A'}
Reasoning: {failed_step.reasoning}

ERROR:
{error.message if error else 'Unknown error'}

REMAINING GOAL:
{state.goal.intent if state.goal else 'Complete the task'}

REMAINING STEPS:
{[s.description for s in remaining_steps[:3]]}

Generate an alternative approach. Respond in JSON:
{{
    "can_recover": true/false,
    "reasoning": "Analysis of what went wrong and how to fix",
    "new_steps": [
        {{
            "description": "Alternative step",
            "skill": "skill_name"
        }}
    ],
    "skip_remaining": false
}}"""

        try:
            response = await self.reasoner.reason(prompt)
            parsed = json.loads(response)

            if not parsed.get("can_recover", True):
                return Err(Error(
                    ErrorCode.EXECUTION_FAILED,
                    f"Cannot recover from failure: {parsed.get('reasoning', 'Unknown')}",
                ))

            new_steps = []
            for step_data in parsed.get("new_steps", []):
                skill_name = step_data.get("skill", "execution")
                skill = self.global_planner.skills.get(skill_name, Skill.execution())

                step = Step.skill_step(
                    skill=skill,
                    description=step_data.get("description", "Recovery step"),
                    reasoning=parsed.get("reasoning", "Recovery from failure"),
                )
                new_steps.append(step)

            if parsed.get("skip_remaining", False):
                return Ok(new_steps)
            else:
                return Ok(new_steps + remaining_steps)

        except (json.JSONDecodeError, Exception) as e:
            return Err(Error(
                ErrorCode.INTERNAL_ERROR,
                f"Replanning failed: {str(e)}",
            ))

    async def update_global_plan(
        self,
        plan: Plan,
        observation: Dict[str, Any],
    ) -> Plan:
        """
        Update the global plan based on new observations.

        This implements GoalAct's "continuously updated global planning".
        """
        # Check if plan needs adjustment based on observations
        prompt = f"""Global Plan Update

Based on new observations, should the plan be adjusted?

CURRENT PLAN PROGRESS:
Completed: {plan.progress:.0%}
Current Step: {plan.current_step_id}

OBSERVATIONS:
{observation}

REMAINING STEPS:
{[s.description for s in plan.get_all_steps() if s.status == PlanStatus.PENDING][:5]}

Should the plan be adjusted? Respond in JSON:
{{
    "needs_adjustment": true/false,
    "reason": "Why adjustment is/isn't needed",
    "adjustments": [
        {{"step_id": "id", "action": "skip|modify|add_before", "details": {{}}}}
    ]
}}"""

        try:
            response = await self.reasoner.reason(prompt)
            parsed = json.loads(response)

            if not parsed.get("needs_adjustment", False):
                return plan

            for adj in parsed.get("adjustments", []):
                step = plan.get_step(adj.get("step_id", ""))
                if step:
                    action = adj.get("action", "")
                    if action == "skip":
                        step.mark_skipped()
                    elif action == "modify":
                        step.description = adj.get("details", {}).get("new_description", step.description)

            return plan

        except (json.JSONDecodeError, Exception):
            return plan


# =============================================================================
# PLAN
# =============================================================================

@dataclass
class Plan:
    """
    A complete hierarchical plan with dynamic replanning support.

    The plan is a tree of Steps with:
    - Root: The top-level goal
    - Skills: High-level skills to apply (GoalAct)
    - Branches: Decomposed sub-goals
    - Leaves: Executable primitives

    Plans support:
    - Incremental execution (step by step)
    - Dynamic replanning on failure (GoalAct)
    - Recovery from failures (fallbacks + alternatives)
    - Verification at checkpoints
    - Explanation of all decisions (Plan-and-Act CoT)
    """
    id: str
    root: Step
    state: State  # Initial state when plan was created

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    global_reasoning: str = ""  # High-level reasoning (Plan-and-Act CoT)

    # Execution tracking
    current_step_id: Optional[str] = None
    execution_log: List[Dict[str, Any]] = field(default_factory=list)

    # Replanning tracking
    replan_count: int = 0
    max_replans: int = 5

    @staticmethod
    def create(goal: Goal, state: State, reasoning: str = "") -> Plan:
        """Create a new plan for a goal."""
        root = Step.intent(goal.intent, goal, reasoning)
        return Plan(
            id=str(uuid.uuid4())[:8],
            root=root,
            state=state,
            global_reasoning=reasoning,
        )

    # =========================================================================
    # QUERIES
    # =========================================================================

    @property
    def status(self) -> PlanStatus:
        """Overall plan status."""
        return self.root.status

    @property
    def is_complete(self) -> bool:
        """Has the plan finished executing?"""
        return self.root.is_complete

    @property
    def is_successful(self) -> bool:
        """Did the plan complete successfully?"""
        return self.root.is_successful

    @property
    def can_replan(self) -> bool:
        """Can we still replan?"""
        return self.replan_count < self.max_replans

    @property
    def progress(self) -> float:
        """Execution progress (0.0 to 1.0)."""
        total = len(self.get_all_steps())
        if total == 0:
            return 0.0
        completed = len([s for s in self.get_all_steps() if s.is_complete])
        return completed / total

    def get_all_steps(self) -> List[Step]:
        """Get all steps in the plan (DFS order)."""
        steps = []
        def collect(step: Step):
            steps.append(step)
            for child in step.children:
                collect(child)
        collect(self.root)
        return steps

    def get_step(self, step_id: str) -> Optional[Step]:
        """Get a step by ID."""
        for step in self.get_all_steps():
            if step.id == step_id:
                return step
        return None

    def get_next_step(self) -> Optional[Step]:
        """Get the next step to execute."""
        return self.root.get_next_executable()

    def get_pending_primitives(self) -> List[Primitive]:
        """Get all pending primitives."""
        primitives = []
        for step in self.get_all_steps():
            if step.status == PlanStatus.PENDING and step.primitive:
                primitives.append(step.primitive)
        return primitives

    def get_remaining_steps(self) -> List[Step]:
        """Get all remaining (pending) steps."""
        return [s for s in self.get_all_steps() if s.status == PlanStatus.PENDING]

    # =========================================================================
    # MODIFICATION
    # =========================================================================

    def decompose_step(self, step_id: str, children: List[Step], reasoning: str = "") -> Result[None, Error]:
        """Decompose a step into children."""
        step = self.get_step(step_id)
        if step is None:
            return Err(Error(ErrorCode.NOT_FOUND, f"Step {step_id} not found"))
        step.decompose(children, reasoning)
        self.version += 1
        return Ok(None)

    def add_fallback(self, step_id: str, fallback: Step) -> Result[None, Error]:
        """Add a fallback to a step."""
        step = self.get_step(step_id)
        if step is None:
            return Err(Error(ErrorCode.NOT_FOUND, f"Step {step_id} not found"))
        step.fallback = fallback
        return Ok(None)

    def add_alternative(self, step_id: str, alternative: Step) -> Result[None, Error]:
        """Add an alternative approach to a step (GoalAct)."""
        step = self.get_step(step_id)
        if step is None:
            return Err(Error(ErrorCode.NOT_FOUND, f"Step {step_id} not found"))
        step.alternatives.append(alternative)
        return Ok(None)

    def replan_from(self, step_id: str, new_children: List[Step], reasoning: str = "") -> Result[None, Error]:
        """Re-plan from a failed step."""
        step = self.get_step(step_id)
        if step is None:
            return Err(Error(ErrorCode.NOT_FOUND, f"Step {step_id} not found"))
        step.reset()
        step.children = new_children
        step.reasoning = reasoning or step.reasoning
        self.version += 1
        self.replan_count += 1
        return Ok(None)

    def record_replan(self) -> None:
        """Record that a replan occurred."""
        self.replan_count += 1
        self.version += 1

    # =========================================================================
    # EXECUTION LOGGING
    # =========================================================================

    def log_execution(self, step_id: str, event: str, data: Any = None) -> None:
        """Log an execution event."""
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "step_id": step_id,
            "event": event,
            "data": data,
            "version": self.version,
        })

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def to_tree_string(self, indent: int = 0) -> str:
        """Convert plan to tree string for visualization."""
        lines = []

        def render(step: Step, depth: int):
            prefix = "  " * depth
            status_icon = {
                PlanStatus.PENDING: "[ ]",
                PlanStatus.IN_PROGRESS: "[~]",
                PlanStatus.COMPLETED: "[x]",
                PlanStatus.FAILED: "[!]",
                PlanStatus.SKIPPED: "[-]",
                PlanStatus.CANCELLED: "[/]",
                PlanStatus.REPLANNING: "[R]",
            }.get(step.status, "[?]")

            type_icon = {
                StepType.INTENT: "INT",
                StepType.STEP: "STP",
                StepType.PRIMITIVE: "PRM",
                StepType.CHECKPOINT: "CHK",
                StepType.PARALLEL: "PAR",
                StepType.CONDITIONAL: "CND",
                StepType.SKILL: "SKL",
            }.get(step.step_type, "???")

            skill_info = f" ({step.skill.name})" if step.skill else ""
            lines.append(f"{prefix}{status_icon} {type_icon}{skill_info} {step.description}")

            if step.reasoning:
                lines.append(f"{prefix}  > {step.reasoning[:60]}...")

            for child in step.children:
                render(child, depth + 1)

        render(self.root, indent)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize plan to dictionary."""
        def step_to_dict(step: Step) -> Dict[str, Any]:
            return {
                "id": step.id,
                "type": step.step_type.name,
                "description": step.description,
                "status": step.status.name,
                "skill": step.skill.name if step.skill else None,
                "reasoning": step.reasoning,
                "children": [step_to_dict(c) for c in step.children],
                "primitive": step.primitive.params if step.primitive else None,
                "has_alternatives": step.has_alternatives,
            }

        return {
            "id": self.id,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "progress": self.progress,
            "replan_count": self.replan_count,
            "global_reasoning": self.global_reasoning,
            "root": step_to_dict(self.root),
        }


# =============================================================================
# PLAN BUILDER (with GoalAct support)
# =============================================================================

class PlanBuilder:
    """
    Fluent builder for creating plans with GoalAct support.

    Usage:
        plan = (
            PlanBuilder(goal, state)
            .add_skill_step(Skill.analysis(), "Analyze requirements")
            .add_step("Execute task", [
                Primitive.shell("command")
            ])
            .add_checkpoint("Verify output")
            .with_alternative("Try different approach")
            .build()
        )
    """

    def __init__(self, goal: Goal, state: State, reasoning: str = ""):
        self.goal = goal
        self.state = state
        self.reasoning = reasoning
        self.steps: List[Step] = []
        self.current_step: Optional[Step] = None

    def add_skill_step(
        self,
        skill: Skill,
        description: str = None,
        reasoning: str = "",
    ) -> PlanBuilder:
        """Add a skill-based step (GoalAct pattern)."""
        step = Step.skill_step(skill, description, reasoning)
        self.steps.append(step)
        self.current_step = step
        return self

    def add_step(
        self,
        description: str,
        primitives: List[Primitive] = None,
        reasoning: str = "",
    ) -> PlanBuilder:
        """Add a step with optional primitives."""
        step = Step(
            id=str(uuid.uuid4())[:8],
            step_type=StepType.STEP if primitives else StepType.INTENT,
            description=description,
            reasoning=reasoning,
        )
        if primitives:
            for prim in primitives:
                child = Step.action(prim.description or "Execute", prim)
                step.children.append(child)
        self.steps.append(step)
        self.current_step = step
        return self

    def add_checkpoint(
        self,
        description: str,
        check_type: str = "output_check",
    ) -> PlanBuilder:
        """Add a checkpoint after the current step."""
        if self.current_step:
            checkpoint = Checkpoint(
                id=str(uuid.uuid4())[:8],
                description=description,
                check_type=check_type,
            )
            self.current_step.postconditions.append(checkpoint)
        return self

    def add_parallel_step(
        self,
        description: str,
        primitives: List[Primitive],
        reasoning: str = "",
    ) -> PlanBuilder:
        """Add a parallel execution step."""
        children = [Step.action(p.description or "Execute", p) for p in primitives]
        step = Step.parallel(description, children, reasoning)
        self.steps.append(step)
        self.current_step = step
        return self

    def with_fallback(self, fallback_primitives: List[Primitive]) -> PlanBuilder:
        """Add fallback to current step."""
        if self.current_step:
            children = [
                Step.action(p.description or "Fallback", p)
                for p in fallback_primitives
            ]
            fallback = Step.sequence("Fallback", children, "Fallback strategy")
            self.current_step.fallback = fallback
        return self

    def with_alternative(self, description: str, reasoning: str = "") -> PlanBuilder:
        """Add an alternative approach to current step (GoalAct)."""
        if self.current_step:
            alt = Step.intent(description, reasoning=reasoning or "Alternative approach")
            self.current_step.alternatives.append(alt)
        return self

    def build(self) -> Plan:
        """Build the plan."""
        plan = Plan.create(self.goal, self.state, self.reasoning)

        # Set up sequential dependencies
        for i, step in enumerate(self.steps):
            if i > 0:
                step.depends_on.append(self.steps[i-1].id)

        plan.root.decompose(self.steps, self.reasoning)
        return plan


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

async def create_plan_with_llm(
    goal: Goal,
    state: State,
    reasoner: Reasoner,
) -> Plan:
    """
    Create a plan using LLM-driven global planning (GoalAct pattern).
    """
    planner = GlobalPlanner(reasoner)

    # Create global plan
    steps = await planner.create_global_plan(goal, state)

    # Create plan with steps
    plan = Plan.create(goal, state, "LLM-generated global plan")
    plan.root.decompose(steps, "GoalAct global planning")

    return plan


def create_simple_plan(goal: Goal, state: State, primitives: List[Primitive]) -> Plan:
    """
    Create a simple sequential plan from primitives.
    """
    builder = PlanBuilder(goal, state)
    for prim in primitives:
        builder.add_step(prim.description or "Execute", [prim])
    return builder.build()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from jack.foundation.state import StateBuilder, GoalType, EntityType

    async def test_planning():
        print("=" * 60)
        print("HIERARCHICAL PLAN - GoalAct-Inspired Dynamic Planning")
        print("=" * 60)

        # Create state and goal
        state = (
            StateBuilder()
            .with_goal("Analyze sales and create forecast", GoalType.ANALYZE)
            .add_entity("sales", EntityType.TABLE, {"rows": 50000})
            .build()
        )

        # Test GoalAct-style planning
        print("\n[TEST] GoalAct Global Planning")
        reasoner = SimpleReasoner()
        planner = GlobalPlanner(reasoner)

        steps = await planner.create_global_plan(state.goal, state)
        print(f"  Generated {len(steps)} high-level steps:")
        for step in steps:
            skill_name = step.skill.name if step.skill else "N/A"
            print(f"    - [{skill_name}] {step.description}")

        # Build a plan with skills
        print("\n[TEST] Plan with Skills")
        plan = (
            PlanBuilder(state.goal, state, "Sales analysis plan")
            .add_skill_step(Skill.analysis(), "Analyze historical sales data")
            .add_checkpoint("Data loaded successfully")
            .add_skill_step(Skill.coding(), "Create forecast model")
            .with_alternative("Use simpler moving average if ML fails")
            .add_skill_step(Skill.verification(), "Validate forecast accuracy")
            .build()
        )

        print("\n[PLAN TREE]")
        print(plan.to_tree_string())

        print("\n[PLAN STATS]")
        print(f"  ID: {plan.id}")
        print(f"  Version: {plan.version}")
        print(f"  Total steps: {len(plan.get_all_steps())}")
        print(f"  Progress: {plan.progress:.0%}")
        print(f"  Can replan: {plan.can_replan}")

        # Test replanning
        print("\n[TEST] Dynamic Replanning")
        replanner = Replanner(reasoner, planner)

        failed_step = plan.get_all_steps()[1]  # Second step
        failed_step.mark_failed(Error(ErrorCode.EXECUTION_FAILED, "Model training failed"))

        remaining = plan.get_remaining_steps()[1:]  # Skip failed step
        result = await replanner.replan_on_failure(failed_step, failed_step.error, state, remaining)

        if result.is_ok():
            new_steps = result.unwrap()
            print(f"  Replanning successful! {len(new_steps)} new steps generated")
            for step in new_steps[:3]:
                print(f"    - {step.description}")
        else:
            print(f"  Replanning failed: {result.unwrap_err().message}")

        # Simulate execution with alternatives
        print("\n[TEST] Execution with Alternatives")
        plan2 = create_simple_plan(
            state.goal,
            state,
            [
                Primitive.shell("echo 'Step 1'"),
                Primitive.shell("echo 'Step 2'"),
                Primitive.shell("echo 'Step 3'"),
            ]
        )

        next_step = plan2.get_next_step()
        while next_step:
            print(f"  Executing: {next_step.description}")
            next_step.mark_started()
            next_step.mark_completed(result="success")
            next_step = plan2.get_next_step()

        print(f"\n  Final progress: {plan2.progress:.0%}")
        print(f"  Plan successful: {plan2.is_successful}")

        print("\n" + "=" * 60)
        print("[OK] GoalAct-Inspired Planning Complete")
        print("=" * 60)
        print("\nKey Components:")
        print("  - Skill: High-level reusable skill patterns")
        print("  - GlobalPlanner: LLM-driven global planning")
        print("  - Replanner: Dynamic replanning on failure")
        print("  - Step: Hierarchical step with alternatives")
        print("  - Plan: Complete plan with continuous updates")
        print("=" * 60)

    asyncio.run(test_planning())
