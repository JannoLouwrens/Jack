"""
GOAL-CONDITIONED STATE - Minimal, Relevant State Representation

This module implements state representation that is:
- Goal-conditioned: Only captures what's relevant to the current goal
- Minimal: No unnecessary information
- Structured: Not raw text, but typed data
- Immutable: Frozen dataclasses for safety

Design Philosophy:
- State should answer: "What do I need to know to achieve this goal?"
- Not: "What is everything about the world?"

The state representation follows research insights:
- STAR (2024): Action-relevant state representation
- ETrSR (2024): Task-relevant features only
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    List, Dict, Optional, Any, Set, FrozenSet,
    TypeVar, Generic, Tuple, Protocol, runtime_checkable
)
from enum import Enum, auto
from datetime import datetime
import hashlib
import json


# =============================================================================
# CORE TYPES
# =============================================================================

class GoalType(Enum):
    """Classification of goal types."""
    QUERY = auto()          # Get information
    TRANSFORM = auto()      # Change data
    CREATE = auto()         # Make something new
    DELETE = auto()         # Remove something
    ANALYZE = auto()        # Understand patterns
    PREDICT = auto()        # Forecast future
    MONITOR = auto()        # Watch for conditions
    COMPOSITE = auto()      # Multiple goals


class EntityType(Enum):
    """Types of entities the system can work with."""
    DATABASE = auto()
    TABLE = auto()
    COLUMN = auto()
    FILE = auto()
    DIRECTORY = auto()
    API = auto()
    MODEL = auto()
    CHART = auto()
    TRIGGER = auto()
    UNKNOWN = auto()


class ConstraintType(Enum):
    """Types of constraints on operations."""
    SAFETY = auto()         # Must not violate safety
    PERMISSION = auto()     # Must have permission
    TIMEOUT = auto()        # Must complete in time
    RESOURCE = auto()       # Must not exceed resources
    FORMAT = auto()         # Must match format
    DEPENDENCY = auto()     # Must have dependencies


# =============================================================================
# GOAL
# =============================================================================

@dataclass(frozen=True)
class Goal:
    """
    Represents what the system is trying to achieve.

    A goal is:
    - Explicit: Clearly stated intent
    - Measurable: Has success criteria
    - Decomposable: Can break into sub-goals

    Examples:
        Goal(
            intent="Query total sales by month",
            goal_type=GoalType.QUERY,
            success_criteria=["returns data", "grouped by month"],
            constraints=[Constraint(type=TIMEOUT, value=30)]
        )
    """
    intent: str                                     # Natural language description
    goal_type: GoalType                             # Classification
    success_criteria: Tuple[str, ...] = ()          # What defines success
    output_format: Optional[str] = None             # Expected output format
    constraints: Tuple[Constraint, ...] = ()        # Limitations
    parent_goal: Optional[Goal] = None              # If this is a sub-goal
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_constraint(self, constraint: Constraint) -> Goal:
        """Create new Goal with additional constraint."""
        return Goal(
            intent=self.intent,
            goal_type=self.goal_type,
            success_criteria=self.success_criteria,
            output_format=self.output_format,
            constraints=self.constraints + (constraint,),
            parent_goal=self.parent_goal,
            metadata=self.metadata,
        )

    def as_subgoal(self, intent: str, goal_type: GoalType) -> Goal:
        """Create a sub-goal with this as parent."""
        return Goal(
            intent=intent,
            goal_type=goal_type,
            constraints=self.constraints,  # Inherit constraints
            parent_goal=self,
        )

    def is_satisfied_by(self, outcome: Any) -> bool:
        """Check if outcome satisfies success criteria (basic check)."""
        if outcome is None:
            return False
        if not self.success_criteria:
            return True  # No criteria = any result is fine
        # More sophisticated checking would be done by Verifier
        return True

    @property
    def hierarchy_depth(self) -> int:
        """How deep in the goal hierarchy is this?"""
        depth = 0
        current = self.parent_goal
        while current is not None:
            depth += 1
            current = current.parent_goal
        return depth

    def __hash__(self) -> int:
        return hash((self.intent, self.goal_type))


@dataclass(frozen=True)
class Constraint:
    """
    A constraint on operations.

    Constraints are inherited down the goal hierarchy.
    """
    constraint_type: ConstraintType
    value: Any
    description: Optional[str] = None
    strict: bool = True  # If False, constraint is a preference

    def is_satisfied(self, context: Dict[str, Any]) -> bool:
        """Check if constraint is satisfied given context."""
        # Override in specific constraint types
        return True


# =============================================================================
# ENTITY
# =============================================================================

@dataclass(frozen=True)
class Entity:
    """
    An entity the system can work with.

    Entities are the "nouns" - things we operate on.
    They are goal-conditioned: we only include relevant properties.
    """
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: Tuple[EntityRelation, ...] = ()

    # Relevance to current goal (computed by RelevanceScorer)
    relevance_score: float = 0.0

    def with_relevance(self, score: float) -> Entity:
        """Create entity with updated relevance score."""
        return Entity(
            name=self.name,
            entity_type=self.entity_type,
            properties=self.properties,
            relationships=self.relationships,
            relevance_score=score,
        )

    @property
    def id(self) -> str:
        """Unique identifier for this entity."""
        return f"{self.entity_type.name}:{self.name}"

    def with_property(self, key: str, value: Any) -> Entity:
        """Create new Entity with additional property."""
        new_props = dict(self.properties)
        new_props[key] = value
        return Entity(
            name=self.name,
            entity_type=self.entity_type,
            properties=new_props,
            relationships=self.relationships,
            relevance_score=self.relevance_score,
        )

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class EntityRelation:
    """Relationship between entities."""
    from_entity: str  # Entity ID
    to_entity: str    # Entity ID
    relation_type: str  # e.g., "foreign_key", "contains", "depends_on"
    properties: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CONTEXT
# =============================================================================

@dataclass(frozen=True)
class Observation:
    """
    A single observation from system interaction.

    Observations build up context over time.
    """
    timestamp: datetime
    observation_type: str  # "action", "result", "error", "user_input"
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Context:
    """
    Accumulated context for decision-making.

    Context includes:
    - Recent observations (what just happened)
    - Retrieved patterns (what worked before)
    - User preferences (how they like things)
    """
    observations: Tuple[Observation, ...] = ()
    retrieved_patterns: Tuple[Any, ...] = ()  # From memory
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_data: Dict[str, Any] = field(default_factory=dict)

    def with_observation(self, obs: Observation) -> Context:
        """Add observation, keeping last N."""
        MAX_OBSERVATIONS = 50
        new_obs = (self.observations + (obs,))[-MAX_OBSERVATIONS:]
        return Context(
            observations=new_obs,
            retrieved_patterns=self.retrieved_patterns,
            user_preferences=self.user_preferences,
            session_data=self.session_data,
        )

    def recent_observations(self, n: int = 10) -> Tuple[Observation, ...]:
        """Get most recent N observations."""
        return self.observations[-n:]

    @property
    def last_action(self) -> Optional[Observation]:
        """Get most recent action observation."""
        for obs in reversed(self.observations):
            if obs.observation_type == "action":
                return obs
        return None

    @property
    def last_result(self) -> Optional[Observation]:
        """Get most recent result observation."""
        for obs in reversed(self.observations):
            if obs.observation_type == "result":
                return obs
        return None


# =============================================================================
# STATE
# =============================================================================

@dataclass(frozen=True)
class State:
    """
    Goal-conditioned state representation.

    This is the complete picture of "where we are" relative to "where we want to be."

    Properties:
    - goal: What we're trying to achieve
    - entities: Relevant things we can work with
    - context: Recent history and retrieved knowledge
    - constraints: What we must respect

    The state is IMMUTABLE. Operations return new State objects.
    This makes reasoning about state changes explicit and safe.
    """
    goal: Goal
    entities: FrozenSet[Entity] = frozenset()
    context: Context = field(default_factory=Context)
    constraints: Tuple[Constraint, ...] = ()

    # Computed fields (for caching, not part of equality)
    _hash: Optional[int] = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        # Merge goal constraints with state constraints
        all_constraints = self.constraints + self.goal.constraints
        if all_constraints != self.constraints:
            object.__setattr__(self, 'constraints', all_constraints)

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name."""
        for entity in self.entities:
            if entity.name == name:
                return entity
        return None

    def get_entities_by_type(self, entity_type: EntityType) -> FrozenSet[Entity]:
        """Get all entities of a type."""
        return frozenset(e for e in self.entities if e.entity_type == entity_type)

    def get_related_entities(self, entity: Entity) -> FrozenSet[Entity]:
        """Get entities related to given entity."""
        related_ids = set()
        for rel in entity.relationships:
            related_ids.add(rel.to_entity)
        return frozenset(
            e for e in self.entities
            if e.id in related_ids
        )

    @property
    def tables(self) -> FrozenSet[Entity]:
        """Convenience: get all table entities."""
        return self.get_entities_by_type(EntityType.TABLE)

    @property
    def files(self) -> FrozenSet[Entity]:
        """Convenience: get all file entities."""
        return self.get_entities_by_type(EntityType.FILE)

    def has_entity(self, name: str) -> bool:
        """Check if entity exists."""
        return any(e.name == name for e in self.entities)

    def is_sufficient(self) -> bool:
        """
        Do we have enough information to proceed?

        This is a heuristic check. The Verifier does more thorough checking.
        """
        # Must have a goal
        if not self.goal:
            return False
        # Must have at least one relevant entity for most goals
        if self.goal.goal_type != GoalType.CREATE and not self.entities:
            return False
        return True

    # =========================================================================
    # UPDATES (return new State)
    # =========================================================================

    def with_entity(self, entity: Entity) -> State:
        """Add or update an entity."""
        # Remove old version if exists
        new_entities = frozenset(
            e for e in self.entities if e.name != entity.name
        ) | {entity}
        return State(
            goal=self.goal,
            entities=new_entities,
            context=self.context,
            constraints=self.constraints,
        )

    def with_entities(self, entities: List[Entity]) -> State:
        """Add multiple entities."""
        result = self
        for entity in entities:
            result = result.with_entity(entity)
        return result

    def with_observation(self, obs: Observation) -> State:
        """Add an observation to context."""
        return State(
            goal=self.goal,
            entities=self.entities,
            context=self.context.with_observation(obs),
            constraints=self.constraints,
        )

    def with_goal(self, goal: Goal) -> State:
        """Change the goal (for sub-task execution)."""
        return State(
            goal=goal,
            entities=self.entities,
            context=self.context,
            constraints=self.constraints,
        )

    def with_constraint(self, constraint: Constraint) -> State:
        """Add a constraint."""
        return State(
            goal=self.goal,
            entities=self.entities,
            context=self.context,
            constraints=self.constraints + (constraint,),
        )

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "goal": {
                "intent": self.goal.intent,
                "type": self.goal.goal_type.name,
                "success_criteria": list(self.goal.success_criteria),
            },
            "entities": [
                {
                    "name": e.name,
                    "type": e.entity_type.name,
                    "properties": e.properties,
                }
                for e in self.entities
            ],
            "context": {
                "observation_count": len(self.context.observations),
                "pattern_count": len(self.context.retrieved_patterns),
            },
            "constraints": [
                {
                    "type": c.constraint_type.name,
                    "value": str(c.value),
                }
                for c in self.constraints
            ],
        }

    def to_prompt_context(self) -> str:
        """Convert to context string for LLM prompts."""
        lines = [
            f"GOAL: {self.goal.intent}",
            f"TYPE: {self.goal.goal_type.name}",
            "",
            "AVAILABLE ENTITIES:",
        ]
        for entity in sorted(self.entities, key=lambda e: e.relevance_score, reverse=True):
            props = ", ".join(f"{k}={v}" for k, v in list(entity.properties.items())[:5])
            lines.append(f"  - {entity.name} ({entity.entity_type.name}): {props}")

        if self.constraints:
            lines.append("")
            lines.append("CONSTRAINTS:")
            for c in self.constraints:
                lines.append(f"  - {c.constraint_type.name}: {c.value}")

        if self.context.observations:
            lines.append("")
            lines.append("RECENT CONTEXT:")
            for obs in self.context.recent_observations(5):
                lines.append(f"  - [{obs.observation_type}] {str(obs.content)[:100]}")

        return "\n".join(lines)

    @property
    def fingerprint(self) -> str:
        """Unique fingerprint for this state (for caching)."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def __hash__(self) -> int:
        if self._hash is None:
            object.__setattr__(self, '_hash', hash((self.goal, self.entities)))
        return self._hash


# =============================================================================
# STATE BUILDER
# =============================================================================

class StateBuilder:
    """
    Fluent builder for constructing State objects.

    Usage:
        state = (
            StateBuilder()
            .with_goal("Query sales data", GoalType.QUERY)
            .add_entity("sales", EntityType.TABLE, {"row_count": 1000})
            .add_constraint(ConstraintType.TIMEOUT, 30)
            .build()
        )
    """

    def __init__(self):
        self._goal: Optional[Goal] = None
        self._entities: List[Entity] = []
        self._context: Context = Context()
        self._constraints: List[Constraint] = []

    def with_goal(
        self,
        intent: str,
        goal_type: GoalType,
        success_criteria: List[str] = None,
        output_format: str = None,
    ) -> StateBuilder:
        """Set the goal."""
        self._goal = Goal(
            intent=intent,
            goal_type=goal_type,
            success_criteria=tuple(success_criteria or []),
            output_format=output_format,
        )
        return self

    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        properties: Dict[str, Any] = None,
        relevance: float = 0.0,
    ) -> StateBuilder:
        """Add an entity."""
        self._entities.append(Entity(
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            relevance_score=relevance,
        ))
        return self

    def add_constraint(
        self,
        constraint_type: ConstraintType,
        value: Any,
        description: str = None,
    ) -> StateBuilder:
        """Add a constraint."""
        self._constraints.append(Constraint(
            constraint_type=constraint_type,
            value=value,
            description=description,
        ))
        return self

    def with_context(self, context: Context) -> StateBuilder:
        """Set context."""
        self._context = context
        return self

    def build(self) -> State:
        """Build the State object."""
        if self._goal is None:
            raise ValueError("Goal is required")
        return State(
            goal=self._goal,
            entities=frozenset(self._entities),
            context=self._context,
            constraints=tuple(self._constraints),
        )


# =============================================================================
# RELEVANCE SCORER (LLM-First Entity Relevance)
# =============================================================================

@runtime_checkable
class Reasoner(Protocol):
    """Protocol for LLM reasoning."""

    async def reason(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Make a reasoning call to the LLM."""
        ...


class RelevanceScorer:
    """
    LLM-based relevance scoring for entities.

    Scores how relevant each entity is to the current goal.
    Used to:
    - Filter entities for prompt context
    - Prioritize entity processing
    - Focus attention on what matters

    This replaces naive keyword matching with LLM understanding.
    """

    def __init__(self, reasoner: Reasoner):
        self.reasoner = reasoner

    async def score_entity(
        self,
        entity: Entity,
        goal: Goal,
    ) -> float:
        """
        Score how relevant an entity is to a goal.

        Returns 0.0 to 1.0 where:
        - 1.0 = Essential for achieving the goal
        - 0.7+ = Highly relevant
        - 0.4-0.7 = Potentially useful
        - <0.4 = Probably not needed
        """
        prompt = f"""Entity Relevance Scoring

Score how relevant this entity is for achieving the goal.

GOAL:
Intent: {goal.intent}
Type: {goal.goal_type.name}

ENTITY:
Name: {entity.name}
Type: {entity.entity_type.name}
Properties: {entity.properties}

Consider:
1. Is this entity directly mentioned or implied in the goal?
2. Would this entity be needed to complete the task?
3. Does it contain information relevant to the goal?
4. Could the goal be achieved without this entity?

Respond with ONLY a JSON object:
{{"relevance": 0.0 to 1.0, "reason": "brief explanation"}}"""

        try:
            response = await self.reasoner.reason(prompt)
            import json
            parsed = json.loads(response)
            return max(0.0, min(1.0, parsed.get("relevance", 0.5)))
        except Exception:
            # Fallback: simple heuristic
            return self._heuristic_score(entity, goal)

    def _heuristic_score(self, entity: Entity, goal: Goal) -> float:
        """Fallback heuristic scoring (not LLM-first, for testing)."""
        intent_lower = goal.intent.lower()
        name_lower = entity.name.lower()

        # Direct mention = high relevance
        if name_lower in intent_lower:
            return 0.9

        # Type matching
        type_scores = {
            GoalType.QUERY: {EntityType.TABLE: 0.8, EntityType.DATABASE: 0.7, EntityType.API: 0.6},
            GoalType.ANALYZE: {EntityType.TABLE: 0.9, EntityType.FILE: 0.7, EntityType.CHART: 0.5},
            GoalType.CREATE: {EntityType.DIRECTORY: 0.7, EntityType.FILE: 0.6, EntityType.MODEL: 0.6},
            GoalType.TRANSFORM: {EntityType.TABLE: 0.8, EntityType.FILE: 0.7, EntityType.COLUMN: 0.6},
        }

        if goal.goal_type in type_scores:
            entity_scores = type_scores[goal.goal_type]
            if entity.entity_type in entity_scores:
                return entity_scores[entity.entity_type]

        return 0.3  # Default low relevance

    async def score_entities(
        self,
        entities: FrozenSet[Entity],
        goal: Goal,
    ) -> List[Entity]:
        """
        Score all entities and return sorted by relevance (highest first).
        """
        scored = []
        for entity in entities:
            score = await self.score_entity(entity, goal)
            scored.append(entity.with_relevance(score))

        return sorted(scored, key=lambda e: e.relevance_score, reverse=True)

    async def filter_relevant(
        self,
        entities: FrozenSet[Entity],
        goal: Goal,
        min_relevance: float = 0.3,
        max_entities: int = 20,
    ) -> List[Entity]:
        """
        Filter and return only relevant entities.

        Useful for building prompt context without overwhelming the LLM.
        """
        scored = await self.score_entities(entities, goal)
        return [e for e in scored if e.relevance_score >= min_relevance][:max_entities]


class SimpleRelevanceScorer:
    """Simple relevance scorer for testing (no LLM required)."""

    def _heuristic_score(self, entity: Entity, goal: Goal) -> float:
        """Simple heuristic scoring."""
        intent_lower = goal.intent.lower()
        name_lower = entity.name.lower()

        if name_lower in intent_lower:
            return 0.9

        # Keyword matching
        keywords = intent_lower.split()
        matches = sum(1 for kw in keywords if kw in name_lower)
        if matches > 0:
            return min(0.8, 0.3 + 0.2 * matches)

        return 0.3

    async def score_entity(self, entity: Entity, goal: Goal) -> float:
        return self._heuristic_score(entity, goal)

    async def score_entities(self, entities: FrozenSet[Entity], goal: Goal) -> List[Entity]:
        scored = []
        for entity in entities:
            score = await self.score_entity(entity, goal)
            scored.append(entity.with_relevance(score))
        return sorted(scored, key=lambda e: e.relevance_score, reverse=True)

    async def filter_relevant(
        self,
        entities: FrozenSet[Entity],
        goal: Goal,
        min_relevance: float = 0.3,
        max_entities: int = 20,
    ) -> List[Entity]:
        scored = await self.score_entities(entities, goal)
        return [e for e in scored if e.relevance_score >= min_relevance][:max_entities]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GOAL-CONDITIONED STATE")
    print("=" * 60)

    # Build a state using the builder
    state = (
        StateBuilder()
        .with_goal(
            intent="Analyze monthly sales trends and predict next month",
            goal_type=GoalType.ANALYZE,
            success_criteria=["returns trend data", "includes prediction"],
        )
        .add_entity("sales", EntityType.TABLE, {
            "row_count": 50000,
            "columns": ["date", "amount", "product_id"],
        }, relevance=1.0)
        .add_entity("products", EntityType.TABLE, {
            "row_count": 100,
            "columns": ["id", "name", "category"],
        }, relevance=0.8)
        .add_constraint(ConstraintType.TIMEOUT, 60, "Must complete within 60 seconds")
        .add_constraint(ConstraintType.SAFETY, "read_only", "Database is read-only")
        .build()
    )

    print("\n[STATE SUMMARY]")
    print(f"  Goal: {state.goal.intent}")
    print(f"  Type: {state.goal.goal_type.name}")
    print(f"  Entities: {len(state.entities)}")
    print(f"  Constraints: {len(state.constraints)}")
    print(f"  Sufficient: {state.is_sufficient()}")
    print(f"  Fingerprint: {state.fingerprint}")

    print("\n[PROMPT CONTEXT]")
    print(state.to_prompt_context())

    # Test immutability
    print("\n[TEST IMMUTABILITY]")
    new_state = state.with_entity(Entity(
        name="customers",
        entity_type=EntityType.TABLE,
        properties={"row_count": 500},
    ))
    print(f"  Original entities: {len(state.entities)}")
    print(f"  New entities: {len(new_state.entities)}")

    # Test observation
    print("\n[TEST OBSERVATION]")
    obs = Observation(
        timestamp=datetime.now(),
        observation_type="action",
        content="Executed query: SELECT * FROM sales",
    )
    state_with_obs = state.with_observation(obs)
    print(f"  Original observations: {len(state.context.observations)}")
    print(f"  New observations: {len(state_with_obs.context.observations)}")

    print("\n" + "=" * 60)
    print("[OK] Goal-conditioned state working")
    print("=" * 60)
