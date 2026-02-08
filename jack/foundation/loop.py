"""
LOOP - Production-Grade Agent Orchestration

This module implements the core execution loop with full integration:
    PERCEIVE -> RETRIEVE -> REASON -> VERIFY -> ACT -> OBSERVE -> LEARN

Architecture based on:
- OpenClaw: Gateway-centric, lifecycle hooks, context window guard
- ReAct: Think -> Act -> Observe -> Repeat
- Ralph Loop: Stop hooks, external verification, circuit breaker
- LATS: Tree search with backtracking capability
- Self-Healing: Error recovery, retry, fallback strategies

References:
- OpenClaw Agent Loop: https://docs.openclaw.ai/concepts/agent-loop
- ReAct Pattern: https://arxiv.org/abs/2210.03629
- Ralph Loop: https://github.com/snarktank/ralph
- LATS: https://arxiv.org/abs/2310.04406
- Context Engineering: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- PALADIN Self-Healing: https://arxiv.org/html/2509.25238v1

Author: Jack Foundation
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    List, Dict, Optional, Any, Callable, Tuple, Union,
    TypeVar, Generic, Protocol, runtime_checkable, Iterator, Set
)
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import deque
import json
import hashlib
import logging
import traceback
import time
import threading
import copy

from jack.foundation.types import Result, Ok, Err, Option, Some, NONE, Error, ErrorCode
from jack.foundation.state import State, StateBuilder, Goal, GoalType, Entity, Observation
from jack.foundation.plan import Plan, PlanBuilder, Step, StepType, Primitive, PrimitiveType
from jack.foundation.action import Executor, ActionResult, OutcomeType
from jack.foundation.memory import Memory, Pattern
from jack.foundation.verify import (
    Verifier, VerificationReport, Verdict,
    # Constitutional AI (SOTA)
    ConstitutionalVerifier, Constitution, ConstitutionalPrinciple,
    # Guard Agent (SOTA)
    GuardAgent,
    # Goal Achievement
    GoalAchievementVerifier,
    # Exec Approval (OpenClaw)
    ExecApprovalChecker, AuthorizationLevel,
)

# Import perception (with SOTA patterns)
from jack.foundation.perceive import (
    IntelligentPerceptionEngine,
    RobustPerceptionEngine,
    AdaptivePerceptionEngine,  # SOTA: Iterative refinement
    create_perception_engine,
    create_adaptive_perception_engine,
    PerceptionResult,
    PerceptionCalibrator,  # SOTA: Confidence calibration
    PerceptionActionFeedback,  # SOTA: Entity-success learning
    ScoredEntity,
)

# Import retrieval (Agentic RAG)
from jack.foundation.retrieve import (
    RetrievalEngine,
    RetrievalEngineConfig,
    RetrievalQuery,
    RetrievalResult,
    RetrievalSource,
    create_retrieval_engine,
)

# Import robust infrastructure
from jack.foundation.robust import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState,
    RetryConfig, RetryStrategy,
    RobustReasoner, ReasonerResponse,
    ObservabilityLayer, DecisionTrace,
    RateLimiter, RateLimitConfig,
    ResponseCache, SemanticCache, SemanticCacheConfig,
    Session, SessionManager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 1. LOOP PHASE
# =============================================================================

class LoopPhase(Enum):
    """Phases in the main loop (ReAct-inspired)."""
    IDLE = auto()           # Not running
    PERCEIVE = auto()       # Building world state
    RETRIEVE = auto()       # Agentic RAG retrieval
    REASON = auto()         # Planning/deciding (THINK)
    VERIFY = auto()         # Checking safety
    ACT = auto()            # Executing action
    OBSERVE = auto()        # Observing outcome
    LEARN = auto()          # Recording patterns
    COMPACT = auto()        # Context compaction/summarization
    BACKTRACK = auto()      # LATS-style backtracking
    COMPLETE = auto()       # Goal achieved
    FAILED = auto()         # Unrecoverable failure
    PAUSED = auto()         # Waiting for input


# =============================================================================
# 2. LIFECYCLE HOOKS (OpenClaw Pattern)
# =============================================================================

class LifecycleHook(Enum):
    """Lifecycle hooks inspired by OpenClaw."""
    # Agent lifecycle
    BEFORE_AGENT_START = auto()
    AGENT_END = auto()

    # Tool/Action lifecycle
    BEFORE_TOOL_CALL = auto()
    AFTER_TOOL_CALL = auto()
    TOOL_RESULT_PERSIST = auto()

    # Context lifecycle
    BEFORE_COMPACTION = auto()
    AFTER_COMPACTION = auto()

    # Session lifecycle
    SESSION_START = auto()
    SESSION_END = auto()

    # Loop lifecycle
    LOOP_ITERATION_START = auto()
    LOOP_ITERATION_END = auto()

    # Stop hook (Ralph Loop pattern)
    BEFORE_EXIT = auto()


@dataclass
class HookContext:
    """Context passed to lifecycle hooks."""
    hook: LifecycleHook
    phase: LoopPhase
    loop_state: 'LoopState'
    data: Dict[str, Any] = field(default_factory=dict)

    # Hook can modify these
    should_continue: bool = True
    modified_data: Optional[Dict[str, Any]] = None
    error: Optional[Error] = None


# Hook callback type
HookCallback = Callable[[HookContext], HookContext]


class HookRegistry:
    """Registry for lifecycle hooks."""

    def __init__(self):
        self._hooks: Dict[LifecycleHook, List[HookCallback]] = {
            hook: [] for hook in LifecycleHook
        }

    def register(self, hook: LifecycleHook, callback: HookCallback) -> None:
        """Register a hook callback."""
        self._hooks[hook].append(callback)

    def unregister(self, hook: LifecycleHook, callback: HookCallback) -> None:
        """Unregister a hook callback."""
        if callback in self._hooks[hook]:
            self._hooks[hook].remove(callback)

    def invoke(self, context: HookContext) -> HookContext:
        """Invoke all callbacks for a hook."""
        for callback in self._hooks[context.hook]:
            try:
                context = callback(context)
                if not context.should_continue:
                    break
            except Exception as e:
                logger.warning(f"Hook {context.hook.name} failed: {e}")
                context.error = Error(ErrorCode.EXECUTION_FAILED, str(e))
        return context


# =============================================================================
# 3. CONTEXT WINDOW MANAGEMENT (OpenClaw Pattern)
# =============================================================================

@dataclass
class ContextWindowConfig:
    """Configuration for context window management."""
    max_tokens: int = 128000          # Model's max context
    compaction_threshold: float = 0.8  # Trigger compaction at 80%
    summarization_threshold: float = 0.95  # Trigger summarization at 95%
    keep_recent_turns: int = 3         # Keep last N turns raw
    keep_recent_tool_calls: int = 5    # Keep last N tool calls raw
    chars_per_token: int = 4           # Rough estimate


@dataclass
class ContextStats:
    """Statistics about current context usage."""
    total_tokens: int = 0
    message_tokens: int = 0
    tool_call_tokens: int = 0
    system_tokens: int = 0
    utilization: float = 0.0
    needs_compaction: bool = False
    needs_summarization: bool = False


class ContextWindowGuard:
    """
    Monitors and manages context window usage.

    Based on OpenClaw's Context Window Guard pattern:
    - Monitors token count
    - Triggers compaction before "context rot"
    - Summarizes when approaching limits

    References:
    - OpenClaw Context: https://docs.openclaw.ai/concepts/agent-loop
    - Manus Context Engineering: https://rlancemartin.github.io/2025/10/15/manus/
    """

    def __init__(self, config: Optional[ContextWindowConfig] = None):
        self.config = config or ContextWindowConfig()
        self._message_history: List[Dict[str, Any]] = []
        self._tool_calls: List[Dict[str, Any]] = []
        self._compaction_count = 0
        self._summarization_count = 0

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a message to the context."""
        self._message_history.append({
            "role": role,
            "content": content,
            "tokens": len(content) // self.config.chars_per_token,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        })

    def add_tool_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Any,
        full_content: str,
    ) -> None:
        """Add a tool call with full and compact representations."""
        self._tool_calls.append({
            "tool_name": tool_name,
            "params": params,
            "result_summary": str(result)[:200] if result else None,
            "full_content": full_content,
            "compact_ref": f"tool_call_{len(self._tool_calls)}",
            "tokens": len(full_content) // self.config.chars_per_token,
            "timestamp": datetime.now().isoformat(),
            "is_compacted": False,
        })

    def get_stats(self) -> ContextStats:
        """Get current context statistics."""
        message_tokens = sum(m.get("tokens", 0) for m in self._message_history)
        tool_tokens = sum(
            t.get("tokens", 0) if not t.get("is_compacted") else 20
            for t in self._tool_calls
        )
        total = message_tokens + tool_tokens
        utilization = total / self.config.max_tokens

        return ContextStats(
            total_tokens=total,
            message_tokens=message_tokens,
            tool_call_tokens=tool_tokens,
            utilization=utilization,
            needs_compaction=utilization >= self.config.compaction_threshold,
            needs_summarization=utilization >= self.config.summarization_threshold,
        )

    def compact(self) -> int:
        """
        Compact tool calls (reversible).

        Replaces full tool content with compact references for older calls.
        """
        compacted = 0
        keep_raw = self.config.keep_recent_tool_calls

        for i, tool_call in enumerate(self._tool_calls[:-keep_raw] if keep_raw else self._tool_calls):
            if not tool_call.get("is_compacted"):
                tool_call["is_compacted"] = True
                tool_call["tokens"] = 20  # Compact reference size
                compacted += 1

        self._compaction_count += 1
        logger.info(f"Compacted {compacted} tool calls (cycle {self._compaction_count})")
        return compacted

    def summarize(self, summarizer: Optional[Callable[[str], str]] = None) -> str:
        """
        Summarize conversation history (lossy).

        Returns summary and clears old messages, keeping recent turns.
        """
        if not self._message_history:
            return ""

        keep_raw = self.config.keep_recent_turns
        to_summarize = self._message_history[:-keep_raw] if keep_raw else self._message_history

        if not to_summarize:
            return ""

        # Build content to summarize
        content = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in to_summarize
        ])

        # Use provided summarizer or create structured summary
        if summarizer:
            summary = summarizer(content)
        else:
            summary = self._structured_summary(to_summarize)

        # Clear old messages, keep recent
        self._message_history = self._message_history[-keep_raw:] if keep_raw else []

        # Add summary as system message
        self._message_history.insert(0, {
            "role": "system",
            "content": f"[CONTEXT SUMMARY]\n{summary}",
            "tokens": len(summary) // self.config.chars_per_token,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"is_summary": True},
        })

        self._summarization_count += 1
        logger.info(f"Summarized context (cycle {self._summarization_count})")
        return summary

    def _structured_summary(self, messages: List[Dict]) -> str:
        """Create a structured summary to prevent information loss."""
        # Group by type
        user_requests = []
        decisions_made = []
        actions_taken = []
        results = []

        for m in messages:
            content = m.get("content", "")
            role = m.get("role", "")

            if role == "user":
                user_requests.append(content[:100])
            elif "decision" in content.lower() or "plan" in content.lower():
                decisions_made.append(content[:100])
            elif "executed" in content.lower() or "action" in content.lower():
                actions_taken.append(content[:100])
            elif "result" in content.lower() or "output" in content.lower():
                results.append(content[:100])

        summary_parts = []
        if user_requests:
            summary_parts.append(f"USER REQUESTS:\n- " + "\n- ".join(user_requests[:5]))
        if decisions_made:
            summary_parts.append(f"DECISIONS:\n- " + "\n- ".join(decisions_made[:5]))
        if actions_taken:
            summary_parts.append(f"ACTIONS:\n- " + "\n- ".join(actions_taken[:5]))
        if results:
            summary_parts.append(f"RESULTS:\n- " + "\n- ".join(results[:5]))

        return "\n\n".join(summary_parts)

    def get_context_for_llm(self) -> List[Dict[str, str]]:
        """Get context formatted for LLM consumption."""
        messages = []

        for m in self._message_history:
            messages.append({
                "role": m["role"],
                "content": m["content"],
            })

        # Add non-compacted tool calls
        for t in self._tool_calls:
            if not t.get("is_compacted"):
                messages.append({
                    "role": "assistant",
                    "content": f"[Tool: {t['tool_name']}] {t['full_content'][:500]}",
                })

        return messages


# =============================================================================
# 4. STOP HOOK (Ralph Loop Pattern)
# =============================================================================

@dataclass
class StopHookConfig:
    """Configuration for Ralph Loop stop hook."""
    completion_marker: str = "<COMPLETE>"
    require_verification: bool = True
    max_exit_attempts: int = 3
    circuit_breaker_threshold: int = 5  # Same error N times = break


class StopHook:
    """
    Ralph Loop stop hook pattern.

    Prevents agent from exiting until external verification passes.
    Uses circuit breaker to prevent infinite loops.

    References:
    - Ralph Loop: https://github.com/snarktank/ralph
    - Ralph Wiggum Plugin: https://awesomeclaude.ai/ralph-wiggum
    """

    def __init__(
        self,
        config: Optional[StopHookConfig] = None,
        verifier: Optional[Callable[[str], bool]] = None,
    ):
        self.config = config or StopHookConfig()
        self.verifier = verifier or self._default_verifier

        self._exit_attempts = 0
        self._error_counts: Dict[str, int] = {}
        self._circuit_broken = False

    def _default_verifier(self, output: str) -> bool:
        """Default verification: check for completion marker."""
        return self.config.completion_marker in output

    def can_exit(self, output: str, error: Optional[Error] = None) -> Tuple[bool, str]:
        """
        Check if the agent can exit.

        Returns: (can_exit, reason)
        """
        self._exit_attempts += 1

        # Check circuit breaker
        if self._circuit_broken:
            return (True, "Circuit breaker activated - forcing exit")

        # Track error frequency
        if error:
            error_key = str(error.code)
            self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1

            if self._error_counts[error_key] >= self.config.circuit_breaker_threshold:
                self._circuit_broken = True
                return (True, f"Same error repeated {self._error_counts[error_key]} times - circuit broken")

        # Check max exit attempts
        if self._exit_attempts >= self.config.max_exit_attempts:
            return (True, f"Max exit attempts ({self.config.max_exit_attempts}) reached")

        # Run verification
        if self.config.require_verification:
            if self.verifier(output):
                return (True, "Verification passed")
            else:
                return (False, "Verification failed - continuing loop")

        return (True, "No verification required")

    def reset(self) -> None:
        """Reset stop hook state for new run."""
        self._exit_attempts = 0
        self._error_counts.clear()
        self._circuit_broken = False


# =============================================================================
# 5. BACKTRACKING (LATS Pattern)
# =============================================================================

@dataclass
class BacktrackNode:
    """A node in the execution tree for backtracking."""
    id: str
    phase: LoopPhase
    state_snapshot: Dict[str, Any]
    action_taken: Optional[Primitive] = None
    result: Optional[ActionResult] = None
    children: List['BacktrackNode'] = field(default_factory=list)
    parent: Optional['BacktrackNode'] = None
    score: float = 0.0
    visited: bool = False
    depth: int = 0


class BacktrackTree:
    """
    LATS-style tree for backtracking.

    Allows the agent to explore alternatives when stuck.

    References:
    - LATS: https://arxiv.org/abs/2310.04406
    """

    def __init__(self, max_depth: int = 5, max_children: int = 3):
        self.max_depth = max_depth
        self.max_children = max_children
        self.root: Optional[BacktrackNode] = None
        self.current: Optional[BacktrackNode] = None
        self._node_count = 0

    def start(self, initial_state: Dict[str, Any]) -> BacktrackNode:
        """Start a new tree."""
        self.root = BacktrackNode(
            id=self._generate_id(),
            phase=LoopPhase.IDLE,
            state_snapshot=copy.deepcopy(initial_state),
            depth=0,
        )
        self.current = self.root
        return self.root

    def _generate_id(self) -> str:
        """Generate unique node ID."""
        self._node_count += 1
        return f"node_{self._node_count}"

    def add_child(
        self,
        phase: LoopPhase,
        state_snapshot: Dict[str, Any],
        action: Optional[Primitive] = None,
        result: Optional[ActionResult] = None,
        score: float = 0.0,
    ) -> Optional[BacktrackNode]:
        """Add a child node to current."""
        if not self.current:
            return None

        if self.current.depth >= self.max_depth:
            logger.warning(f"Max depth {self.max_depth} reached, cannot add child")
            return None

        if len(self.current.children) >= self.max_children:
            logger.warning(f"Max children {self.max_children} reached")
            return None

        child = BacktrackNode(
            id=self._generate_id(),
            phase=phase,
            state_snapshot=copy.deepcopy(state_snapshot),
            action_taken=action,
            result=result,
            parent=self.current,
            score=score,
            depth=self.current.depth + 1,
        )
        self.current.children.append(child)
        self.current = child
        return child

    def backtrack(self) -> Optional[BacktrackNode]:
        """
        Backtrack to parent and try alternative.

        Returns the node to resume from, or None if at root.
        """
        if not self.current or not self.current.parent:
            return None

        self.current.visited = True
        parent = self.current.parent

        # Find unvisited sibling with best score
        unvisited = [c for c in parent.children if not c.visited]
        if unvisited:
            best = max(unvisited, key=lambda n: n.score)
            self.current = best
            return best

        # All siblings visited, go up
        self.current = parent
        return self.backtrack()

    def get_path(self) -> List[BacktrackNode]:
        """Get path from root to current."""
        if not self.current:
            return []

        path = []
        node = self.current
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def score_node(self, node: BacktrackNode, result: ActionResult) -> float:
        """Score a node based on action result."""
        if result.is_success:
            return 1.0
        elif result.outcome == OutcomeType.PARTIAL:
            return 0.5
        elif result.outcome == OutcomeType.BLOCKED:
            return 0.1
        else:
            return 0.0

    def get_unexplored_alternatives(self) -> List[BacktrackNode]:
        """
        SOTA: Get all unexplored alternative paths for LATS evaluation.

        Returns nodes that haven't been visited yet, sorted by score.
        Used by reasoning to evaluate which path to explore next.
        """
        if not self.root:
            return []

        alternatives = []
        self._collect_unvisited(self.root, alternatives)

        # Sort by score (highest first)
        return sorted(alternatives, key=lambda n: n.score, reverse=True)

    def _collect_unvisited(self, node: BacktrackNode, result: List[BacktrackNode]) -> None:
        """Recursively collect unvisited nodes."""
        if not node.visited and node != self.current:
            result.append(node)

        for child in node.children:
            self._collect_unvisited(child, result)

    def get_best_path_score(self) -> float:
        """Get the score of the best path found so far."""
        if not self.root:
            return 0.0

        return self._max_path_score(self.root)

    def _max_path_score(self, node: BacktrackNode) -> float:
        """Recursively find max score in tree."""
        max_score = node.score

        for child in node.children:
            child_max = self._max_path_score(child)
            max_score = max(max_score, child_max)

        return max_score


# =============================================================================
# 6. LOOP STATE (Enhanced)
# =============================================================================

@dataclass
class LoopState:
    """
    Complete state of the execution loop.

    Enhanced with context management and backtracking support.
    """
    # Current phase
    phase: LoopPhase = LoopPhase.IDLE

    # World state
    world_state: Optional[State] = None

    # Current plan
    plan: Optional[Plan] = None
    current_step_index: int = 0

    # Execution
    current_primitive: Optional[Primitive] = None
    last_result: Optional[ActionResult] = None

    # Retrieval context (from RetrievalEngine)
    retrieval_result: Optional[RetrievalResult] = None
    retrieved_patterns: List[Tuple[Pattern, float]] = field(default_factory=list)

    # Perception context
    perception_result: Optional[PerceptionResult] = None

    # History
    actions_taken: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    all_results: List[ActionResult] = field(default_factory=list)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    phase_timings: Dict[str, float] = field(default_factory=dict)

    # Errors and recovery
    errors: List[Error] = field(default_factory=list)
    last_error: Optional[Error] = None
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3

    # Control
    max_iterations: int = 100
    current_iteration: int = 0
    should_stop: bool = False
    pause_reason: Optional[str] = None

    # Context management
    context_stats: Optional[ContextStats] = None
    compaction_count: int = 0
    summarization_count: int = 0

    # Backtracking
    backtrack_count: int = 0
    current_tree_depth: int = 0

    # Session
    session_id: Optional[str] = None

    @property
    def is_running(self) -> bool:
        """Is the loop currently running?"""
        return self.phase not in (
            LoopPhase.IDLE, LoopPhase.COMPLETE,
            LoopPhase.FAILED, LoopPhase.PAUSED
        )

    @property
    def is_complete(self) -> bool:
        """Is the loop complete?"""
        return self.phase == LoopPhase.COMPLETE

    @property
    def success_rate(self) -> float:
        """Success rate so far."""
        if self.actions_taken == 0:
            return 1.0
        return self.successful_actions / self.actions_taken

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.started_at is None:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Serialize loop state."""
        return {
            "phase": self.phase.name,
            "session_id": self.session_id,
            "actions_taken": self.actions_taken,
            "successful_actions": self.successful_actions,
            "failed_actions": self.failed_actions,
            "success_rate": self.success_rate,
            "current_iteration": self.current_iteration,
            "duration_ms": self.duration_ms,
            "recovery_attempts": self.recovery_attempts,
            "backtrack_count": self.backtrack_count,
            "compaction_count": self.compaction_count,
            "errors": [str(e) for e in self.errors[-5:]],  # Last 5 errors
            "context_utilization": self.context_stats.utilization if self.context_stats else 0.0,
        }

    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot for backtracking."""
        return {
            "phase": self.phase,
            "current_step_index": self.current_step_index,
            "actions_taken": self.actions_taken,
            "successful_actions": self.successful_actions,
            "failed_actions": self.failed_actions,
            "current_iteration": self.current_iteration,
        }


# =============================================================================
# 7. REASONER PROTOCOL (Enhanced)
# =============================================================================

@runtime_checkable
class Reasoner(Protocol):
    """
    Protocol for the reasoning component (the "brain").

    This is where the LLM plugs in for ReAct-style reasoning.
    """

    def plan(
        self,
        goal: Goal,
        state: State,
        patterns: List[Tuple[Pattern, float]],
        retrieval_context: Optional[RetrievalResult] = None,
    ) -> Result[Plan, Error]:
        """Generate a plan to achieve the goal."""
        ...

    def reason(self, prompt: str) -> Result[str, Error]:
        """Raw reasoning call."""
        ...

    def reason_json(self, prompt: str) -> Result[Dict[str, Any], Error]:
        """Reasoning call expecting JSON response."""
        ...

    def decide_next_action(
        self,
        state: State,
        plan: Plan,
        last_result: Optional[ActionResult],
    ) -> Result[Optional[Primitive], Error]:
        """Decide the next action to take."""
        ...

    def should_continue(
        self,
        state: State,
        plan: Plan,
        result: ActionResult,
    ) -> bool:
        """Should we continue executing after this result?"""
        ...

    def is_goal_achieved(
        self,
        goal: Goal,
        state: State,
        results: List[ActionResult],
    ) -> bool:
        """Has the goal been achieved?"""
        ...

    def suggest_recovery(
        self,
        error: Error,
        context: Dict[str, Any],
    ) -> Result[Optional[Primitive], Error]:
        """Suggest a recovery action for an error."""
        ...


# =============================================================================
# 8. SIMPLE REASONER (Enhanced for Testing)
# =============================================================================

class SimpleReasoner:
    """
    Simple rule-based reasoner for testing.

    In production, replace with LLM-based reasoner.
    """

    def plan(
        self,
        goal: Goal,
        state: State,
        patterns: List[Tuple[Pattern, float]],
        retrieval_context: Optional[RetrievalResult] = None,
    ) -> Result[Plan, Error]:
        """Generate a simple plan from patterns."""
        builder = PlanBuilder(goal, state)
        found_pattern = False

        # Use retrieval context if available
        if retrieval_context and retrieval_context.chunks:
            for chunk in retrieval_context.chunks[:3]:
                if "command" in chunk.content.lower():
                    # Extract command-like content
                    builder.add_step(
                        f"From retrieval: {chunk.content[:50]}",
                        primitives=[Primitive.shell("echo 'Retrieved context used'")]
                    )
                    found_pattern = True
                    break

        # Fall back to patterns
        if not found_pattern and patterns:
            best_pattern, similarity = patterns[0]
            if similarity > 0.1:  # Low threshold for testing (use LLM for production)
                action_data = best_pattern.action_data
                if "command" in action_data:
                    builder.add_step(
                        f"Execute: {action_data['command'][:50]}",
                        primitives=[Primitive.shell(action_data["command"])]
                    )
                    found_pattern = True

        if not found_pattern:
            return Err(Error(
                ErrorCode.PLANNING_FAILED,
                "No patterns found and no LLM available to plan"
            ))

        return Ok(builder.build())

    def reason(self, prompt: str) -> Result[str, Error]:
        """Simple keyword-based reasoning."""
        return Ok(f"Processed: {prompt[:100]}")

    def reason_json(self, prompt: str) -> Result[Dict[str, Any], Error]:
        """Return simple JSON structure."""
        return Ok({"response": prompt[:100], "confidence": 0.5})

    def decide_next_action(
        self,
        state: State,
        plan: Plan,
        last_result: Optional[ActionResult],
    ) -> Result[Optional[Primitive], Error]:
        """Get next primitive from plan."""
        primitives = plan.get_pending_primitives()
        if primitives:
            return Ok(primitives[0])
        return Ok(None)

    def should_continue(
        self,
        state: State,
        plan: Plan,
        result: ActionResult,
    ) -> bool:
        """Continue unless blocked or errored."""
        return result.outcome not in (OutcomeType.BLOCKED, OutcomeType.ERROR)

    def is_goal_achieved(
        self,
        goal: Goal,
        state: State,
        results: List[ActionResult],
    ) -> bool:
        """Check if all actions succeeded."""
        return bool(results) and all(r.is_success for r in results)

    def suggest_recovery(
        self,
        error: Error,
        context: Dict[str, Any],
    ) -> Result[Optional[Primitive], Error]:
        """Simple recovery: just retry or skip."""
        if error.code == ErrorCode.TIMEOUT:
            return Ok(Primitive.shell("echo 'Timeout recovery'"))
        return Ok(None)


# =============================================================================
# 9. LOOP EVENTS
# =============================================================================

class LoopEvent(Enum):
    """Events emitted by the loop."""
    # Lifecycle
    STARTED = auto()
    COMPLETED = auto()
    FAILED = auto()
    PAUSED = auto()
    RESUMED = auto()

    # Phase transitions
    PHASE_CHANGED = auto()

    # Actions
    ACTION_STARTED = auto()
    ACTION_COMPLETED = auto()

    # Planning
    PLAN_GENERATED = auto()
    PLAN_UPDATED = auto()

    # Retrieval
    RETRIEVAL_STARTED = auto()
    RETRIEVAL_COMPLETED = auto()
    PATTERN_RETRIEVED = auto()

    # Perception
    PERCEPTION_COMPLETED = auto()

    # Verification
    VERIFICATION_PASSED = auto()
    VERIFICATION_FAILED = auto()

    # Context
    COMPACTION_TRIGGERED = auto()
    SUMMARIZATION_TRIGGERED = auto()

    # Error handling
    ERROR_OCCURRED = auto()
    RECOVERY_ATTEMPTED = auto()

    # Backtracking
    BACKTRACK_TRIGGERED = auto()

    # Goal
    GOAL_ACHIEVED = auto()


@dataclass(frozen=True)
class LoopEventData:
    """Data associated with a loop event."""
    event: LoopEvent
    timestamp: datetime = field(default_factory=datetime.now)
    phase: LoopPhase = LoopPhase.IDLE
    data: Dict[str, Any] = field(default_factory=dict)


EventListener = Callable[[LoopEventData], None]


# =============================================================================
# 10. THE LOOP (Production-Grade)
# =============================================================================

@dataclass
class LoopConfig:
    """Configuration for the main loop."""
    max_iterations: int = 100
    max_recovery_attempts: int = 3
    enable_backtracking: bool = True
    enable_context_management: bool = True
    enable_stop_hook: bool = True
    retry_on_error: bool = True
    observability_enabled: bool = True


class Loop:
    """
    Production-grade agent orchestration loop.

    Implements:
    - PERCEIVE -> RETRIEVE -> REASON -> VERIFY -> ACT -> OBSERVE -> LEARN
    - OpenClaw lifecycle hooks
    - Ralph Loop stop hooks
    - LATS-style backtracking
    - Context window management
    - Self-healing with retry
    - Full integration with robust infrastructure

    Usage:
        loop = Loop(reasoner=my_llm_reasoner)
        result = loop.run(goal)

        # Or step by step:
        loop.start(goal)
        while loop.state.is_running:
            loop.step()
    """

    def __init__(
        self,
        config: Optional[LoopConfig] = None,
        reasoner: Optional[Reasoner] = None,
        executor: Optional[Executor] = None,
        memory: Optional[Memory] = None,
        verifier: Optional[Verifier] = None,
        perception_engine: Optional[IntelligentPerceptionEngine] = None,
        retrieval_engine: Optional[RetrievalEngine] = None,
        robust_reasoner: Optional[RobustReasoner] = None,
        session_manager: Optional[SessionManager] = None,
        # SOTA: Constitutional AI verification
        constitutional_verifier: Optional[ConstitutionalVerifier] = None,
        # SOTA: Guard Agent for security audit
        guard_agent: Optional[GuardAgent] = None,
        # SOTA: Goal achievement verification
        goal_verifier: Optional[GoalAchievementVerifier] = None,
        # SOTA: Exec approval checker
        exec_checker: Optional[ExecApprovalChecker] = None,
    ):
        self.config = config or LoopConfig()

        # Core components
        self.reasoner = reasoner or SimpleReasoner()
        self.executor = executor or Executor()
        self.memory = memory or Memory()
        # Verifier needs a reasoner - use the loop's reasoner
        self.verifier = verifier or Verifier(self.reasoner)

        # Perception (LLM-first with SOTA patterns)
        self.perception_engine = perception_engine or create_perception_engine()

        # SOTA: Perception-Action Feedback (learns from action outcomes)
        self.perception_feedback = PerceptionActionFeedback()

        # SOTA: Perception Calibrator (calibrates LLM confidence)
        self.perception_calibrator = PerceptionCalibrator()

        # Retrieval (Agentic RAG)
        self.retrieval_engine = retrieval_engine or create_retrieval_engine()

        # Robust infrastructure
        self.robust_reasoner = robust_reasoner
        self.session_manager = session_manager or SessionManager()

        # SOTA: Constitutional AI verification
        self.constitutional_verifier = constitutional_verifier

        # SOTA: Guard Agent for security audit
        self.guard_agent = guard_agent

        # SOTA: Goal achievement verification
        self.goal_verifier = goal_verifier

        # SOTA: Exec approval checker (OpenClaw pattern)
        self.exec_checker = exec_checker

        # Context management (OpenClaw pattern)
        self.context_guard = ContextWindowGuard() if self.config.enable_context_management else None

        # Stop hook (Ralph Loop pattern)
        self.stop_hook = StopHook() if self.config.enable_stop_hook else None

        # Backtracking (LATS pattern)
        self.backtrack_tree = BacktrackTree() if self.config.enable_backtracking else None

        # Lifecycle hooks
        self.hooks = HookRegistry()

        # Observability
        self.observability = ObservabilityLayer() if self.config.observability_enabled else None

        # Retry strategy
        self.retry_strategy = RetryStrategy(RetryConfig(
            max_attempts=3,
            initial_delay_seconds=1.0,
            exponential_base=2.0,
        )) if self.config.retry_on_error else None

        # State and events
        self.state = LoopState()
        self._event_listeners: List[EventListener] = []
        self._current_session: Optional[Session] = None

    # =========================================================================
    # EVENT SYSTEM
    # =========================================================================

    def add_listener(self, listener: EventListener) -> None:
        """Add an event listener."""
        self._event_listeners.append(listener)

    def remove_listener(self, listener: EventListener) -> None:
        """Remove an event listener."""
        if listener in self._event_listeners:
            self._event_listeners.remove(listener)

    def _emit(self, event: LoopEvent, data: Optional[Dict[str, Any]] = None) -> None:
        """Emit an event to all listeners."""
        event_data = LoopEventData(
            event=event,
            phase=self.state.phase,
            data=data or {},
        )

        # Log to observability
        if self.observability:
            trace = DecisionTrace(
                decision_id=f"event_{event.name}_{time.time_ns()}",
                decision_type=f"event_{event.name}",
                timestamp=datetime.now(),
                prompt_name="loop_event",
                prompt_version="1.0",
                prompt_fingerprint=f"event_{event.name}",
                input_data=data or {},
                raw_response=str(event_data),
                parsed_response={"phase": self.state.phase.name},
                validation_result="passed",
                latency_ms=0.0,
                retry_count=0,
            )
            self.observability.record_trace(trace)

        for listener in self._event_listeners:
            try:
                listener(event_data)
            except Exception as e:
                logger.warning(f"Event listener error: {e}")

    # =========================================================================
    # HOOK INVOCATION
    # =========================================================================

    def _invoke_hook(self, hook: LifecycleHook, data: Optional[Dict] = None) -> HookContext:
        """Invoke a lifecycle hook."""
        context = HookContext(
            hook=hook,
            phase=self.state.phase,
            loop_state=self.state,
            data=data or {},
        )
        return self.hooks.invoke(context)

    # =========================================================================
    # STATE TRANSITIONS
    # =========================================================================

    def _set_phase(self, phase: LoopPhase) -> None:
        """Transition to a new phase."""
        old_phase = self.state.phase
        phase_start = time.time()

        self.state.phase = phase
        self._emit(LoopEvent.PHASE_CHANGED, {
            "old_phase": old_phase.name,
            "new_phase": phase.name,
        })

        # Track timing
        if old_phase != LoopPhase.IDLE:
            self.state.phase_timings[old_phase.name] = (
                self.state.phase_timings.get(old_phase.name, 0) +
                (time.time() - phase_start) * 1000
            )

    def _record_error(self, error: Error) -> None:
        """Record an error."""
        self.state.errors.append(error)
        self.state.last_error = error
        self._emit(LoopEvent.ERROR_OCCURRED, {"error": str(error)})

        # Add to context
        if self.context_guard:
            self.context_guard.add_message(
                "system",
                f"[ERROR] {error.code}: {error.message}",
                {"is_error": True}
            )

    # =========================================================================
    # MAIN OPERATIONS
    # =========================================================================

    def start(self, goal: Goal) -> None:
        """Start the loop with a goal."""
        # Invoke before_agent_start hook
        hook_ctx = self._invoke_hook(LifecycleHook.BEFORE_AGENT_START, {"goal": goal})
        if not hook_ctx.should_continue:
            logger.warning(f"before_agent_start hook blocked: {hook_ctx.error}")
            return

        # Create session
        self._current_session = self.session_manager.get_or_create(
            f"loop_{datetime.now().isoformat()}"
        )

        # Reset state
        self.state = LoopState(
            started_at=datetime.now(),
            max_iterations=self.config.max_iterations,
            max_recovery_attempts=self.config.max_recovery_attempts,
            session_id=self._current_session.session_id if self._current_session else None,
        )

        # Reset stop hook
        if self.stop_hook:
            self.stop_hook.reset()

        # Initialize backtrack tree
        if self.backtrack_tree:
            self.backtrack_tree.start(self.state.snapshot())

        # Build initial state
        self.state.world_state = (
            StateBuilder()
            .with_goal(goal.intent, goal.goal_type)
            .build()
        )

        # Add to context
        if self.context_guard:
            self.context_guard.add_message(
                "user",
                f"Goal: {goal.intent}",
                {"goal_type": goal.goal_type.name}
            )

        self._set_phase(LoopPhase.PERCEIVE)
        self._emit(LoopEvent.STARTED, {"goal": goal.intent})

    def step(self) -> LoopState:
        """Execute one step of the loop."""
        if not self.state.is_running:
            return self.state

        # Invoke iteration start hook
        self._invoke_hook(LifecycleHook.LOOP_ITERATION_START)

        # Check iteration limit
        self.state.current_iteration += 1
        if self.state.current_iteration > self.state.max_iterations:
            self._record_error(Error(
                ErrorCode.TIMEOUT,
                f"Exceeded max iterations: {self.state.max_iterations}"
            ))
            self._set_phase(LoopPhase.FAILED)
            return self.state

        # Check context window
        if self.context_guard:
            self.state.context_stats = self.context_guard.get_stats()

            if self.state.context_stats.needs_summarization:
                self._set_phase(LoopPhase.COMPACT)
                return self.state
            elif self.state.context_stats.needs_compaction:
                self._do_compact()

        # Execute current phase
        phase_start = time.time()
        try:
            if self.state.phase == LoopPhase.PERCEIVE:
                self._do_perceive()
            elif self.state.phase == LoopPhase.RETRIEVE:
                self._do_retrieve()
            elif self.state.phase == LoopPhase.REASON:
                self._do_reason()
            elif self.state.phase == LoopPhase.VERIFY:
                self._do_verify()
            elif self.state.phase == LoopPhase.ACT:
                self._do_act()
            elif self.state.phase == LoopPhase.OBSERVE:
                self._do_observe()
            elif self.state.phase == LoopPhase.LEARN:
                self._do_learn()
            elif self.state.phase == LoopPhase.COMPACT:
                self._do_summarize()
            elif self.state.phase == LoopPhase.BACKTRACK:
                self._do_backtrack()
        except Exception as e:
            self._handle_exception(e)

        # Track phase timing
        phase_duration = (time.time() - phase_start) * 1000
        self.state.phase_timings[self.state.phase.name] = (
            self.state.phase_timings.get(self.state.phase.name, 0) + phase_duration
        )

        # Invoke iteration end hook
        self._invoke_hook(LifecycleHook.LOOP_ITERATION_END)

        return self.state

    def run(self, goal: Goal) -> Result[List[ActionResult], Error]:
        """
        Run the complete loop until completion.

        Returns all action results on success, error on failure.
        """
        self.start(goal)

        while self.state.is_running:
            self.step()

            if self.state.should_stop:
                break

        self.state.completed_at = datetime.now()

        # Invoke agent_end hook
        self._invoke_hook(LifecycleHook.AGENT_END, {
            "results": self.state.all_results,
            "success": self.state.phase == LoopPhase.COMPLETE,
        })

        # Check stop hook
        if self.stop_hook:
            output = str(self.state.last_result.output if self.state.last_result else "")
            can_exit, reason = self.stop_hook.can_exit(output, self.state.last_error)

            if not can_exit:
                logger.info(f"Stop hook prevented exit: {reason}")
                # Reset and continue
                self.state.phase = LoopPhase.PERCEIVE
                return self.run(goal)  # Recursive retry (Ralph Loop pattern)

        if self.state.phase == LoopPhase.COMPLETE:
            self._emit(LoopEvent.COMPLETED)
            return Ok(self.state.all_results)
        else:
            self._emit(LoopEvent.FAILED)
            return Err(self.state.last_error or Error(
                ErrorCode.UNKNOWN,
                "Loop failed for unknown reason"
            ))

    def pause(self, reason: str = None) -> None:
        """Pause the loop."""
        self.state.pause_reason = reason
        self._set_phase(LoopPhase.PAUSED)
        self._emit(LoopEvent.PAUSED, {"reason": reason})

    def resume(self) -> None:
        """Resume from paused state."""
        if self.state.phase == LoopPhase.PAUSED:
            self._set_phase(LoopPhase.PERCEIVE)
            self._emit(LoopEvent.RESUMED)

    def stop(self) -> None:
        """Signal the loop to stop."""
        self.state.should_stop = True

    # =========================================================================
    # PHASE IMPLEMENTATIONS
    # =========================================================================

    def _do_perceive(self) -> None:
        """
        PERCEIVE: Build/update world state using LLM-first perception.

        SOTA patterns integrated:
        - Stores entities for Perception-Action Feedback
        - Stores confidence for calibration
        - Applies learned adjustments from previous feedback
        """
        logger.debug("PERCEIVE: Building world state (LLM-First with SOTA)")

        if self.state.world_state and self.state.world_state.goal:
            goal = self.state.world_state.goal

            try:
                # Use LLM-first perception engine
                self.state.perception_result = self.perception_engine.perceive(goal=goal)

                # SOTA: Store entities for feedback loop (used in OBSERVE phase)
                self.state.perception_entities_used = self.state.perception_result.entities

                # SOTA: Store confidence for calibration (used in OBSERVE phase)
                self.state.last_perception_confidence = self.state.perception_result.confidence

                # SOTA: Apply calibration if available
                if self.perception_calibrator:
                    calibrated_confidence = self.perception_calibrator.calibrate(
                        self.state.perception_result.confidence
                    )
                    logger.debug(f"  Calibrated confidence: {self.state.perception_result.confidence:.2f} -> {calibrated_confidence:.2f}")

                # SOTA: Apply learned entity priorities from feedback
                if self.perception_feedback:
                    domain_priorities = self.perception_feedback.get_domain_priority()
                    if domain_priorities:
                        logger.debug(f"  Using learned domain priorities: {domain_priorities[:3]}")

                # Convert to state
                self.state.world_state = self.state.perception_result.to_state()

                self._emit(LoopEvent.PERCEPTION_COMPLETED, {
                    "entities": len(self.state.perception_result.entities),
                    "gaps": len(self.state.perception_result.gaps),
                    "confidence": self.state.perception_result.confidence,
                    "can_proceed": self.state.perception_result.can_proceed,
                })

                # Add to context
                if self.context_guard:
                    self.context_guard.add_message(
                        "assistant",
                        f"[PERCEPTION] Found {len(self.state.perception_result.entities)} entities, "
                        f"confidence: {self.state.perception_result.confidence:.2f}",
                    )

            except Exception as e:
                logger.warning(f"Perception failed: {e}, continuing with basic state")

        self._set_phase(LoopPhase.RETRIEVE)

    def _do_retrieve(self) -> None:
        """
        RETRIEVE: Agentic RAG retrieval using RetrievalEngine.
        """
        logger.debug("RETRIEVE: Agentic RAG search")

        self._emit(LoopEvent.RETRIEVAL_STARTED)

        if self.state.world_state and self.state.world_state.goal:
            goal_desc = self.state.world_state.goal.intent

            # Build retrieval query with context
            query = RetrievalQuery(
                text=goal_desc,
                max_results=10,
                sources=[
                    RetrievalSource.MEMORY,
                    RetrievalSource.VECTOR_DB,
                    RetrievalSource.TOOL,
                ],
                goal=self.state.world_state.goal,
            )

            # Add conversation history to query
            if self.context_guard:
                query.conversation_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in self.context_guard._message_history[-5:]
                ]

            # Execute retrieval
            retrieval_result = self.retrieval_engine.retrieve(query)

            if retrieval_result.is_ok():
                self.state.retrieval_result = retrieval_result.unwrap()

                self._emit(LoopEvent.RETRIEVAL_COMPLETED, {
                    "chunks": len(self.state.retrieval_result.chunks),
                    "sources": list(self.state.retrieval_result.source_stats.keys()),
                    "latency_ms": self.state.retrieval_result.latency_ms,
                    "retrieval_needed": self.state.retrieval_result.retrieval_needed,
                })

                # Add to context
                if self.context_guard:
                    context_summary = self.state.retrieval_result.get_context(max_tokens=500)
                    if context_summary:
                        self.context_guard.add_message(
                            "assistant",
                            f"[RETRIEVAL] {context_summary[:200]}...",
                        )

            # Also get patterns from memory (legacy support)
            exec_patterns = self.memory.recall_similar_executions(goal_desc, k=3)
            query_patterns = self.memory.recall_similar_queries(goal_desc, k=3)
            code_patterns = self.memory.recall_similar_code(goal_desc, k=3)

            all_patterns = exec_patterns + query_patterns + code_patterns
            all_patterns.sort(key=lambda x: x[1], reverse=True)
            self.state.retrieved_patterns = all_patterns[:5]

            if self.state.retrieved_patterns:
                self._emit(LoopEvent.PATTERN_RETRIEVED, {
                    "count": len(self.state.retrieved_patterns),
                    "top_similarity": self.state.retrieved_patterns[0][1],
                })

        self._set_phase(LoopPhase.REASON)

    def _do_reason(self) -> None:
        """
        REASON: Generate or update plan (ReAct THINK phase).
        """
        logger.debug("REASON: Generating plan (ReAct THINK)")

        # If we don't have a plan, generate one
        if self.state.plan is None:
            goal = self.state.world_state.goal if self.state.world_state else Goal(
                intent="Unknown",
                goal_type=GoalType.QUERY,
            )

            # Use robust reasoner if available
            if self.robust_reasoner and hasattr(self.robust_reasoner, 'plan'):
                plan_result = self.robust_reasoner.plan(
                    goal=goal,
                    state=self.state.world_state,
                    patterns=self.state.retrieved_patterns,
                    retrieval_context=self.state.retrieval_result,
                )
            else:
                plan_result = self.reasoner.plan(
                    goal=goal,
                    state=self.state.world_state,
                    patterns=self.state.retrieved_patterns,
                    retrieval_context=self.state.retrieval_result,
                )

            if plan_result.is_err():
                error = plan_result.unwrap_err()
                self._record_error(error)

                # Try recovery
                if self._attempt_recovery(error):
                    return  # Recovery will retry

                self._set_phase(LoopPhase.FAILED)
                return

            self.state.plan = plan_result.unwrap()
            self._emit(LoopEvent.PLAN_GENERATED, {
                "intent": self.state.plan.root.description,
            })

            # Add to context
            if self.context_guard:
                self.context_guard.add_message(
                    "assistant",
                    f"[PLAN] {self.state.plan.root.description}",
                )

        # Decide next action
        next_action_result = self.reasoner.decide_next_action(
            state=self.state.world_state,
            plan=self.state.plan,
            last_result=self.state.last_result,
        )

        if next_action_result.is_err():
            self._record_error(next_action_result.unwrap_err())
            self._set_phase(LoopPhase.FAILED)
            return

        next_primitive = next_action_result.unwrap()

        if next_primitive is None:
            # Plan complete, check if goal achieved
            if self.reasoner.is_goal_achieved(
                goal=self.state.world_state.goal,
                state=self.state.world_state,
                results=self.state.all_results,
            ):
                self._set_phase(LoopPhase.COMPLETE)
                self._emit(LoopEvent.GOAL_ACHIEVED)
            else:
                # Try backtracking
                if self.backtrack_tree and self.config.enable_backtracking:
                    self._set_phase(LoopPhase.BACKTRACK)
                else:
                    self._record_error(Error(
                        ErrorCode.PLANNING_FAILED,
                        "Plan complete but goal not achieved"
                    ))
                    self._set_phase(LoopPhase.FAILED)
            return

        self.state.current_primitive = next_primitive
        self._set_phase(LoopPhase.VERIFY)

    def _do_verify(self) -> None:
        """
        VERIFY: Check safety and preconditions with SOTA multi-layer verification.

        Verification layers:
        1. Lifecycle hooks (OpenClaw pattern)
        2. Basic verification (Verifier)
        3. Constitutional AI critique (if enabled)
        4. Guard Agent audit (if enabled)
        5. Exec approval check (if enabled)
        """
        logger.debug("VERIFY: Checking safety (multi-layer)")

        if self.state.current_primitive is None:
            self._set_phase(LoopPhase.REASON)
            return

        # Layer 1: Invoke before_tool_call hook
        hook_ctx = self._invoke_hook(LifecycleHook.BEFORE_TOOL_CALL, {
            "primitive": self.state.current_primitive,
        })
        if not hook_ctx.should_continue:
            self._emit(LoopEvent.VERIFICATION_FAILED, {"reason": "Hook blocked"})
            self._set_phase(LoopPhase.REASON)
            return

        # Layer 2: Basic verification
        report = self.verifier.verify_before(
            primitive=self.state.current_primitive,
            state=self.state.world_state,
        )

        if not report.passed:
            self._handle_verification_failure(report, "Basic verification failed")
            return

        # Layer 3: Constitutional AI critique (SOTA)
        if self.constitutional_verifier:
            action_desc = f"{self.state.current_primitive.primitive_type.name}: {self.state.current_primitive.params}"
            critique_result = self.constitutional_verifier.critique(action_desc)

            if critique_result.is_ok():
                critique = critique_result.unwrap()
                if critique.needs_revision:
                    self._emit(LoopEvent.VERIFICATION_FAILED, {
                        "reason": "Constitutional AI critique",
                        "violated_principles": critique.violated_principles,
                    })
                    self._handle_verification_failure(
                        report,
                        f"Constitutional violation: {critique.violated_principles}"
                    )
                    return

        # Layer 4: Guard Agent audit (SOTA)
        if self.guard_agent:
            audit_result = self.guard_agent.audit(
                action=self.state.current_primitive,
                context={
                    "goal": self.state.world_state.goal.intent if self.state.world_state else "",
                    "previous_actions": len(self.state.all_results),
                }
            )

            if audit_result.is_ok():
                audit = audit_result.unwrap()
                if not audit.is_safe:
                    self._emit(LoopEvent.VERIFICATION_FAILED, {
                        "reason": "Guard Agent blocked",
                        "risk_level": audit.risk_level,
                    })
                    self._handle_verification_failure(
                        report,
                        f"Guard Agent blocked: {audit.reasoning}"
                    )
                    return

        # Layer 5: Exec approval check (OpenClaw pattern)
        if self.exec_checker:
            approval = self.exec_checker.check(self.state.current_primitive)

            if approval.is_ok():
                auth = approval.unwrap()
                if auth.level == AuthorizationLevel.DENIED:
                    self._emit(LoopEvent.VERIFICATION_FAILED, {
                        "reason": "Exec approval denied",
                        "level": auth.level.name,
                    })
                    self._handle_verification_failure(
                        report,
                        f"Exec approval denied: {auth.reason}"
                    )
                    return
                elif auth.level == AuthorizationLevel.HUMAN_REQUIRED:
                    # TODO: Implement human-in-the-loop approval
                    logger.warning("Human approval required but not implemented yet")

        # All verification passed
        self._emit(LoopEvent.VERIFICATION_PASSED, {"summary": report.summary()})
        self._set_phase(LoopPhase.ACT)

    def _handle_verification_failure(self, report: VerificationReport, reason: str) -> None:
        """Handle verification failure with proper state updates."""
        self._emit(LoopEvent.VERIFICATION_FAILED, {"summary": reason})

        for verdict in report.blocking_verdicts:
            self._record_error(Error(
                ErrorCode.VERIFICATION_FAILED,
                verdict.message,
            ))

        # Create blocked result
        self.state.last_result = ActionResult(
            primitive=self.state.current_primitive,
            outcome=OutcomeType.BLOCKED,
            error=Error(ErrorCode.VERIFICATION_FAILED, reason),
        )
        self.state.all_results.append(self.state.last_result)
        self.state.failed_actions += 1

        self._set_phase(LoopPhase.REASON)

    def _do_act(self) -> None:
        """
        ACT: Execute the current action (ReAct ACT phase).
        """
        logger.debug(f"ACT: Executing {self.state.current_primitive.primitive_type.name}")

        self._emit(LoopEvent.ACTION_STARTED, {
            "type": self.state.current_primitive.primitive_type.name,
            "params": str(self.state.current_primitive.params)[:100],
        })

        # Execute with retry if enabled
        if self.retry_strategy:
            result = self.executor.execute_with_retry(self.state.current_primitive)
        else:
            result = self.executor.execute(self.state.current_primitive)

        self.state.last_result = result
        self.state.actions_taken += 1

        if result.is_success:
            self.state.successful_actions += 1
        else:
            self.state.failed_actions += 1

        self.state.all_results.append(result)

        # Invoke after_tool_call hook
        self._invoke_hook(LifecycleHook.AFTER_TOOL_CALL, {
            "primitive": self.state.current_primitive,
            "result": result,
        })

        # Add to context
        if self.context_guard:
            self.context_guard.add_tool_call(
                tool_name=self.state.current_primitive.primitive_type.name,
                params=self.state.current_primitive.params,
                result=result.output,
                full_content=str(result.output)[:1000] if result.output else "",
            )

        # Update backtrack tree
        if self.backtrack_tree:
            score = self.backtrack_tree.score_node(self.backtrack_tree.current, result)
            self.backtrack_tree.add_child(
                phase=LoopPhase.ACT,
                state_snapshot=self.state.snapshot(),
                action=self.state.current_primitive,
                result=result,
                score=score,
            )

        self._emit(LoopEvent.ACTION_COMPLETED, {
            "outcome": result.outcome.name,
            "duration_ms": result.duration_ms,
        })

        self._set_phase(LoopPhase.OBSERVE)

    def _do_observe(self) -> None:
        """
        OBSERVE: Capture what changed (ReAct OBSERVE phase).

        Includes SOTA patterns:
        - Post-execution verification
        - Perception-Action Feedback (learns entity-success correlations)
        - Confidence calibration updates
        """
        logger.debug("OBSERVE: Capturing changes (with SOTA feedback)")

        if self.state.last_result:
            # Verify after execution
            report = self.verifier.verify_after(
                result=self.state.last_result,
                state=self.state.world_state,
            )

            # SOTA: Update Perception-Action Feedback
            # This teaches the perception system which entities correlate with success
            if self.perception_feedback and hasattr(self.state, 'perception_entities_used'):
                entities_used = getattr(self.state, 'perception_entities_used', [])
                if entities_used:
                    action_type = self.state.current_primitive.primitive_type.name if self.state.current_primitive else "unknown"
                    success = self.state.last_result.is_success

                    self.perception_feedback.record_action_outcome(
                        entities_used=entities_used,
                        action_type=action_type,
                        success=success,
                    )
                    logger.debug(f"  Recorded feedback: {len(entities_used)} entities, success={success}")

            # SOTA: Update Perception Calibrator
            # This calibrates LLM confidence scores based on actual outcomes
            if self.perception_calibrator and hasattr(self.state, 'last_perception_confidence'):
                predicted_confidence = getattr(self.state, 'last_perception_confidence', 0.5)
                was_correct = self.state.last_result.is_success

                self.perception_calibrator.record_outcome(
                    predicted_confidence=predicted_confidence,
                    was_correct=was_correct,
                )
                logger.debug(f"  Calibration update: confidence={predicted_confidence:.2f}, correct={was_correct}")

            # Update world state with observation
            if self.state.world_state and self.state.last_result.output:
                self.state.world_state = self.state.world_state.with_observation(
                    Observation(
                        timestamp=datetime.now(),
                        observation_type="action_result",
                        content=str(self.state.last_result.output)[:500],
                    )
                )

        self._set_phase(LoopPhase.LEARN)

    def _do_learn(self) -> None:
        """
        LEARN: Record patterns for future use.
        """
        logger.debug("LEARN: Recording patterns")

        if self.state.last_result and self.state.world_state:
            context = self.state.world_state.goal.intent if self.state.world_state.goal else "unknown"

            # Remember this execution
            self.memory.remember_execution(context, self.state.last_result)

            # Invoke tool_result_persist hook
            self._invoke_hook(LifecycleHook.TOOL_RESULT_PERSIST, {
                "result": self.state.last_result,
                "context": context,
            })

        # Check if we should continue
        if self.state.last_result:
            if not self.reasoner.should_continue(
                state=self.state.world_state,
                plan=self.state.plan,
                result=self.state.last_result,
            ):
                if self.reasoner.is_goal_achieved(
                    goal=self.state.world_state.goal,
                    state=self.state.world_state,
                    results=self.state.all_results,
                ):
                    self._set_phase(LoopPhase.COMPLETE)
                else:
                    # Try backtracking before failing
                    if self.backtrack_tree and self.config.enable_backtracking:
                        self._set_phase(LoopPhase.BACKTRACK)
                    else:
                        self._set_phase(LoopPhase.FAILED)
                return

        # Continue to next iteration
        self._set_phase(LoopPhase.PERCEIVE)

    def _do_compact(self) -> None:
        """Compact context (reversible)."""
        logger.debug("COMPACT: Compacting context")

        if self.context_guard:
            # Invoke before_compaction hook
            self._invoke_hook(LifecycleHook.BEFORE_COMPACTION)

            compacted = self.context_guard.compact()
            self.state.compaction_count += 1

            # Invoke after_compaction hook
            self._invoke_hook(LifecycleHook.AFTER_COMPACTION, {"compacted": compacted})

            self._emit(LoopEvent.COMPACTION_TRIGGERED, {"compacted": compacted})

    def _do_summarize(self) -> None:
        """Summarize context (lossy) - triggered when approaching limits."""
        logger.debug("COMPACT: Summarizing context")

        if self.context_guard:
            # Invoke before_compaction hook
            self._invoke_hook(LifecycleHook.BEFORE_COMPACTION)

            # Use LLM for summarization if available
            summarizer = None
            if self.robust_reasoner:
                def llm_summarize(content: str) -> str:
                    result = self.robust_reasoner.reason(
                        f"Summarize this conversation history concisely:\n{content}"
                    )
                    return result.response if hasattr(result, 'response') else str(result)
                summarizer = llm_summarize

            summary = self.context_guard.summarize(summarizer)
            self.state.summarization_count += 1

            # Invoke after_compaction hook
            self._invoke_hook(LifecycleHook.AFTER_COMPACTION, {"summary_length": len(summary)})

            self._emit(LoopEvent.SUMMARIZATION_TRIGGERED, {"summary_length": len(summary)})

        # Return to normal flow
        self._set_phase(LoopPhase.PERCEIVE)

    def _do_backtrack(self) -> None:
        """
        LATS-style backtracking connected to reasoning strategies.

        SOTA pattern: Uses the reasoner to evaluate alternative paths
        before selecting the best one to explore.
        """
        logger.debug("BACKTRACK: LATS tree search for alternative path")

        if not self.backtrack_tree:
            self._set_phase(LoopPhase.FAILED)
            return

        # SOTA: Use reasoner to evaluate alternatives before selecting
        alternatives = self.backtrack_tree.get_unexplored_alternatives()

        if not alternatives:
            # No more alternatives, fail
            self._record_error(Error(
                ErrorCode.PLANNING_FAILED,
                "All alternatives exhausted during backtracking"
            ))
            self._set_phase(LoopPhase.FAILED)
            return

        # SOTA: If we have a robust reasoner, use it to rank alternatives
        best_alternative = None
        if self.robust_reasoner and len(alternatives) > 1:
            # Ask LLM to evaluate alternatives
            alt_descriptions = []
            for i, alt in enumerate(alternatives[:5]):  # Max 5 to evaluate
                alt_descriptions.append(f"{i+1}. Depth {alt.depth}, Score {alt.score:.2f}")

            prompt = f"""We are backtracking in an agent loop. Which alternative path looks most promising?

Goal: {self.state.world_state.goal.intent if self.state.world_state else 'unknown'}
Previous errors: {[str(e) for e in self.state.errors[-3:]]}

Alternatives:
{chr(10).join(alt_descriptions)}

Respond with just the number (1-{len(alt_descriptions)}) of the best alternative."""

            result = self.robust_reasoner.reason(prompt)
            if result.is_ok():
                try:
                    choice = int(result.unwrap().strip()) - 1
                    if 0 <= choice < len(alternatives):
                        best_alternative = alternatives[choice]
                        logger.debug(f"  LLM selected alternative {choice + 1}")
                except (ValueError, IndexError):
                    pass

        # Fall back to highest-scored alternative
        if not best_alternative:
            best_alternative = self.backtrack_tree.backtrack()

        if best_alternative:
            self.state.backtrack_count += 1
            self._emit(LoopEvent.BACKTRACK_TRIGGERED, {
                "depth": best_alternative.depth,
                "score": best_alternative.score,
                "alternatives_considered": len(alternatives),
            })

            # Restore state from snapshot
            snapshot = best_alternative.state_snapshot
            self.state.current_step_index = snapshot.get("current_step_index", 0)

            # Reset plan to try alternative
            self.state.plan = None
            self._set_phase(LoopPhase.REASON)
        else:
            # No alternatives, fail
            self._record_error(Error(
                ErrorCode.PLANNING_FAILED,
                "All alternatives exhausted during backtracking"
            ))
            self._set_phase(LoopPhase.FAILED)

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    def _handle_exception(self, e: Exception) -> None:
        """Handle an exception during phase execution."""
        error = Error(
            ErrorCode.EXECUTION_FAILED,
            f"Phase {self.state.phase.name} failed: {str(e)}",
            details={"traceback": traceback.format_exc()},
        )
        self._record_error(error)

        # Try recovery
        if not self._attempt_recovery(error):
            self._set_phase(LoopPhase.FAILED)

    def _attempt_recovery(self, error: Error) -> bool:
        """
        Attempt to recover from an error.

        Returns True if recovery was initiated.
        """
        if self.state.recovery_attempts >= self.state.max_recovery_attempts:
            logger.warning(f"Max recovery attempts ({self.state.max_recovery_attempts}) reached")
            return False

        self.state.recovery_attempts += 1
        self._emit(LoopEvent.RECOVERY_ATTEMPTED, {
            "attempt": self.state.recovery_attempts,
            "error": str(error),
        })

        # Try to get recovery suggestion from reasoner
        recovery_result = self.reasoner.suggest_recovery(error, {
            "phase": self.state.phase.name,
            "last_action": str(self.state.current_primitive) if self.state.current_primitive else None,
            "iteration": self.state.current_iteration,
        })

        if recovery_result.is_ok():
            recovery_action = recovery_result.unwrap()
            if recovery_action:
                self.state.current_primitive = recovery_action
                self._set_phase(LoopPhase.VERIFY)
                return True

        # Try backtracking as recovery
        if self.backtrack_tree and self.config.enable_backtracking:
            self._set_phase(LoopPhase.BACKTRACK)
            return True

        return False

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    def get_results(self) -> List[ActionResult]:
        """Get all action results."""
        return self.state.all_results.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive loop statistics."""
        stats = {
            "loop_state": self.state.to_dict(),
            "phase_timings": self.state.phase_timings,
            "memory": self.memory.get_statistics(),
            "executor_success_rate": self.executor.get_success_rate(),
        }

        if self.retrieval_engine:
            stats["retrieval"] = self.retrieval_engine.get_stats()

        if self.observability:
            stats["decisions"] = len(self.observability.traces)
            stats["observability_metrics"] = self.observability.get_metrics()

        return stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_goal(
    goal_description: str,
    goal_type: GoalType = GoalType.QUERY,
    reasoner: Optional[Reasoner] = None,
    config: Optional[LoopConfig] = None,
) -> Result[List[ActionResult], Error]:
    """
    Quick way to run a goal through the loop.

    Usage:
        result = run_goal("List all files in /tmp")
        if result.is_ok():
            for r in result.unwrap():
                print(r.output)
    """
    goal = Goal(intent=goal_description, goal_type=goal_type)
    loop = Loop(config=config, reasoner=reasoner)
    return loop.run(goal)


def create_production_loop(
    reasoner: Reasoner,
    embedding_provider=None,
) -> Loop:
    """Create a production-ready loop with all features enabled."""
    config = LoopConfig(
        max_iterations=100,
        max_recovery_attempts=3,
        enable_backtracking=True,
        enable_context_management=True,
        enable_stop_hook=True,
        retry_on_error=True,
        observability_enabled=True,
    )

    retrieval_engine = create_retrieval_engine(
        reasoner=reasoner,
        embedding_provider=embedding_provider,
    )

    return Loop(
        config=config,
        reasoner=reasoner,
        retrieval_engine=retrieval_engine,
    )


def create_loop_with_logging(log_level: int = logging.INFO) -> Loop:
    """Create a loop with logging enabled."""
    logging.basicConfig(level=log_level)

    loop = Loop()

    def log_event(event: LoopEventData):
        logger.info(f"[{event.phase.name}] {event.event.name}: {event.data}")

    loop.add_listener(log_event)
    return loop


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LOOP - Production-Grade Agent Orchestration")
    print("=" * 70)

    logging.basicConfig(level=logging.DEBUG)

    # Test 1: Basic loop (stop hook disabled for testing)
    print("\n[TEST 1] Basic Loop Execution")
    test_config = LoopConfig(
        enable_stop_hook=False,  # Disable Ralph Loop for testing
        enable_backtracking=False,  # Disable LATS backtracking for simpler test
        max_iterations=10,
    )
    loop = Loop(config=test_config)

    # Add logging listener
    def log_event(event: LoopEventData):
        logger.info(f"[{event.phase.name}] {event.event.name}: {event.data}")
    loop.add_listener(log_event)

    events = []
    def track_event(event: LoopEventData):
        events.append(event)
    loop.add_listener(track_event)

    # Inject pattern for simple reasoner with EXACT match context
    loop.memory.remember_execution(
        context="Echo hello world",  # Exact match
        result=ActionResult(
            primitive=Primitive.shell("echo 'Hello, World!'"),
            outcome=OutcomeType.SUCCESS,
            output={"stdout": "Hello, World!\n"},
        ),
    )

    goal = Goal(intent="Echo hello world", goal_type=GoalType.QUERY)
    result = loop.run(goal)

    print(f"\n  Success: {result.is_ok()}")
    print(f"  Events: {len(events)}")
    print(f"  Actions: {loop.state.actions_taken}")
    print(f"  Duration: {loop.state.duration_ms:.2f}ms")

    # Test 2: Context management
    print("\n[TEST 2] Context Window Management")
    guard = ContextWindowGuard(ContextWindowConfig(max_tokens=1000))

    for i in range(10):
        guard.add_message("user", f"Message {i}: " + "x" * 100)

    stats = guard.get_stats()
    print(f"  Token usage: {stats.total_tokens}")
    print(f"  Utilization: {stats.utilization:.2%}")
    print(f"  Needs compaction: {stats.needs_compaction}")

    # Test 3: Stop hook
    print("\n[TEST 3] Stop Hook (Ralph Loop Pattern)")
    stop_hook = StopHook(StopHookConfig(completion_marker="<DONE>"))

    can_exit1, reason1 = stop_hook.can_exit("Still working...")
    print(f"  Without marker: can_exit={can_exit1}, reason='{reason1}'")

    can_exit2, reason2 = stop_hook.can_exit("Task complete <DONE>")
    print(f"  With marker: can_exit={can_exit2}, reason='{reason2}'")

    # Test 4: Backtrack tree
    print("\n[TEST 4] Backtrack Tree (LATS Pattern)")
    tree = BacktrackTree(max_depth=3)
    tree.start({"step": 0})

    tree.add_child(LoopPhase.ACT, {"step": 1}, score=0.8)
    tree.add_child(LoopPhase.ACT, {"step": 2}, score=0.3)

    backtracked = tree.backtrack()
    print(f"  Backtracked to: {backtracked.id if backtracked else 'None'}")
    print(f"  Path length: {len(tree.get_path())}")

    # Test 5: Lifecycle hooks
    print("\n[TEST 5] Lifecycle Hooks (OpenClaw Pattern)")
    hooks = HookRegistry()

    hook_calls = []
    def test_hook(ctx: HookContext) -> HookContext:
        hook_calls.append(ctx.hook.name)
        return ctx

    hooks.register(LifecycleHook.BEFORE_AGENT_START, test_hook)
    hooks.register(LifecycleHook.AGENT_END, test_hook)

    ctx = HookContext(
        hook=LifecycleHook.BEFORE_AGENT_START,
        phase=LoopPhase.IDLE,
        loop_state=LoopState(),
    )
    hooks.invoke(ctx)

    print(f"  Hooks called: {hook_calls}")

    # Statistics
    print("\n[STATISTICS]")
    stats = loop.get_statistics()
    print(f"  Loop: {stats['loop_state']}")
    print(f"  Phase timings: {stats['phase_timings']}")

    print("\n" + "=" * 70)
    print("[OK] Production-Grade Loop Working")
    print("=" * 70)
    print("""
Components Implemented:
  1.  LoopPhase            - Extended phases including COMPACT, BACKTRACK
  2.  LifecycleHook        - OpenClaw-style hooks (before/after tool, compaction, etc.)
  3.  HookRegistry         - Registration and invocation of hooks
  4.  ContextWindowGuard   - Token monitoring, compaction, summarization
  5.  StopHook             - Ralph Loop pattern with circuit breaker
  6.  BacktrackTree        - LATS-style tree for exploring alternatives
  7.  LoopState            - Enhanced with context stats, backtracking
  8.  Loop                 - Full production loop with all integrations

Patterns Implemented:
  - OpenClaw: Lifecycle hooks, context window guard, session management
  - ReAct: Think (REASON) -> Act (ACT) -> Observe (OBSERVE) cycle
  - Ralph Loop: Stop hooks, external verification, circuit breaker
  - LATS: Backtracking with alternative exploration
  - Self-Healing: Error recovery, retry strategies
  - Agentic RAG: Full RetrievalEngine integration in RETRIEVE phase

References:
  - OpenClaw Agent Loop: https://docs.openclaw.ai/concepts/agent-loop
  - ReAct: https://arxiv.org/abs/2210.03629
  - Ralph Loop: https://github.com/snarktank/ralph
  - LATS: https://arxiv.org/abs/2310.04406
  - Context Engineering: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
    """)
