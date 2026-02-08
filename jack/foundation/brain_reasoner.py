"""
BRAIN REASONER - Connects JackBrain to the Agent Loop

This module bridges:
- JackBrain (85K parameter transformer) for action prediction
- LLMReasoner (OpenAI-compatible) for complex reasoning
- SOTA reasoning patterns (CoT, ToT, Reflexion, etc.)

The gap this solves:
- Loop uses Reasoner protocol (sync)
- JackBrain predicts actions but doesn't implement Reasoner
- LLMReasoner is async
- Need to connect these pieces

Author: Jack Foundation
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import torch
import logging

from jack.foundation.types import Result, Ok, Err, Error, ErrorCode
from jack.foundation.state import State, Goal, GoalType
from jack.foundation.plan import Plan, PlanBuilder, Primitive
from jack.foundation.action import ActionResult, OutcomeType
from jack.foundation.memory import Pattern
from jack.foundation.retrieve import RetrievalResult

# Import JackBrain
try:
    from jack.core.jack_brain import JackBrain, JackConfig, ACTION_TYPES
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False
    JackBrain = None
    JackConfig = None
    ACTION_TYPES = {0: 'shell_run', 1: 'file_read', 2: 'file_write', 3: 'http_request', 4: 'get_state'}

# Import training utilities for state encoding
try:
    from jack.training.phase0_digital import ActionState, ActionEffect
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# ACTION HISTORY
# =============================================================================

@dataclass
class ActionHistoryEntry:
    """Single entry in action history."""
    action_type: int  # 0-4 mapping to ACTION_TYPES
    action_target: str  # path or command
    params: Dict[str, Any] = field(default_factory=dict)
    outcome: int = 2  # 0=fail, 1=success, 2=pending


class ActionHistory:
    """Maintains action history for JackBrain input."""

    def __init__(self, max_length: int = 10):
        self.max_length = max_length
        self.history: List[ActionHistoryEntry] = []

    def add(self, entry: ActionHistoryEntry) -> None:
        """Add an action to history."""
        self.history.append(entry)
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def update_last_outcome(self, outcome: int) -> None:
        """Update the outcome of the last action."""
        if self.history:
            self.history[-1].outcome = outcome

    def to_tensors(self, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert history to tensors for JackBrain."""
        device = device or torch.device('cpu')

        # Pad to max_length
        padded = self.history + [ActionHistoryEntry(5, "")] * (self.max_length - len(self.history))
        padded = padded[:self.max_length]

        action_types = torch.tensor([e.action_type for e in padded], dtype=torch.long, device=device)
        action_params = torch.zeros(self.max_length, 64, dtype=torch.float32, device=device)
        outcomes = torch.tensor([e.outcome for e in padded], dtype=torch.long, device=device)

        # Encode targets as params
        for i, entry in enumerate(padded):
            if entry.action_target:
                for j, char in enumerate(entry.action_target[:64]):
                    action_params[i, j] = (ord(char) - 32) / 95  # Normalize ASCII

        return action_types.unsqueeze(0), action_params.unsqueeze(0), outcomes.unsqueeze(0)

    def clear(self) -> None:
        """Clear history."""
        self.history.clear()


# =============================================================================
# STATE ENCODER
# =============================================================================

class StateEncoder:
    """Encodes world state to JackBrain input format."""

    def __init__(self, state_dim: int = 128):
        self.state_dim = state_dim

    def encode(self, state: State) -> torch.Tensor:
        """Encode State to tensor."""
        features = torch.zeros(1, self.state_dim, dtype=torch.float32)

        # Basic features
        features[0, 0] = 1.0  # Active flag
        features[0, 1] = 0.5  # Default confidence

        # Goal encoding
        if state.goal:
            goal_type_map = {
                GoalType.QUERY: 0.1, GoalType.TRANSFORM: 0.2,
                GoalType.CREATE: 0.3, GoalType.DELETE: 0.4,
                GoalType.ANALYZE: 0.5, GoalType.PREDICT: 0.6,
                GoalType.MONITOR: 0.7, GoalType.COMPOSITE: 0.8,
            }
            features[0, 2] = goal_type_map.get(state.goal.goal_type, 0.5)

            # Encode goal intent
            intent = state.goal.intent or ""
            for i, char in enumerate(intent[:32]):
                features[0, 10 + i] = (ord(char) - 32) / 95

        # Entity count features
        features[0, 3] = min(len(state.entities), 100) / 100
        features[0, 4] = min(len(state.files), 100) / 100

        return features

    def encode_action_state(self, action_state: 'ActionState') -> torch.Tensor:
        """Encode ActionState from training module."""
        if not TRAINING_AVAILABLE:
            return torch.zeros(1, self.state_dim, dtype=torch.float32)

        features = torch.zeros(1, self.state_dim, dtype=torch.float32)

        # Action-conditioned state (7 features)
        state_vec = action_state.to_vector()
        for i, val in enumerate(state_vec):
            features[0, i] = val

        # Encode target path
        if action_state.target:
            for i, char in enumerate(action_state.target[:64]):
                features[0, 10 + i] = (ord(char) - 32) / 95

        return features


# =============================================================================
# ACTION DECODER
# =============================================================================

class ActionDecoder:
    """Decodes JackBrain output to Primitives."""

    def decode(
        self,
        action_type: int,
        params: torch.Tensor,
        success_prob: float,
    ) -> Tuple[Optional[Primitive], float]:
        """Decode action type and params to Primitive."""
        type_name = ACTION_TYPES.get(action_type, 'get_state')

        # Decode params to string (path or command)
        param_str = self._params_to_string(params)

        # Create appropriate primitive
        if type_name == 'shell_run':
            primitive = Primitive.shell(param_str if param_str else "echo 'no command'")
        elif type_name == 'file_read':
            primitive = Primitive.read(param_str if param_str else ".")
        elif type_name == 'file_write':
            primitive = Primitive.write(param_str if param_str else "output.txt", "")
        elif type_name == 'http_request':
            primitive = Primitive.http("GET", param_str if param_str else "http://localhost")
        else:  # get_state
            primitive = Primitive.observe()

        return primitive, success_prob

    def _params_to_string(self, params: torch.Tensor) -> str:
        """Convert param tensor to string."""
        chars = []
        for val in params.squeeze().tolist():
            if val > 0.01:  # Non-zero
                char_code = int(val * 95 + 32)
                if 32 <= char_code <= 126:  # Printable ASCII
                    chars.append(chr(char_code))
        return ''.join(chars).strip()


# =============================================================================
# BRAIN REASONER
# =============================================================================

class BrainReasoner:
    """
    Reasoner that uses JackBrain for action prediction.

    Implements the Reasoner protocol expected by Loop.
    Falls back to pattern matching when brain is uncertain.

    Usage:
        brain = JackBrain.load("checkpoints/jack_final.pt")
        reasoner = BrainReasoner(brain)
        loop = Loop(reasoner=reasoner)
    """

    def __init__(
        self,
        brain: Optional['JackBrain'] = None,
        confidence_threshold: float = 0.6,
        device: Optional[torch.device] = None,
    ):
        self.brain = brain
        self.confidence_threshold = confidence_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.state_encoder = StateEncoder()
        self.action_decoder = ActionDecoder()
        self.action_history = ActionHistory()

        if self.brain:
            self.brain = self.brain.to(self.device)
            self.brain.eval()

    @classmethod
    def load(cls, checkpoint_path: str, **kwargs) -> 'BrainReasoner':
        """Load BrainReasoner from checkpoint."""
        if not BRAIN_AVAILABLE:
            raise ImportError("JackBrain not available - install torch")

        device = kwargs.get('device', torch.device('cpu'))
        brain = JackBrain(JackConfig())
        checkpoint = torch.load(checkpoint_path, map_location=device)
        brain.load_state_dict(checkpoint['model'])

        return cls(brain=brain, device=device, **kwargs)

    def plan(
        self,
        goal: Goal,
        state: State,
        patterns: List[Tuple[Pattern, float]],
        retrieval_context: Optional[RetrievalResult] = None,
    ) -> Result[Plan, Error]:
        """Generate plan using brain prediction."""
        # Clear history for new goal
        self.action_history.clear()

        # Encode state
        state_tensor = self.state_encoder.encode(state).to(self.device)

        # Get history tensors
        action_types, action_params, outcomes = self.action_history.to_tensors(self.device)

        # Predict with brain
        if self.brain:
            action_type, params, success_prob = self.brain.predict_action(
                state_tensor, action_types, action_params, outcomes
            )
        else:
            # Fallback: use patterns
            return self._plan_from_patterns(goal, state, patterns, retrieval_context)

        # Decode action
        primitive, confidence = self.action_decoder.decode(action_type, params, success_prob)

        # Check confidence
        if confidence < self.confidence_threshold:
            logger.info(f"Brain confidence {confidence:.2f} below threshold, using patterns")
            return self._plan_from_patterns(goal, state, patterns, retrieval_context)

        # Build plan
        builder = PlanBuilder(goal, state)
        builder.add_step(
            f"Brain-predicted: {ACTION_TYPES[action_type]}",
            primitives=[primitive] if primitive else []
        )

        return Ok(builder.build())

    def _plan_from_patterns(
        self,
        goal: Goal,
        state: State,
        patterns: List[Tuple[Pattern, float]],
        retrieval_context: Optional[RetrievalResult] = None,
    ) -> Result[Plan, Error]:
        """Fallback planning using patterns."""
        builder = PlanBuilder(goal, state)

        # Use retrieval context
        if retrieval_context and retrieval_context.chunks:
            for chunk in retrieval_context.chunks[:3]:
                if "command" in chunk.content.lower():
                    builder.add_step(
                        f"From retrieval: {chunk.content[:50]}",
                        primitives=[Primitive.shell("echo 'Using retrieved context'")]
                    )
                    return Ok(builder.build())

        # Use patterns
        if patterns:
            best_pattern, similarity = patterns[0]
            if similarity > 0.1:
                action_data = best_pattern.action_data
                if "command" in action_data:
                    builder.add_step(
                        f"From pattern: {action_data['command'][:50]}",
                        primitives=[Primitive.shell(action_data["command"])]
                    )
                    return Ok(builder.build())

        return Err(Error(ErrorCode.PLANNING_FAILED, "No patterns and brain uncertain"))

    def reason(self, prompt: str) -> Result[str, Error]:
        """Basic reasoning - just echo for brain-based."""
        return Ok(f"Brain processed: {prompt[:100]}")

    def reason_json(self, prompt: str) -> Result[Dict[str, Any], Error]:
        """JSON reasoning - return simple structure."""
        return Ok({"response": prompt[:100], "source": "brain"})

    def decide_next_action(
        self,
        state: State,
        plan: Plan,
        last_result: Optional[ActionResult],
    ) -> Result[Optional[Primitive], Error]:
        """Decide next action using brain."""
        # Update history with last result
        if last_result:
            outcome = 1 if last_result.is_success else 0
            self.action_history.update_last_outcome(outcome)

        # Get pending primitives from plan
        primitives = plan.get_pending_primitives()
        if primitives:
            # Add to history
            primitive = primitives[0]
            entry = ActionHistoryEntry(
                action_type=self._primitive_to_type(primitive),
                action_target=self._primitive_to_target(primitive),
            )
            self.action_history.add(entry)
            return Ok(primitive)

        # Use brain to predict next action
        if self.brain:
            state_tensor = self.state_encoder.encode(state).to(self.device)
            action_types, action_params, outcomes = self.action_history.to_tensors(self.device)

            action_type, params, success_prob = self.brain.predict_action(
                state_tensor, action_types, action_params, outcomes
            )

            if success_prob >= self.confidence_threshold:
                primitive, _ = self.action_decoder.decode(action_type, params, success_prob)
                return Ok(primitive)

        return Ok(None)

    def _primitive_to_type(self, primitive: Primitive) -> int:
        """Convert Primitive to action type index."""
        type_map = {'SHELL': 0, 'READ': 1, 'WRITE': 2, 'HTTP': 3, 'OBSERVE': 4}
        return type_map.get(primitive.type.name, 4)

    def _primitive_to_target(self, primitive: Primitive) -> str:
        """Extract target from Primitive."""
        if primitive.type.name == 'SHELL':
            return primitive.params.get('command', '')
        elif primitive.type.name in ('READ', 'WRITE'):
            return primitive.params.get('path', '')
        elif primitive.type.name == 'HTTP':
            return primitive.params.get('url', '')
        return ''

    def should_continue(
        self,
        state: State,
        plan: Plan,
        result: ActionResult,
    ) -> bool:
        """Should we continue after this result?"""
        return result.outcome not in (OutcomeType.BLOCKED, OutcomeType.ERROR)

    def is_goal_achieved(
        self,
        goal: Goal,
        state: State,
        results: List[ActionResult],
    ) -> bool:
        """Check if goal is achieved."""
        return bool(results) and all(r.is_success for r in results)

    def suggest_recovery(
        self,
        error: Error,
        context: Dict[str, Any],
    ) -> Result[Optional[Primitive], Error]:
        """Suggest recovery action."""
        if error.code == ErrorCode.TIMEOUT:
            return Ok(Primitive.shell("echo 'Recovery from timeout'"))
        return Ok(None)


# =============================================================================
# LLM REASONER (Sync Wrapper)
# =============================================================================

class SyncLLMReasoner:
    """
    Synchronous wrapper for LLMReasoner.

    Bridges the async LLMReasoner to the sync Reasoner protocol.

    Usage:
        from jack.foundation.llm import LLMReasoner, LLMConfig

        async_reasoner = LLMReasoner(LLMConfig(base_url="http://localhost:8080/v1"))
        sync_reasoner = SyncLLMReasoner(async_reasoner)
        loop = Loop(reasoner=sync_reasoner)
    """

    def __init__(self, llm_reasoner: Any):
        self.llm = llm_reasoner
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop

    def _run_async(self, coro):
        """Run coroutine synchronously."""
        loop = self._get_loop()
        try:
            return asyncio.get_running_loop().run_until_complete(coro)
        except RuntimeError:
            return loop.run_until_complete(coro)

    def plan(
        self,
        goal: Goal,
        state: State,
        patterns: List[Tuple[Pattern, float]],
        retrieval_context: Optional[RetrievalResult] = None,
    ) -> Result[Plan, Error]:
        """Generate plan using LLM."""
        # Build prompt for planning
        prompt = self._build_plan_prompt(goal, state, patterns, retrieval_context)

        result = self._run_async(self.llm.reason_json(prompt))

        if result.is_err():
            return Err(result.unwrap_err())

        data = result.unwrap()
        return self._parse_plan(data, goal, state)

    def _build_plan_prompt(
        self,
        goal: Goal,
        state: State,
        patterns: List[Tuple[Pattern, float]],
        retrieval_context: Optional[RetrievalResult],
    ) -> str:
        """Build prompt for plan generation."""
        prompt_parts = [
            "You are a planning agent. Generate a plan to achieve the goal.",
            "",
            f"GOAL: {goal.intent}",
            f"GOAL TYPE: {goal.goal_type.name}",
            "",
        ]

        if retrieval_context and retrieval_context.chunks:
            prompt_parts.append("CONTEXT:")
            for chunk in retrieval_context.chunks[:3]:
                prompt_parts.append(f"  - {chunk.content[:100]}")
            prompt_parts.append("")

        if patterns:
            prompt_parts.append("SIMILAR PATTERNS:")
            for pattern, score in patterns[:3]:
                prompt_parts.append(f"  - [{score:.2f}] {pattern.pattern_type}: {pattern.description[:50]}")
            prompt_parts.append("")

        prompt_parts.extend([
            "Available actions: shell_run, file_read, file_write, http_request, get_state",
            "",
            'Respond with JSON: {"steps": [{"description": "...", "action": "shell_run|file_read|...", "target": "command or path"}]}',
        ])

        return "\n".join(prompt_parts)

    def _parse_plan(
        self,
        data: Dict[str, Any],
        goal: Goal,
        state: State,
    ) -> Result[Plan, Error]:
        """Parse LLM response into Plan."""
        builder = PlanBuilder(goal, state)

        steps = data.get("steps", [])
        for step_data in steps:
            description = step_data.get("description", "LLM step")
            action = step_data.get("action", "get_state")
            target = step_data.get("target", "")

            if action == "shell_run":
                primitive = Primitive.shell(target)
            elif action == "file_read":
                primitive = Primitive.read(target)
            elif action == "file_write":
                primitive = Primitive.write(target, step_data.get("content", ""))
            elif action == "http_request":
                primitive = Primitive.http("GET", target)
            else:
                primitive = Primitive.observe()

            builder.add_step(description, primitives=[primitive])

        return Ok(builder.build())

    def reason(self, prompt: str) -> Result[str, Error]:
        """Synchronous reasoning."""
        return self._run_async(self.llm.reason(prompt))

    def reason_json(self, prompt: str) -> Result[Dict[str, Any], Error]:
        """Synchronous JSON reasoning."""
        return self._run_async(self.llm.reason_json(prompt))

    def decide_next_action(
        self,
        state: State,
        plan: Plan,
        last_result: Optional[ActionResult],
    ) -> Result[Optional[Primitive], Error]:
        """Get next action from plan."""
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
        """Check if goal is achieved."""
        return bool(results) and all(r.is_success for r in results)

    def suggest_recovery(
        self,
        error: Error,
        context: Dict[str, Any],
    ) -> Result[Optional[Primitive], Error]:
        """Use LLM to suggest recovery."""
        prompt = f"""Error occurred: {error.code.name}: {error.message}
Context: {context}

Suggest a recovery action. Respond with JSON: {{"action": "shell_run|file_read|...", "target": "..."}}
Or {{"action": "none"}} if no recovery possible."""

        result = self._run_async(self.llm.reason_json(prompt))

        if result.is_err():
            return Ok(None)

        data = result.unwrap()
        action = data.get("action", "none")

        if action == "none":
            return Ok(None)

        target = data.get("target", "")

        if action == "shell_run":
            return Ok(Primitive.shell(target))
        elif action == "file_read":
            return Ok(Primitive.read(target))

        return Ok(None)


# =============================================================================
# HYBRID REASONER
# =============================================================================

class HybridReasoner:
    """
    Combines BrainReasoner with LLM for best of both worlds.

    Strategy:
    - Fast actions: Use JackBrain (no LLM latency)
    - Complex reasoning: Use LLM
    - Uncertain situations: Consult LLM

    Usage:
        hybrid = HybridReasoner(
            brain=JackBrain.load("checkpoint.pt"),
            llm=LLMReasoner(LLMConfig())
        )
        loop = Loop(reasoner=hybrid)
    """

    def __init__(
        self,
        brain: Optional['JackBrain'] = None,
        llm: Any = None,
        brain_confidence_threshold: float = 0.7,
    ):
        self.brain_reasoner = BrainReasoner(brain, brain_confidence_threshold) if brain else None
        self.llm_reasoner = SyncLLMReasoner(llm) if llm else None
        self.threshold = brain_confidence_threshold

    def plan(
        self,
        goal: Goal,
        state: State,
        patterns: List[Tuple[Pattern, float]],
        retrieval_context: Optional[RetrievalResult] = None,
    ) -> Result[Plan, Error]:
        """Try brain first, fall back to LLM."""
        if self.brain_reasoner:
            result = self.brain_reasoner.plan(goal, state, patterns, retrieval_context)
            if result.is_ok():
                return result

        if self.llm_reasoner:
            return self.llm_reasoner.plan(goal, state, patterns, retrieval_context)

        return Err(Error(ErrorCode.PLANNING_FAILED, "No reasoner available"))

    def reason(self, prompt: str) -> Result[str, Error]:
        """Use LLM for complex reasoning."""
        if self.llm_reasoner:
            return self.llm_reasoner.reason(prompt)
        if self.brain_reasoner:
            return self.brain_reasoner.reason(prompt)
        return Err(Error(ErrorCode.REASONING_FAILED, "No reasoner available"))

    def reason_json(self, prompt: str) -> Result[Dict[str, Any], Error]:
        """Use LLM for JSON reasoning."""
        if self.llm_reasoner:
            return self.llm_reasoner.reason_json(prompt)
        if self.brain_reasoner:
            return self.brain_reasoner.reason_json(prompt)
        return Err(Error(ErrorCode.REASONING_FAILED, "No reasoner available"))

    def decide_next_action(
        self,
        state: State,
        plan: Plan,
        last_result: Optional[ActionResult],
    ) -> Result[Optional[Primitive], Error]:
        """Use brain for fast action decisions."""
        if self.brain_reasoner:
            return self.brain_reasoner.decide_next_action(state, plan, last_result)
        if self.llm_reasoner:
            return self.llm_reasoner.decide_next_action(state, plan, last_result)
        return Ok(None)

    def should_continue(
        self,
        state: State,
        plan: Plan,
        result: ActionResult,
    ) -> bool:
        """Continue unless blocked."""
        return result.outcome not in (OutcomeType.BLOCKED, OutcomeType.ERROR)

    def is_goal_achieved(
        self,
        goal: Goal,
        state: State,
        results: List[ActionResult],
    ) -> bool:
        """Check goal achievement."""
        return bool(results) and all(r.is_success for r in results)

    def suggest_recovery(
        self,
        error: Error,
        context: Dict[str, Any],
    ) -> Result[Optional[Primitive], Error]:
        """Use LLM for recovery suggestions."""
        if self.llm_reasoner:
            return self.llm_reasoner.suggest_recovery(error, context)
        if self.brain_reasoner:
            return self.brain_reasoner.suggest_recovery(error, context)
        return Ok(None)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_brain_reasoner(checkpoint_path: str = "checkpoints/jack_final.pt") -> BrainReasoner:
    """Create BrainReasoner from checkpoint."""
    try:
        return BrainReasoner.load(checkpoint_path)
    except Exception as e:
        logger.warning(f"Failed to load brain from {checkpoint_path}: {e}")
        return BrainReasoner(brain=None)


def create_llm_reasoner(base_url: str = "http://localhost:8080/v1") -> SyncLLMReasoner:
    """Create SyncLLMReasoner."""
    try:
        from jack.foundation.llm import LLMReasoner, LLMConfig
        llm = LLMReasoner(LLMConfig(base_url=base_url))
        return SyncLLMReasoner(llm)
    except Exception as e:
        logger.warning(f"Failed to create LLM reasoner: {e}")
        raise


def create_hybrid_reasoner(
    checkpoint_path: str = "checkpoints/jack_final.pt",
    llm_base_url: str = "http://localhost:8080/v1",
) -> HybridReasoner:
    """Create HybridReasoner with brain and LLM."""
    brain = None
    llm = None

    # Try to load brain
    try:
        if BRAIN_AVAILABLE:
            import torch
            brain = JackBrain(JackConfig())
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            brain.load_state_dict(checkpoint['model'])
            logger.info(f"Loaded brain from {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Failed to load brain: {e}")

    # Try to create LLM
    try:
        from jack.foundation.llm import LLMReasoner, LLMConfig
        llm = LLMReasoner(LLMConfig(base_url=llm_base_url))
        logger.info(f"Created LLM reasoner at {llm_base_url}")
    except Exception as e:
        logger.warning(f"Failed to create LLM reasoner: {e}")

    return HybridReasoner(brain=brain, llm=llm)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BRAIN REASONER - JackBrain + LLM Integration")
    print("=" * 60)

    # Test BrainReasoner (without trained model)
    print("\n[TEST 1] BrainReasoner (no model)")
    reasoner = BrainReasoner(brain=None)

    from jack.foundation.state import Goal, GoalType, StateBuilder
    goal = Goal(intent="List files", goal_type=GoalType.QUERY)
    state = StateBuilder().with_goal("List files", GoalType.QUERY).build()

    result = reasoner.plan(goal, state, [])
    print(f"  Plan result: {result}")

    print(f"  Reason: {reasoner.reason('Test prompt')}")

    # Test ActionHistory
    print("\n[TEST 2] ActionHistory")
    history = ActionHistory()
    history.add(ActionHistoryEntry(0, "ls -la", outcome=1))
    history.add(ActionHistoryEntry(1, "readme.txt", outcome=1))

    types, params, outcomes = history.to_tensors()
    print(f"  Types shape: {types.shape}")
    print(f"  Params shape: {params.shape}")
    print(f"  Outcomes shape: {outcomes.shape}")

    # Test StateEncoder
    print("\n[TEST 3] StateEncoder")
    encoder = StateEncoder()
    test_state = StateBuilder().with_goal("Test", GoalType.QUERY).build()
    encoded = encoder.encode(test_state)
    print(f"  Encoded state shape: {encoded.shape}")

    # Test ActionDecoder
    print("\n[TEST 4] ActionDecoder")
    decoder = ActionDecoder()
    import torch
    params = torch.zeros(64)
    params[:5] = torch.tensor([0.7, 0.8, 0.32, 0.0, 0.0])  # "ls"

    primitive, conf = decoder.decode(0, params, 0.9)
    print(f"  Decoded: {primitive}, confidence: {conf}")

    print("\n" + "=" * 60)
    print("Integration Complete!")
    print("""
Components:
  1. BrainReasoner    - Uses JackBrain for action prediction
  2. SyncLLMReasoner  - Sync wrapper for async LLMReasoner
  3. HybridReasoner   - Combines brain + LLM

Usage:
  from jack.foundation.brain_reasoner import create_hybrid_reasoner
  from jack.foundation.loop import Loop

  reasoner = create_hybrid_reasoner()
  loop = Loop(reasoner=reasoner)
  loop.run(goal)
""")
    print("=" * 60)
