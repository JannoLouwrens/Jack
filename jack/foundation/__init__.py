"""
JACK FOUNDATION - Robust Building Blocks for Intelligent Systems

This module implements the core building blocks for a robust, scalable AI agent:

1. TYPES      - Result, Option, and core type system (Rust-inspired)
2. STATE      - Goal-conditioned state representation
3. PLAN       - Hierarchical task network planning
4. ACTION     - Observable action primitives
5. MEMORY     - Pattern memory with embeddings
6. VERIFY     - Verification layer
7. PERCEIVE   - LLM-First intelligent perception
8. RETRIEVE   - Agentic RAG (multi-source, hybrid search, Tool RAG)
9. REASON     - Advanced LLM reasoning (CoT, ToT, Reflexion, ReAct, LATS)
10. ROBUST    - Production-grade LLM infrastructure (reusable by all stages)
11. LOOP      - Main orchestration loop
12. LLM       - Real LLM integration (llama.cpp, OpenAI, Anthropic)

Design Principles:
- LLM-First: The LLM makes ALL intelligent decisions
- No Brittle Approaches: No regex, no keyword matching, no hardcoded rules
- Functional Core, Imperative Shell (pure logic, isolated side effects)
- Explicit error handling (Result types, not exceptions)
- Immutable data structures (frozen dataclasses)
- Clear contracts (Protocol-based interfaces)
- Composition over inheritance

References:
- Functional Core, Imperative Shell: https://kennethlange.com/functional-core-imperative-shell/
- Result types: https://github.com/rustedpy/result
- HTN Planning: https://en.wikipedia.org/wiki/Hierarchical_task_network
- GoalAct: https://arxiv.org/abs/2504.16563 (Hierarchical goal decomposition)
- Agentic RAG: https://arxiv.org/abs/2501.09136 (Memory-augmented agents)
"""

from jack.foundation.types import Result, Ok, Err, Option, Some, NONE, Error, ErrorCode
from jack.foundation.state import (
    State, StateBuilder, Goal, GoalType, Entity, EntityType, Constraint, Context, Observation,
    # Relevance Scoring (LLM-First)
    RelevanceScorer, SimpleRelevanceScorer,
)
from jack.foundation.plan import (
    Plan, PlanBuilder, Step, StepType, Primitive, PrimitiveType, Checkpoint,
    # GoalAct Dynamic Planning
    Skill, GlobalPlanner, Replanner, PlanStatus,
    # Factory functions
    create_plan_with_llm, create_simple_plan,
)
from jack.foundation.action import (
    ActionResult, Executor, OutcomeType, StateDelta,
    # Resource Tracking
    ResourceTracker,
    # Action Handlers
    ShellActionHandler, FileReadHandler, FileWriteHandler, HttpActionHandler,
)
from jack.foundation.memory import (
    # A-MEM Agentic Memory
    AgenticMemory, MemoryNote, MemoryType, MemoryStore,
    MemoryLinker, MemoryConsolidator, ActiveForgetter,
    # Embedding
    EmbeddingProvider, SimpleEmbedding,
    # Backward compatibility
    Memory, Pattern, PatternStore,
)
from jack.foundation.verify import (
    # Core
    Verifier, Verdict, VerdictType, VerificationReport,
    # Constitutional AI
    ConstitutionalVerifier, Constitution, ConstitutionalPrinciple, CritiqueResult,
    # Semantic Safety
    SemanticSafetyVerifier, SafetyAssessment, RiskLevel,
    # Exec Approvals (OpenClaw)
    ExecApprovalChecker, ExecApproval, AuthorizationLevel, PermissionPolicy,
    # Guard Agent
    GuardAgent,
    # Goal Verification
    GoalAchievementVerifier, GoalVerificationResult,
    # Semantic Validation (NLI)
    SemanticValidator,
    # Human-in-the-Loop
    HumanApprovalCallback, DefaultHumanApproval,
    # Factory functions
    create_verifier, create_production_verifier,
    # Backward compatibility
    SafetyCheck,
    # Reasoner protocol
    Reasoner as VerifyReasoner,
    SimpleReasoner as VerifySimpleReasoner,
)

# Perceive (LLM-First with SOTA Patterns)
from jack.foundation.perceive import (
    # Main engines
    IntelligentPerceptionEngine,
    RobustPerceptionEngine,
    AdaptivePerceptionEngine,  # SOTA: Iterative refinement
    # Factory functions
    create_perception_engine,
    create_robust_perception_engine,
    create_adaptive_perception_engine,  # SOTA factory
    perceive_intelligent,
    # Data structures
    PerceptionResult,
    PerceptionDomain,
    GoalDecomposition,
    ScoredEntity,
    PerceptionGap,
    RawEntity,
    RawPerceptionData,
    RawDataCollector,
    VerificationStatus,
    # SOTA: Confidence Calibration
    PerceptionCalibrator,
    CalibrationStats,
    AdaptivePerceptionConfig,
    # SOTA: Perception-Action Feedback
    PerceptionActionFeedback,
    EntityActionOutcome,
    # Protocol
    Reasoner as PerceiveReasoner,
    # Testing
    SimpleReasoner,
    # Backward compatibility aliases
    PerceptionEngine,  # Alias for IntelligentPerceptionEngine
    GoalAnalyzer,      # Deprecated (None)
    Perceiver,         # Alias for RawDataCollector
)

# Retrieve (Agentic RAG)
from jack.foundation.retrieve import (
    # Core data structures
    RetrievalSource,
    Chunk, ChunkingConfig, ChunkingStrategy,
    RetrievalQuery, RetrievalResult,
    # Retrievers
    VectorIndex, VectorRetriever,
    KeywordRetriever,
    HybridConfig, HybridRetriever,
    ToolDescription, ToolRetriever,
    # Query Processing
    QueryPlanner,
    ShouldRetrieveDecider,
    # Reranking
    SimpleReranker as RetrieveSimpleReranker,
    LLMReranker,
    # Fusion
    ResultFusion,
    # Main Engine
    RetrievalEngineConfig, RetrievalEngine,
    # Factory functions
    create_retrieval_engine, create_simple_retrieval_engine,
    # Protocol
    Reasoner as RetrieveReasoner,
    Retriever, Reranker,
)

# Reason (Advanced LLM Reasoning Patterns)
from jack.foundation.reason import (
    # Core reasoning result
    ReasoningResult, ReasoningStep,
    # Chain of Thought
    ChainOfThought, CoTConfig,
    # Self-Consistency
    SelfConsistency, SelfConsistencyConfig,
    # Tree of Thoughts
    TreeOfThoughts, ToTConfig, ThoughtNode,
    # Reflexion
    Reflexion, ReflexionConfig, ReflexionMemory,
    # Least-to-Most
    LeastToMost, LeastToMostConfig,
    # ReAct
    ReActReasoner, ReActConfig,
    # Sketch of Thought
    SketchOfThought, SketchConfig,
    # Composite Reasoner
    CompositeReasoner, CompositeConfig, ReasoningStrategy,
    # Factory functions
    create_cot_reasoner, create_self_consistent_reasoner,
    create_tot_reasoner, create_reflexion_reasoner,
    create_react_reasoner, create_composite_reasoner,
    # Reasoner Protocol
    Reasoner as ReasonReasoner,
)

# Robust Infrastructure (reusable by ALL stages)
from jack.foundation.robust import (
    # Validation
    ResponseSchema, FieldSchema, Schemas, ValidationError,
    # Retry
    RetryConfig, RetryStrategy, PromptRephraser,
    # Learning
    LearningLoop, Outcome,
    # Prompts
    PromptTemplate, PromptExample, PromptLibrary,
    # Observability
    ObservabilityLayer, DecisionTrace,
    # Calibration
    ConfidenceCalibrator,
    # Circuit Breaker
    CircuitBreaker, CircuitState, CircuitBreakerConfig,
    # Cache
    ResponseCache, CacheEntry,
    # Rate Limiting
    RateLimiter, RateLimitConfig,
    # Deduplication
    RequestDeduplicator,
    # Fallback
    FallbackProvider,
    # Wrapper
    RobustReasoner, ReasonerResponse,
    # Semantic Cache (GPTCache pattern)
    SemanticCache, SemanticCacheConfig,
    # Multi-Provider Failover (Portkey pattern)
    ProviderConfig, ProviderRegistry, MultiProviderReasoner,
    # Tool Registry (Function Calling)
    ToolDefinition, ToolCall, ToolResult, ToolRegistry,
    # Self-Healing (CRITIC pattern)
    VerificationResult, SelfHealingEngine,
    # Semantic Validation (Guardrails)
    GuardrailConfig, SemanticValidator,
    # Ultimate Wrapper
    UltimateRobustReasoner,
    # OpenClaw-Inspired Patterns
    MessageQueue, QueueConfig,
    Session, SessionManager,
    ConnectionManager, ConnectionConfig, ConnectionState,
    ChannelAdapter, StandardMessage, ChannelRouter,
)

# Loop (Production-Grade Orchestration)
from jack.foundation.loop import (
    # Main loop
    Loop, LoopConfig, LoopState, LoopPhase, LoopEvent, LoopEventData,
    # Reasoner protocol
    Reasoner, SimpleReasoner as LoopSimpleReasoner,
    # Context management (OpenClaw pattern)
    ContextWindowGuard, ContextWindowConfig, ContextStats,
    # Stop hook (Ralph Loop pattern)
    StopHook, StopHookConfig,
    # Backtracking (LATS pattern)
    BacktrackTree, BacktrackNode,
    # Lifecycle hooks (OpenClaw pattern)
    LifecycleHook, HookRegistry, HookContext, HookCallback,
    # Factory functions
    run_goal, create_production_loop, create_loop_with_logging,
)

# LLM (Real LLM Integration)
from jack.foundation.llm import (
    # Core reasoner classes
    LLMReasoner, LLMConfig, LLMResponse,
    AnthropicReasoner,
    MultiProviderReasoner,
    SyncReasonerWrapper,
    # Unified protocol
    Reasoner as LLMReasonerProtocol,
    # Factory functions
    create_local_reasoner,
    create_openai_reasoner,
    create_anthropic_reasoner,
    create_failover_reasoner,
)

# Brain Reasoner (JackBrain + LLM Integration)
from jack.foundation.brain_reasoner import (
    # Core reasoners
    BrainReasoner,
    SyncLLMReasoner,
    HybridReasoner,
    # Support classes
    ActionHistory, ActionHistoryEntry,
    StateEncoder, ActionDecoder,
    # Factory functions
    create_brain_reasoner,
    create_llm_reasoner as create_sync_llm_reasoner,
    create_hybrid_reasoner,
)

__all__ = [
    # Types
    'Result', 'Ok', 'Err', 'Option', 'Some', 'NONE', 'Error', 'ErrorCode',
    # State
    'State', 'StateBuilder', 'Goal', 'GoalType', 'Entity', 'EntityType', 'Constraint', 'Context', 'Observation',
    'RelevanceScorer', 'SimpleRelevanceScorer',
    # Plan (GoalAct Dynamic Planning)
    'Plan', 'PlanBuilder', 'Step', 'StepType', 'Primitive', 'PrimitiveType', 'Checkpoint',
    'Skill', 'GlobalPlanner', 'Replanner', 'PlanStatus',
    'create_plan_with_llm', 'create_simple_plan',
    # Action
    'ActionResult', 'Executor', 'OutcomeType', 'StateDelta',
    'ResourceTracker', 'ShellActionHandler', 'FileReadHandler', 'FileWriteHandler', 'HttpActionHandler',
    # Memory (A-MEM Agentic Memory)
    'AgenticMemory', 'MemoryNote', 'MemoryType', 'MemoryStore',
    'MemoryLinker', 'MemoryConsolidator', 'ActiveForgetter',
    'EmbeddingProvider', 'SimpleEmbedding',
    'Memory', 'Pattern', 'PatternStore',  # Backward compatibility
    # Verify (LLM-First)
    'Verifier', 'Verdict', 'VerdictType', 'VerificationReport',
    # Constitutional AI
    'ConstitutionalVerifier', 'Constitution', 'ConstitutionalPrinciple', 'CritiqueResult',
    # Semantic Safety
    'SemanticSafetyVerifier', 'SafetyAssessment', 'RiskLevel',
    # Exec Approvals (OpenClaw)
    'ExecApprovalChecker', 'ExecApproval', 'AuthorizationLevel', 'PermissionPolicy',
    # Guard Agent
    'GuardAgent',
    # Goal Verification
    'GoalAchievementVerifier', 'GoalVerificationResult',
    # Semantic Validation (NLI)
    'SemanticValidator',
    # Human-in-the-Loop
    'HumanApprovalCallback', 'DefaultHumanApproval',
    # Factory functions
    'create_verifier', 'create_production_verifier',
    # Backward compatibility
    'SafetyCheck',
    # Verify Reasoner
    'VerifyReasoner', 'VerifySimpleReasoner',
    # Perceive (LLM-First with SOTA Patterns)
    'IntelligentPerceptionEngine', 'RobustPerceptionEngine', 'AdaptivePerceptionEngine',
    'create_perception_engine', 'create_robust_perception_engine', 'create_adaptive_perception_engine',
    'perceive_intelligent',
    'PerceptionResult', 'PerceptionDomain', 'GoalDecomposition',
    'ScoredEntity', 'PerceptionGap', 'RawEntity', 'RawPerceptionData',
    'RawDataCollector', 'VerificationStatus',
    # SOTA: Confidence Calibration
    'PerceptionCalibrator', 'CalibrationStats', 'AdaptivePerceptionConfig',
    # SOTA: Perception-Action Feedback
    'PerceptionActionFeedback', 'EntityActionOutcome',
    'PerceiveReasoner', 'SimpleReasoner',
    'PerceptionEngine', 'GoalAnalyzer', 'Perceiver',  # Backward compatibility
    # Retrieve (Agentic RAG)
    'RetrievalSource',
    'Chunk', 'ChunkingConfig', 'ChunkingStrategy',
    'RetrievalQuery', 'RetrievalResult',
    'VectorIndex', 'VectorRetriever',
    'KeywordRetriever',
    'HybridConfig', 'HybridRetriever',
    'ToolDescription', 'ToolRetriever',
    'QueryPlanner', 'ShouldRetrieveDecider',
    'RetrieveSimpleReranker', 'LLMReranker',
    'ResultFusion',
    'RetrievalEngineConfig', 'RetrievalEngine',
    'create_retrieval_engine', 'create_simple_retrieval_engine',
    'RetrieveReasoner', 'Retriever', 'Reranker',
    # Reason (Advanced LLM Reasoning Patterns)
    'ReasoningResult', 'ReasoningStep',
    'ChainOfThought', 'CoTConfig',
    'SelfConsistency', 'SelfConsistencyConfig',
    'TreeOfThoughts', 'ToTConfig', 'ThoughtNode',
    'Reflexion', 'ReflexionConfig', 'ReflexionMemory',
    'LeastToMost', 'LeastToMostConfig',
    'ReActReasoner', 'ReActConfig',
    'SketchOfThought', 'SketchConfig',
    'CompositeReasoner', 'CompositeConfig', 'ReasoningStrategy',
    'create_cot_reasoner', 'create_self_consistent_reasoner',
    'create_tot_reasoner', 'create_reflexion_reasoner',
    'create_react_reasoner', 'create_composite_reasoner',
    'ReasonReasoner',
    # Robust Infrastructure (reusable by all stages)
    'ResponseSchema', 'FieldSchema', 'Schemas', 'ValidationError',
    'RetryConfig', 'RetryStrategy', 'PromptRephraser',
    'LearningLoop', 'Outcome',
    'PromptTemplate', 'PromptExample', 'PromptLibrary',
    'ObservabilityLayer', 'DecisionTrace',
    'ConfidenceCalibrator',
    'CircuitBreaker', 'CircuitState', 'CircuitBreakerConfig',
    'ResponseCache', 'CacheEntry',
    'RateLimiter', 'RateLimitConfig',
    'RequestDeduplicator',
    'FallbackProvider',
    'RobustReasoner', 'ReasonerResponse',
    # Semantic Cache (GPTCache pattern)
    'SemanticCache', 'SemanticCacheConfig',
    # Multi-Provider Failover (Portkey pattern)
    'ProviderConfig', 'ProviderRegistry', 'MultiProviderReasoner',
    # Tool Registry (Function Calling)
    'ToolDefinition', 'ToolCall', 'ToolResult', 'ToolRegistry',
    # Self-Healing (CRITIC pattern)
    'VerificationResult', 'SelfHealingEngine',
    # Semantic Validation (Guardrails)
    'GuardrailConfig', 'SemanticValidator',
    # Ultimate Wrapper
    'UltimateRobustReasoner',
    # OpenClaw-Inspired Patterns
    'MessageQueue', 'QueueConfig',
    'Session', 'SessionManager',
    'ConnectionManager', 'ConnectionConfig', 'ConnectionState',
    'ChannelAdapter', 'StandardMessage', 'ChannelRouter',
    # Loop (Production-Grade Orchestration)
    'Loop', 'LoopConfig', 'LoopState', 'LoopPhase', 'LoopEvent', 'LoopEventData',
    'Reasoner', 'LoopSimpleReasoner',
    'ContextWindowGuard', 'ContextWindowConfig', 'ContextStats',
    'StopHook', 'StopHookConfig',
    'BacktrackTree', 'BacktrackNode',
    'LifecycleHook', 'HookRegistry', 'HookContext', 'HookCallback',
    'run_goal', 'create_production_loop', 'create_loop_with_logging',
    # LLM (Real LLM Integration)
    'LLMReasoner', 'LLMConfig', 'LLMResponse',
    'AnthropicReasoner', 'MultiProviderReasoner', 'SyncReasonerWrapper',
    'LLMReasonerProtocol',
    'create_local_reasoner', 'create_openai_reasoner',
    'create_anthropic_reasoner', 'create_failover_reasoner',
    # Brain Reasoner (JackBrain + LLM Integration)
    'BrainReasoner', 'SyncLLMReasoner', 'HybridReasoner',
    'ActionHistory', 'ActionHistoryEntry',
    'StateEncoder', 'ActionDecoder',
    'create_brain_reasoner', 'create_sync_llm_reasoner', 'create_hybrid_reasoner',
]
