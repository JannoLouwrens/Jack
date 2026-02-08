"""
JACK FOUNDATION - Robust Building Blocks for Intelligent Systems

This module implements the core building blocks for a robust, scalable AI agent.
Uses lazy imports to avoid requiring torch for server-only deployments.
"""

# Light imports that don't require torch
from jack.foundation.types import Result, Ok, Err, Option, Some, NONE, Error, ErrorCode
from jack.foundation.state import (
    State, StateBuilder, Goal, GoalType, Entity, EntityType, Constraint, Context, Observation,
    RelevanceScorer, SimpleRelevanceScorer,
)
from jack.foundation.plan import (
    Plan, PlanBuilder, Step, StepType, Primitive, PrimitiveType, Checkpoint,
    Skill, GlobalPlanner, Replanner, PlanStatus,
    create_plan_with_llm, create_simple_plan,
)
from jack.foundation.action import (
    ActionResult, Executor, OutcomeType, StateDelta,
    ResourceTracker,
    ShellActionHandler, FileReadHandler, FileWriteHandler, HttpActionHandler,
)

# Lazy imports for heavier modules (torch-dependent or optional)
_lazy_imports = {}


def __getattr__(name):
    """Lazy loading for optional/heavy modules."""
    if name in _lazy_imports:
        return _lazy_imports[name]

    # Memory module
    if name in ('AgenticMemory', 'MemoryNote', 'MemoryType', 'MemoryStore',
                'MemoryLinker', 'MemoryConsolidator', 'ActiveForgetter',
                'EmbeddingProvider', 'SimpleEmbedding', 'Memory', 'Pattern', 'PatternStore'):
        from jack.foundation.memory import (
            AgenticMemory, MemoryNote, MemoryType, MemoryStore,
            MemoryLinker, MemoryConsolidator, ActiveForgetter,
            EmbeddingProvider, SimpleEmbedding, Memory, Pattern, PatternStore,
        )
        _lazy_imports.update({
            'AgenticMemory': AgenticMemory, 'MemoryNote': MemoryNote,
            'MemoryType': MemoryType, 'MemoryStore': MemoryStore,
            'MemoryLinker': MemoryLinker, 'MemoryConsolidator': MemoryConsolidator,
            'ActiveForgetter': ActiveForgetter, 'EmbeddingProvider': EmbeddingProvider,
            'SimpleEmbedding': SimpleEmbedding, 'Memory': Memory,
            'Pattern': Pattern, 'PatternStore': PatternStore,
        })
        return _lazy_imports.get(name)

    # Verify module
    if name in ('Verifier', 'Verdict', 'VerdictType', 'VerificationReport',
                'ConstitutionalVerifier', 'Constitution', 'ConstitutionalPrinciple', 'CritiqueResult',
                'SemanticSafetyVerifier', 'SafetyAssessment', 'RiskLevel',
                'ExecApprovalChecker', 'ExecApproval', 'AuthorizationLevel', 'PermissionPolicy',
                'GuardAgent', 'GoalAchievementVerifier', 'GoalVerificationResult',
                'SemanticValidator', 'HumanApprovalCallback', 'DefaultHumanApproval',
                'create_verifier', 'create_production_verifier', 'SafetyCheck'):
        from jack.foundation.verify import (
            Verifier, Verdict, VerdictType, VerificationReport,
            ConstitutionalVerifier, Constitution, ConstitutionalPrinciple, CritiqueResult,
            SemanticSafetyVerifier, SafetyAssessment, RiskLevel,
            ExecApprovalChecker, ExecApproval, AuthorizationLevel, PermissionPolicy,
            GuardAgent, GoalAchievementVerifier, GoalVerificationResult,
            SemanticValidator, HumanApprovalCallback, DefaultHumanApproval,
            create_verifier, create_production_verifier, SafetyCheck,
        )
        _lazy_imports.update({
            'Verifier': Verifier, 'Verdict': Verdict, 'VerdictType': VerdictType,
            'VerificationReport': VerificationReport, 'ConstitutionalVerifier': ConstitutionalVerifier,
            'Constitution': Constitution, 'ConstitutionalPrinciple': ConstitutionalPrinciple,
            'CritiqueResult': CritiqueResult, 'SemanticSafetyVerifier': SemanticSafetyVerifier,
            'SafetyAssessment': SafetyAssessment, 'RiskLevel': RiskLevel,
            'ExecApprovalChecker': ExecApprovalChecker, 'ExecApproval': ExecApproval,
            'AuthorizationLevel': AuthorizationLevel, 'PermissionPolicy': PermissionPolicy,
            'GuardAgent': GuardAgent, 'GoalAchievementVerifier': GoalAchievementVerifier,
            'GoalVerificationResult': GoalVerificationResult, 'SemanticValidator': SemanticValidator,
            'HumanApprovalCallback': HumanApprovalCallback, 'DefaultHumanApproval': DefaultHumanApproval,
            'create_verifier': create_verifier, 'create_production_verifier': create_production_verifier,
            'SafetyCheck': SafetyCheck,
        })
        return _lazy_imports.get(name)

    # Loop module
    if name in ('Loop', 'LoopConfig', 'LoopState', 'LoopPhase', 'LoopEvent', 'LoopEventData',
                'Reasoner', 'LoopSimpleReasoner', 'ContextWindowGuard', 'ContextWindowConfig',
                'ContextStats', 'StopHook', 'StopHookConfig', 'BacktrackTree', 'BacktrackNode',
                'LifecycleHook', 'HookRegistry', 'HookContext', 'HookCallback',
                'run_goal', 'create_production_loop', 'create_loop_with_logging'):
        from jack.foundation.loop import (
            Loop, LoopConfig, LoopState, LoopPhase, LoopEvent, LoopEventData,
            Reasoner, ContextWindowGuard, ContextWindowConfig, ContextStats,
            StopHook, StopHookConfig, BacktrackTree, BacktrackNode,
            LifecycleHook, HookRegistry, HookContext, HookCallback,
            run_goal, create_production_loop, create_loop_with_logging,
        )
        from jack.foundation.loop import SimpleReasoner as LoopSimpleReasoner
        _lazy_imports.update({
            'Loop': Loop, 'LoopConfig': LoopConfig, 'LoopState': LoopState,
            'LoopPhase': LoopPhase, 'LoopEvent': LoopEvent, 'LoopEventData': LoopEventData,
            'Reasoner': Reasoner, 'LoopSimpleReasoner': LoopSimpleReasoner,
            'ContextWindowGuard': ContextWindowGuard, 'ContextWindowConfig': ContextWindowConfig,
            'ContextStats': ContextStats, 'StopHook': StopHook, 'StopHookConfig': StopHookConfig,
            'BacktrackTree': BacktrackTree, 'BacktrackNode': BacktrackNode,
            'LifecycleHook': LifecycleHook, 'HookRegistry': HookRegistry,
            'HookContext': HookContext, 'HookCallback': HookCallback,
            'run_goal': run_goal, 'create_production_loop': create_production_loop,
            'create_loop_with_logging': create_loop_with_logging,
        })
        return _lazy_imports.get(name)

    # LLM module
    if name in ('LLMReasoner', 'LLMConfig', 'LLMResponse', 'AnthropicReasoner',
                'MultiProviderReasoner', 'SyncReasonerWrapper', 'LLMReasonerProtocol',
                'create_local_reasoner', 'create_openai_reasoner',
                'create_anthropic_reasoner', 'create_failover_reasoner'):
        from jack.foundation.llm import (
            LLMReasoner, LLMConfig, LLMResponse, AnthropicReasoner,
            MultiProviderReasoner, SyncReasonerWrapper,
            create_local_reasoner, create_openai_reasoner,
            create_anthropic_reasoner, create_failover_reasoner,
        )
        from jack.foundation.llm import Reasoner as LLMReasonerProtocol
        _lazy_imports.update({
            'LLMReasoner': LLMReasoner, 'LLMConfig': LLMConfig, 'LLMResponse': LLMResponse,
            'AnthropicReasoner': AnthropicReasoner, 'MultiProviderReasoner': MultiProviderReasoner,
            'SyncReasonerWrapper': SyncReasonerWrapper, 'LLMReasonerProtocol': LLMReasonerProtocol,
            'create_local_reasoner': create_local_reasoner, 'create_openai_reasoner': create_openai_reasoner,
            'create_anthropic_reasoner': create_anthropic_reasoner, 'create_failover_reasoner': create_failover_reasoner,
        })
        return _lazy_imports.get(name)

    # Brain reasoner (requires torch)
    if name in ('BrainReasoner', 'SyncLLMReasoner', 'HybridReasoner',
                'ActionHistory', 'ActionHistoryEntry', 'StateEncoder', 'ActionDecoder',
                'create_brain_reasoner', 'create_sync_llm_reasoner', 'create_hybrid_reasoner'):
        try:
            from jack.foundation.brain_reasoner import (
                BrainReasoner, SyncLLMReasoner, HybridReasoner,
                ActionHistory, ActionHistoryEntry, StateEncoder, ActionDecoder,
                create_brain_reasoner, create_hybrid_reasoner,
            )
            from jack.foundation.brain_reasoner import create_llm_reasoner as create_sync_llm_reasoner
            _lazy_imports.update({
                'BrainReasoner': BrainReasoner, 'SyncLLMReasoner': SyncLLMReasoner,
                'HybridReasoner': HybridReasoner, 'ActionHistory': ActionHistory,
                'ActionHistoryEntry': ActionHistoryEntry, 'StateEncoder': StateEncoder,
                'ActionDecoder': ActionDecoder, 'create_brain_reasoner': create_brain_reasoner,
                'create_sync_llm_reasoner': create_sync_llm_reasoner,
                'create_hybrid_reasoner': create_hybrid_reasoner,
            })
            return _lazy_imports.get(name)
        except ImportError as e:
            raise ImportError(f"BrainReasoner requires torch: {e}")

    # Other modules can be added similarly...
    # For now, raise AttributeError for unknown names
    raise AttributeError(f"module 'jack.foundation' has no attribute '{name}'")


__all__ = [
    # Types
    'Result', 'Ok', 'Err', 'Option', 'Some', 'NONE', 'Error', 'ErrorCode',
    # State
    'State', 'StateBuilder', 'Goal', 'GoalType', 'Entity', 'EntityType', 'Constraint', 'Context', 'Observation',
    'RelevanceScorer', 'SimpleRelevanceScorer',
    # Plan
    'Plan', 'PlanBuilder', 'Step', 'StepType', 'Primitive', 'PrimitiveType', 'Checkpoint',
    'Skill', 'GlobalPlanner', 'Replanner', 'PlanStatus',
    'create_plan_with_llm', 'create_simple_plan',
    # Action
    'ActionResult', 'Executor', 'OutcomeType', 'StateDelta',
    'ResourceTracker', 'ShellActionHandler', 'FileReadHandler', 'FileWriteHandler', 'HttpActionHandler',
    # Memory (lazy)
    'AgenticMemory', 'MemoryNote', 'MemoryType', 'MemoryStore',
    'MemoryLinker', 'MemoryConsolidator', 'ActiveForgetter',
    'EmbeddingProvider', 'SimpleEmbedding', 'Memory', 'Pattern', 'PatternStore',
    # Verify (lazy)
    'Verifier', 'Verdict', 'VerdictType', 'VerificationReport',
    'ConstitutionalVerifier', 'Constitution', 'ConstitutionalPrinciple', 'CritiqueResult',
    'SemanticSafetyVerifier', 'SafetyAssessment', 'RiskLevel',
    'ExecApprovalChecker', 'ExecApproval', 'AuthorizationLevel', 'PermissionPolicy',
    'GuardAgent', 'GoalAchievementVerifier', 'GoalVerificationResult',
    'SemanticValidator', 'HumanApprovalCallback', 'DefaultHumanApproval',
    'create_verifier', 'create_production_verifier', 'SafetyCheck',
    # Loop (lazy)
    'Loop', 'LoopConfig', 'LoopState', 'LoopPhase', 'LoopEvent', 'LoopEventData',
    'Reasoner', 'LoopSimpleReasoner',
    'ContextWindowGuard', 'ContextWindowConfig', 'ContextStats',
    'StopHook', 'StopHookConfig', 'BacktrackTree', 'BacktrackNode',
    'LifecycleHook', 'HookRegistry', 'HookContext', 'HookCallback',
    'run_goal', 'create_production_loop', 'create_loop_with_logging',
    # LLM (lazy)
    'LLMReasoner', 'LLMConfig', 'LLMResponse',
    'AnthropicReasoner', 'MultiProviderReasoner', 'SyncReasonerWrapper',
    'LLMReasonerProtocol',
    'create_local_reasoner', 'create_openai_reasoner',
    'create_anthropic_reasoner', 'create_failover_reasoner',
    # Brain Reasoner (lazy, requires torch)
    'BrainReasoner', 'SyncLLMReasoner', 'HybridReasoner',
    'ActionHistory', 'ActionHistoryEntry', 'StateEncoder', 'ActionDecoder',
    'create_brain_reasoner', 'create_sync_llm_reasoner', 'create_hybrid_reasoner',
]
