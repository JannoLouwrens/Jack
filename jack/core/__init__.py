"""
Jack Core - Simple Digital Agent

Jack's 5 primitives:
- shell_run: Execute shell commands
- file_read: Read file contents
- file_write: Write to files
- http_request: Make HTTP requests
- get_state: Observe system state

Components:
- JackBrain: Transformer that predicts actions from state + history
- Executor: Executes actions safely in sandbox
- Verifier: Checks if actions succeeded
- Memory: Stores experiences for learning
"""

# Light imports that don't require torch
from .executor import Executor, ActionResult
from .verifier import Verifier, VerifyResult
from .memory import Memory

# Lazy imports for torch-dependent modules
_lazy_imports = {}


def __getattr__(name):
    """Lazy loading for torch-dependent modules."""
    if name in _lazy_imports:
        return _lazy_imports[name]

    # JackBrain requires torch
    if name in ("JackBrain", "JackConfig", "DEFAULT_CONFIG", "ACTION_TYPES"):
        try:
            from .jack_brain import JackBrain, JackConfig, DEFAULT_CONFIG, ACTION_TYPES
            _lazy_imports.update({
                "JackBrain": JackBrain,
                "JackConfig": JackConfig,
                "DEFAULT_CONFIG": DEFAULT_CONFIG,
                "ACTION_TYPES": ACTION_TYPES,
            })
            return _lazy_imports[name]
        except ImportError as e:
            raise ImportError(f"JackBrain requires torch: {e}")

    # Code loop may require anthropic
    if name in ("CodeRefinementLoop", "CodeLanguage", "CodeAttempt", "RefinementResult", "AnthropicLLM", "refine_code"):
        try:
            from .code_loop import (
                CodeRefinementLoop, CodeLanguage, CodeAttempt,
                RefinementResult, AnthropicLLM, refine_code,
            )
            _lazy_imports.update({
                "CodeRefinementLoop": CodeRefinementLoop,
                "CodeLanguage": CodeLanguage,
                "CodeAttempt": CodeAttempt,
                "RefinementResult": RefinementResult,
                "AnthropicLLM": AnthropicLLM,
                "refine_code": refine_code,
            })
            return _lazy_imports[name]
        except ImportError as e:
            raise ImportError(f"CodeRefinementLoop requires anthropic: {e}")

    raise AttributeError(f"module 'jack.core' has no attribute '{name}'")


__all__ = [
    # Brain
    "JackBrain",
    "JackConfig",
    "DEFAULT_CONFIG",
    "ACTION_TYPES",
    # Execution
    "Executor",
    "ActionResult",
    "Verifier",
    "VerifyResult",
    "Memory",
    # Code refinement
    "CodeRefinementLoop",
    "CodeLanguage",
    "CodeAttempt",
    "RefinementResult",
    "AnthropicLLM",
    "refine_code",
]
