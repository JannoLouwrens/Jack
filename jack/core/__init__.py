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

from .executor import Executor, ActionResult
from .verifier import Verifier, VerifyResult
from .memory import Memory
from .jack_brain import JackBrain, JackConfig, DEFAULT_CONFIG, ACTION_TYPES
from .code_loop import (
    CodeRefinementLoop,
    CodeLanguage,
    CodeAttempt,
    RefinementResult,
    AnthropicLLM,
    refine_code,
)

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
