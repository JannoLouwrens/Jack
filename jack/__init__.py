"""
Jack - Simple Digital Agent

An AI that controls computers through 5 primitives:
- shell_run: Execute commands
- file_read: Read files
- file_write: Write files
- http_request: Make HTTP calls
- get_state: Observe system

Components:
- JackBrain: Transformer that predicts actions
- Executor: Executes actions safely
- Verifier: Checks results
- Training: Learn from experience
"""

__version__ = "0.2.0"
__author__ = "Janno Louwrens"

# Core
from .core import JackBrain, JackConfig, Executor, Verifier, Memory, ActionResult, VerifyResult

# Adapters
from .adapters import get_adapter, OSAdapter

# Training
from .training import JackTrainer, DigitalSandbox, TransitionBuffer, run_training

# Network
from .network import JackNode, NodeInfo, NodeDiscovery, Message, MessageType

# CLI
from .interfaces import JackCLI, run_cli

__all__ = [
    "__version__",
    "__author__",
    # Core
    "JackBrain",
    "JackConfig",
    "Executor",
    "Verifier",
    "Memory",
    "ActionResult",
    "VerifyResult",
    # Adapters
    "get_adapter",
    "OSAdapter",
    # Training
    "JackTrainer",
    "DigitalSandbox",
    "TransitionBuffer",
    "run_training",
    # Network
    "JackNode",
    "NodeInfo",
    "NodeDiscovery",
    "Message",
    "MessageType",
    # CLI
    "JackCLI",
    "run_cli",
]
