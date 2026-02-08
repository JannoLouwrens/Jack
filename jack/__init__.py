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

# Lazy imports - only load when accessed
# This allows the server to run without torch installed

_lazy_imports = {}


def __getattr__(name):
    """Lazy loading of heavy modules to avoid torch requirement for server."""
    if name in _lazy_imports:
        return _lazy_imports[name]

    # Core imports (some require torch)
    if name in ("JackBrain", "JackConfig", "Executor", "Verifier", "Memory", "ActionResult", "VerifyResult"):
        try:
            from .core import JackBrain, JackConfig, Executor, Verifier, Memory, ActionResult, VerifyResult
            _lazy_imports.update({
                "JackBrain": JackBrain,
                "JackConfig": JackConfig,
                "Executor": Executor,
                "Verifier": Verifier,
                "Memory": Memory,
                "ActionResult": ActionResult,
                "VerifyResult": VerifyResult,
            })
            return _lazy_imports[name]
        except ImportError as e:
            raise ImportError(f"Core module requires torch: {e}")

    # Adapters
    if name in ("get_adapter", "OSAdapter"):
        from .adapters import get_adapter, OSAdapter
        _lazy_imports["get_adapter"] = get_adapter
        _lazy_imports["OSAdapter"] = OSAdapter
        return _lazy_imports[name]

    # Training (requires torch)
    if name in ("JackTrainer", "DigitalSandbox", "TransitionBuffer", "run_training"):
        try:
            from .training import JackTrainer, DigitalSandbox, TransitionBuffer, run_training
            _lazy_imports.update({
                "JackTrainer": JackTrainer,
                "DigitalSandbox": DigitalSandbox,
                "TransitionBuffer": TransitionBuffer,
                "run_training": run_training,
            })
            return _lazy_imports[name]
        except ImportError as e:
            raise ImportError(f"Training module requires torch: {e}")

    # Network
    if name in ("JackNode", "NodeInfo", "NodeDiscovery", "Message", "MessageType"):
        from .network import JackNode, NodeInfo, NodeDiscovery, Message, MessageType
        _lazy_imports.update({
            "JackNode": JackNode,
            "NodeInfo": NodeInfo,
            "NodeDiscovery": NodeDiscovery,
            "Message": Message,
            "MessageType": MessageType,
        })
        return _lazy_imports[name]

    # CLI
    if name in ("JackCLI", "run_cli"):
        from .interfaces import JackCLI, run_cli
        _lazy_imports["JackCLI"] = JackCLI
        _lazy_imports["run_cli"] = run_cli
        return _lazy_imports[name]

    raise AttributeError(f"module 'jack' has no attribute '{name}'")


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
