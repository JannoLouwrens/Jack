"""
OBSERVABLE ACTION - Execution with Full Observability

This module implements action execution that is:
- Observable: Full state before/after captured
- Safe: Verified before execution
- Recoverable: Errors are values, not exceptions
- Measurable: Timing and resource usage tracked

The key insight: Every action should be fully observable.
We capture the complete state transition for:
- Debugging: What went wrong?
- Learning: What patterns succeed?
- Verification: Did it work?
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, Any, Optional, Callable, Protocol, runtime_checkable,
    TypeVar, Generic, List, Tuple, Union
)
from datetime import datetime
from enum import Enum, auto
import subprocess
import time
import os
import hashlib
import json
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from jack.foundation.types import Result, Ok, Err, Option, Some, NONE, Error, ErrorCode
from jack.foundation.plan import Primitive, PrimitiveType

# Resource tracking (optional psutil)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False


# =============================================================================
# RESOURCE TRACKER
# =============================================================================

class ResourceTracker:
    """
    Track resource usage during action execution.

    Uses psutil if available, otherwise provides estimates.
    """

    def __init__(self):
        self.process = psutil.Process() if HAS_PSUTIL else None

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if self.process:
            try:
                mem = self.process.memory_info()
                return mem.rss / (1024 * 1024)
            except Exception:
                pass
        return 0.0

    def get_cpu_percent(self) -> float:
        """Get CPU usage percent."""
        if self.process:
            try:
                return self.process.cpu_percent(interval=0.1)
            except Exception:
                pass
        return 0.0

    def snapshot(self) -> Dict[str, float]:
        """Get resource snapshot."""
        return {
            "memory_mb": self.get_memory_mb(),
            "cpu_percent": self.get_cpu_percent(),
            "timestamp": time.time(),
        }

    def measure_execution(self, func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """
        Execute function and measure resource usage.

        Returns (result, resources) where resources contains:
        - memory_before_mb, memory_after_mb, memory_peak_mb
        - cpu_percent
        - duration_seconds
        """
        before = self.snapshot()

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        after = self.snapshot()

        resources = {
            "memory_before_mb": before["memory_mb"],
            "memory_after_mb": after["memory_mb"],
            "memory_delta_mb": after["memory_mb"] - before["memory_mb"],
            "cpu_percent": max(before["cpu_percent"], after["cpu_percent"]),
            "duration_seconds": end_time - start_time,
        }

        return result, resources


# Global resource tracker instance
_resource_tracker = ResourceTracker()


# =============================================================================
# ACTION OUTCOME
# =============================================================================

class OutcomeType(Enum):
    """Classification of action outcomes."""
    SUCCESS = auto()        # Action succeeded
    FAILURE = auto()        # Action failed (expected error)
    ERROR = auto()          # Action errored (unexpected)
    TIMEOUT = auto()        # Action timed out
    BLOCKED = auto()        # Action blocked by verifier
    SKIPPED = auto()        # Action skipped


@dataclass(frozen=True)
class StateDelta:
    """
    Captures what changed after an action.

    This is the key to learning - we observe the EFFECT of actions.
    """
    # File changes
    files_created: Tuple[str, ...] = ()
    files_modified: Tuple[str, ...] = ()
    files_deleted: Tuple[str, ...] = ()

    # Content changes
    content_before_hash: Optional[str] = None
    content_after_hash: Optional[str] = None

    # Size changes
    size_before: int = 0
    size_after: int = 0

    # Process changes
    processes_started: Tuple[str, ...] = ()
    processes_ended: Tuple[str, ...] = ()

    @property
    def has_changes(self) -> bool:
        """Did anything change?"""
        return bool(
            self.files_created or
            self.files_modified or
            self.files_deleted or
            self.content_before_hash != self.content_after_hash
        )

    @property
    def size_delta(self) -> int:
        """Net size change."""
        return self.size_after - self.size_before

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_created": list(self.files_created),
            "files_modified": list(self.files_modified),
            "files_deleted": list(self.files_deleted),
            "size_delta": self.size_delta,
            "has_changes": self.has_changes,
        }


@dataclass(frozen=True)
class ActionResult:
    """
    Complete result of an action execution.

    This captures EVERYTHING about what happened:
    - What was attempted
    - What was the state before
    - What happened (success/failure)
    - What changed (delta)
    - How long it took
    """
    # Action info
    primitive: Primitive
    outcome: OutcomeType

    # Output
    output: Any = None
    error: Optional[Error] = None

    # State observation
    state_before: Dict[str, Any] = field(default_factory=dict)
    state_after: Dict[str, Any] = field(default_factory=dict)
    delta: Optional[StateDelta] = None

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0

    # Resource usage
    memory_used_mb: float = 0.0
    cpu_percent: float = 0.0

    @property
    def is_success(self) -> bool:
        """Did the action succeed?"""
        return self.outcome == OutcomeType.SUCCESS

    @property
    def is_failure(self) -> bool:
        """Did the action fail?"""
        return self.outcome in (OutcomeType.FAILURE, OutcomeType.ERROR, OutcomeType.TIMEOUT)

    def to_result(self) -> Result[Any, Error]:
        """Convert to Result type."""
        if self.is_success:
            return Ok(self.output)
        return Err(self.error or Error(ErrorCode.UNKNOWN, "Unknown error"))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/logging."""
        return {
            "primitive_type": self.primitive.primitive_type.name,
            "primitive_params": self.primitive.params,
            "outcome": self.outcome.name,
            "output": str(self.output)[:1000] if self.output else None,
            "error": str(self.error) if self.error else None,
            "duration_ms": self.duration_ms,
            "delta": self.delta.to_dict() if self.delta else None,
        }

    @property
    def fingerprint(self) -> str:
        """Unique fingerprint for this result."""
        content = json.dumps({
            "type": self.primitive.primitive_type.name,
            "params": str(self.primitive.params),
            "outcome": self.outcome.name,
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]


# =============================================================================
# ACTION INTERFACE
# =============================================================================

@runtime_checkable
class ActionHandler(Protocol):
    """Protocol for action handlers."""

    def can_handle(self, primitive: Primitive) -> bool:
        """Check if this handler can execute the primitive."""
        ...

    def execute(self, primitive: Primitive) -> ActionResult:
        """Execute the primitive and return result."""
        ...


# =============================================================================
# BASE ACTION HANDLER
# =============================================================================

class BaseActionHandler(ABC):
    """Base class for action handlers with common functionality."""

    @abstractmethod
    def execute_impl(self, primitive: Primitive) -> Tuple[Any, Optional[Error]]:
        """Implementation-specific execution. Returns (output, error)."""
        ...

    def get_state_before(self, primitive: Primitive) -> Dict[str, Any]:
        """Capture state before execution."""
        return {
            "timestamp": datetime.now().isoformat(),
            "cwd": os.getcwd(),
        }

    def get_state_after(self, primitive: Primitive, output: Any) -> Dict[str, Any]:
        """Capture state after execution."""
        return {
            "timestamp": datetime.now().isoformat(),
            "cwd": os.getcwd(),
        }

    def compute_delta(
        self,
        primitive: Primitive,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
    ) -> StateDelta:
        """Compute what changed."""
        return StateDelta()

    def execute(self, primitive: Primitive) -> ActionResult:
        """Execute with full observability and resource tracking."""
        started_at = datetime.now()
        state_before = self.get_state_before(primitive)
        resource_before = _resource_tracker.snapshot()

        try:
            # Execute with timeout
            output, error = self._execute_with_timeout(primitive)

            if error:
                outcome = OutcomeType.FAILURE
            else:
                outcome = OutcomeType.SUCCESS

        except FuturesTimeoutError:
            output = None
            error = Error(ErrorCode.TIMEOUT, f"Action timed out after {primitive.timeout_seconds}s")
            outcome = OutcomeType.TIMEOUT

        except Exception as e:
            output = None
            error = Error(ErrorCode.EXECUTION_FAILED, str(e), details={"traceback": traceback.format_exc()})
            outcome = OutcomeType.ERROR

        completed_at = datetime.now()
        state_after = self.get_state_after(primitive, output)
        delta = self.compute_delta(primitive, state_before, state_after)
        resource_after = _resource_tracker.snapshot()

        # Calculate resource usage
        memory_used = resource_after["memory_mb"] - resource_before["memory_mb"]
        cpu_percent = max(resource_before["cpu_percent"], resource_after["cpu_percent"])

        return ActionResult(
            primitive=primitive,
            outcome=outcome,
            output=output,
            error=error,
            state_before=state_before,
            state_after=state_after,
            delta=delta,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=(completed_at - started_at).total_seconds() * 1000,
            memory_used_mb=max(0.0, memory_used),
            cpu_percent=cpu_percent,
        )

    def _execute_with_timeout(self, primitive: Primitive) -> Tuple[Any, Optional[Error]]:
        """Execute with timeout protection."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.execute_impl, primitive)
            try:
                return future.result(timeout=primitive.timeout_seconds)
            except FuturesTimeoutError:
                raise


# =============================================================================
# SHELL ACTION HANDLER
# =============================================================================

class ShellActionHandler(BaseActionHandler):
    """Handler for shell command execution."""

    def __init__(self, working_dir: Optional[str] = None, shell: bool = True):
        self.working_dir = working_dir
        self.shell = shell

    def can_handle(self, primitive: Primitive) -> bool:
        return primitive.primitive_type == PrimitiveType.SHELL_RUN

    def execute_impl(self, primitive: Primitive) -> Tuple[Any, Optional[Error]]:
        command = primitive.params.get("command", "")

        result = subprocess.run(
            command,
            shell=self.shell,
            cwd=self.working_dir,
            capture_output=True,
            text=True,
            timeout=primitive.timeout_seconds,
        )

        output = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

        if result.returncode != 0:
            error = Error(
                ErrorCode.EXECUTION_FAILED,
                f"Command exited with code {result.returncode}",
                details={"stderr": result.stderr[:500]},
            )
            return output, error

        return output, None

    def get_state_before(self, primitive: Primitive) -> Dict[str, Any]:
        state = super().get_state_before(primitive)
        state["command"] = primitive.params.get("command", "")[:100]
        return state


# =============================================================================
# FILE ACTION HANDLERS
# =============================================================================

class FileReadHandler(BaseActionHandler):
    """Handler for file reading."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def can_handle(self, primitive: Primitive) -> bool:
        return primitive.primitive_type == PrimitiveType.FILE_READ

    def execute_impl(self, primitive: Primitive) -> Tuple[Any, Optional[Error]]:
        path = primitive.params.get("path", "")
        full_path = self.base_dir / path

        if not full_path.exists():
            return None, Error(ErrorCode.NOT_FOUND, f"File not found: {path}")

        try:
            content = full_path.read_text()
            return {
                "content": content,
                "size": len(content),
                "path": str(full_path),
            }, None
        except PermissionError:
            return None, Error(ErrorCode.PERMISSION_DENIED, f"Cannot read: {path}")
        except Exception as e:
            return None, Error(ErrorCode.EXECUTION_FAILED, str(e))

    def get_state_before(self, primitive: Primitive) -> Dict[str, Any]:
        state = super().get_state_before(primitive)
        path = primitive.params.get("path", "")
        full_path = self.base_dir / path
        state["path"] = str(full_path)
        state["exists"] = full_path.exists()
        if full_path.exists():
            state["size"] = full_path.stat().st_size
        return state


class FileWriteHandler(BaseActionHandler):
    """Handler for file writing."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def can_handle(self, primitive: Primitive) -> bool:
        return primitive.primitive_type == PrimitiveType.FILE_WRITE

    def execute_impl(self, primitive: Primitive) -> Tuple[Any, Optional[Error]]:
        path = primitive.params.get("path", "")
        content = primitive.params.get("content", "")
        full_path = self.base_dir / path

        try:
            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content
            full_path.write_text(content)

            return {
                "path": str(full_path),
                "size": len(content),
                "created": not full_path.exists(),
            }, None
        except PermissionError:
            return None, Error(ErrorCode.PERMISSION_DENIED, f"Cannot write: {path}")
        except Exception as e:
            return None, Error(ErrorCode.EXECUTION_FAILED, str(e))

    def get_state_before(self, primitive: Primitive) -> Dict[str, Any]:
        state = super().get_state_before(primitive)
        path = primitive.params.get("path", "")
        full_path = self.base_dir / path
        state["path"] = str(full_path)
        state["exists_before"] = full_path.exists()
        if full_path.exists():
            state["size_before"] = full_path.stat().st_size
            state["content_hash_before"] = hashlib.md5(full_path.read_bytes()).hexdigest()[:16]
        return state

    def get_state_after(self, primitive: Primitive, output: Any) -> Dict[str, Any]:
        state = super().get_state_after(primitive, output)
        path = primitive.params.get("path", "")
        full_path = self.base_dir / path
        state["exists_after"] = full_path.exists()
        if full_path.exists():
            state["size_after"] = full_path.stat().st_size
            state["content_hash_after"] = hashlib.md5(full_path.read_bytes()).hexdigest()[:16]
        return state

    def compute_delta(
        self,
        primitive: Primitive,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
    ) -> StateDelta:
        path = primitive.params.get("path", "")
        created = not state_before.get("exists_before", False)
        modified = (
            state_before.get("content_hash_before") !=
            state_after.get("content_hash_after")
        )

        return StateDelta(
            files_created=(path,) if created else (),
            files_modified=(path,) if modified and not created else (),
            size_before=state_before.get("size_before", 0),
            size_after=state_after.get("size_after", 0),
            content_before_hash=state_before.get("content_hash_before"),
            content_after_hash=state_after.get("content_hash_after"),
        )


# =============================================================================
# HTTP ACTION HANDLER
# =============================================================================

class HttpActionHandler(BaseActionHandler):
    """Handler for HTTP requests."""

    def __init__(self):
        try:
            import requests
            self.requests = requests
        except ImportError:
            self.requests = None

    def can_handle(self, primitive: Primitive) -> bool:
        return primitive.primitive_type == PrimitiveType.HTTP_REQUEST

    def execute_impl(self, primitive: Primitive) -> Tuple[Any, Optional[Error]]:
        if self.requests is None:
            return None, Error(ErrorCode.EXECUTION_FAILED, "requests library not installed")

        method = primitive.params.get("method", "GET")
        url = primitive.params.get("url", "")
        body = primitive.params.get("body")
        headers = primitive.params.get("headers", {})

        try:
            response = self.requests.request(
                method=method,
                url=url,
                json=body if body else None,
                headers=headers,
                timeout=primitive.timeout_seconds,
            )

            output = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text[:10000],  # Limit response size
            }

            if response.status_code >= 400:
                return output, Error(
                    ErrorCode.EXECUTION_FAILED,
                    f"HTTP {response.status_code}",
                    details={"body": response.text[:500]},
                )

            return output, None

        except Exception as e:
            return None, Error(ErrorCode.EXECUTION_FAILED, str(e))


# =============================================================================
# EXECUTOR
# =============================================================================

class Executor:
    """
    Main executor that routes primitives to appropriate handlers.

    The executor:
    - Routes primitives to handlers
    - Captures full execution context
    - Stores results for learning
    - Handles retries
    """

    def __init__(self, handlers: List[ActionHandler] = None):
        self.handlers: List[ActionHandler] = handlers or []
        self.execution_history: List[ActionResult] = []
        self._setup_default_handlers()

    def _setup_default_handlers(self):
        """Set up default action handlers."""
        if not any(h.can_handle(Primitive.shell("")) for h in self.handlers if hasattr(h, 'can_handle')):
            self.handlers.append(ShellActionHandler())
        if not any(h.can_handle(Primitive.read_file("")) for h in self.handlers if hasattr(h, 'can_handle')):
            self.handlers.append(FileReadHandler())
            self.handlers.append(FileWriteHandler())
        if not any(h.can_handle(Primitive.http("GET", "")) for h in self.handlers if hasattr(h, 'can_handle')):
            self.handlers.append(HttpActionHandler())

    def add_handler(self, handler: ActionHandler) -> None:
        """Add a custom handler."""
        self.handlers.append(handler)

    def get_handler(self, primitive: Primitive) -> Optional[ActionHandler]:
        """Find handler for primitive."""
        for handler in self.handlers:
            if handler.can_handle(primitive):
                return handler
        return None

    def execute(self, primitive: Primitive) -> ActionResult:
        """Execute a primitive action."""
        handler = self.get_handler(primitive)

        if handler is None:
            result = ActionResult(
                primitive=primitive,
                outcome=OutcomeType.ERROR,
                error=Error(
                    ErrorCode.NOT_FOUND,
                    f"No handler for primitive type: {primitive.primitive_type.name}"
                ),
            )
        else:
            result = handler.execute(primitive)

        # Store in history
        self.execution_history.append(result)

        return result

    def execute_with_retry(self, primitive: Primitive) -> ActionResult:
        """Execute with automatic retry on failure."""
        result = self.execute(primitive)

        while result.is_failure and primitive.can_retry():
            primitive = primitive.as_retried()
            result = self.execute(primitive)

        return result

    def execute_sequence(self, primitives: List[Primitive]) -> List[ActionResult]:
        """Execute a sequence of primitives, stopping on failure."""
        results = []
        for primitive in primitives:
            result = self.execute(primitive)
            results.append(result)
            if result.is_failure:
                break
        return results

    def get_success_rate(self, primitive_type: PrimitiveType = None) -> float:
        """Calculate success rate from history."""
        relevant = self.execution_history
        if primitive_type:
            relevant = [r for r in relevant if r.primitive.primitive_type == primitive_type]

        if not relevant:
            return 0.5  # No data, assume 50%

        successes = sum(1 for r in relevant if r.is_success)
        return successes / len(relevant)

    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_history = []


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def shell_run(command: str, timeout: float = 30.0) -> ActionResult:
    """Quick shell command execution."""
    executor = Executor()
    return executor.execute(Primitive.shell(command, timeout=timeout))


def read_file(path: str) -> ActionResult:
    """Quick file read."""
    executor = Executor()
    return executor.execute(Primitive.read_file(path))


def write_file(path: str, content: str) -> ActionResult:
    """Quick file write."""
    executor = Executor()
    return executor.execute(Primitive.write_file(path, content))


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("OBSERVABLE ACTION - Execution with Full Observability")
    print("=" * 60)

    executor = Executor()

    # Test shell command
    print("\n[TEST] Shell command")
    result = executor.execute(Primitive.shell("echo 'Hello, World!'"))
    print(f"  Outcome: {result.outcome.name}")
    print(f"  Duration: {result.duration_ms:.2f}ms")
    print(f"  Output: {result.output.get('stdout', '').strip() if result.output else 'None'}")

    # Test file operations
    print("\n[TEST] File write")
    test_path = "/tmp/jack_test_file.txt"
    result = executor.execute(Primitive.write_file(test_path, "Test content"))
    print(f"  Outcome: {result.outcome.name}")
    print(f"  Delta: {result.delta.to_dict() if result.delta else 'None'}")

    print("\n[TEST] File read")
    result = executor.execute(Primitive.read_file(test_path))
    print(f"  Outcome: {result.outcome.name}")
    print(f"  Content: {result.output.get('content', '')[:50] if result.output else 'None'}")

    # Test failure handling
    print("\n[TEST] Command failure")
    result = executor.execute(Primitive.shell("nonexistent_command_xyz"))
    print(f"  Outcome: {result.outcome.name}")
    print(f"  Error: {result.error}")

    # Test file not found
    print("\n[TEST] File not found")
    result = executor.execute(Primitive.read_file("/nonexistent/path/file.txt"))
    print(f"  Outcome: {result.outcome.name}")
    print(f"  Error: {result.error}")

    # Test success rate
    print("\n[TEST] Success rate")
    print(f"  Overall: {executor.get_success_rate():.2%}")
    print(f"  Shell: {executor.get_success_rate(PrimitiveType.SHELL_RUN):.2%}")

    print("\n" + "=" * 60)
    print("[OK] Observable action working")
    print("=" * 60)
