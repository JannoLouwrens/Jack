"""
Sandbox Execution Layer - Safe code execution for agents.

Provides isolated Python code execution with resource limits.
Prevents untrusted code from accessing the filesystem, network, or system.
"""

from __future__ import annotations
import ast
import sys
import io
import time
import signal
import logging
import threading
import traceback
from typing import Any, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

class SandboxError(Exception):
    """Base sandbox error."""
    pass


class SandboxSecurityError(SandboxError):
    """Security violation in sandbox."""
    pass


class SandboxTimeoutError(SandboxError):
    """Execution timeout in sandbox."""
    pass


class SandboxMemoryError(SandboxError):
    """Memory limit exceeded in sandbox."""
    pass


# =============================================================================
# AST Security Validator
# =============================================================================

# Dangerous builtins that should never be accessible
FORBIDDEN_BUILTINS: Set[str] = {
    "eval", "exec", "compile", "open", "input",
    "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "hasattr",
    "memoryview", "breakpoint",
}

# Dangerous AST node types
FORBIDDEN_NODES: Set[type] = {
    ast.Import,      # import x
    ast.ImportFrom,  # from x import y
}

# Dangerous attribute accesses
FORBIDDEN_ATTRIBUTES: Set[str] = {
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__code__", "__closure__", "__func__",
    "__self__", "__dict__", "__builtins__", "__import__",
    "__loader__", "__spec__", "__path__", "__file__",
    "__cached__", "__annotations__", "__kwdefaults__",
    "func_globals", "func_code", "func_closure",
    "im_class", "im_func", "im_self",
    "gi_frame", "gi_code", "cr_frame", "cr_code",
    "ag_frame", "ag_code",
}


class ASTSecurityVisitor(ast.NodeVisitor):
    """
    AST visitor that checks for security violations.

    Prevents:
    - Import statements
    - Access to dangerous attributes
    - Access to forbidden builtins
    """

    def __init__(self, allowed_modules: Set[str]):
        self.allowed_modules = allowed_modules
        self.violations: list = []

    def visit_Import(self, node: ast.Import) -> None:
        """Block import statements."""
        for alias in node.names:
            if alias.name.split(".")[0] not in self.allowed_modules:
                self.violations.append(
                    f"Import not allowed: {alias.name}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Block from...import statements."""
        if node.module:
            module_base = node.module.split(".")[0]
            if module_base not in self.allowed_modules:
                self.violations.append(
                    f"Import not allowed: from {node.module}"
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Block dangerous attribute access."""
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.violations.append(
                f"Forbidden attribute access: {node.attr}"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Block forbidden builtins."""
        if node.id in FORBIDDEN_BUILTINS:
            self.violations.append(
                f"Forbidden builtin: {node.id}"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls."""
        # Check for eval/exec style calls
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_BUILTINS:
                self.violations.append(
                    f"Forbidden function call: {node.func.id}"
                )
        self.generic_visit(node)

    def check(self, tree: ast.AST) -> list:
        """Check AST for security violations."""
        self.violations = []
        self.visit(tree)
        return self.violations


# =============================================================================
# Safe Builtins
# =============================================================================

def create_safe_builtins() -> Dict[str, Any]:
    """Create a restricted set of safe builtins."""
    import builtins

    # Whitelist of safe builtins
    safe_names = [
        # Types
        "bool", "int", "float", "str", "bytes", "list", "tuple",
        "dict", "set", "frozenset", "type", "object",
        # Functions
        "abs", "all", "any", "bin", "chr", "divmod", "enumerate",
        "filter", "format", "hash", "hex", "id", "isinstance",
        "issubclass", "iter", "len", "map", "max", "min", "next",
        "oct", "ord", "pow", "print", "range", "repr", "reversed",
        "round", "slice", "sorted", "sum", "zip",
        # Constants
        "True", "False", "None", "Ellipsis", "NotImplemented",
        # Exceptions (for try/except)
        "Exception", "BaseException", "ValueError", "TypeError",
        "KeyError", "IndexError", "AttributeError", "RuntimeError",
        "StopIteration", "ZeroDivisionError", "OverflowError",
        "AssertionError", "ArithmeticError", "LookupError",
    ]

    return {name: getattr(builtins, name) for name in safe_names if hasattr(builtins, name)}


# =============================================================================
# Sandbox Configuration
# =============================================================================

@dataclass
class SandboxConfig:
    """Sandbox execution configuration."""

    max_time: float = 30.0          # Maximum execution time (seconds)
    max_memory_mb: int = 512        # Maximum memory (MB) - advisory
    max_output_size: int = 1000000  # Maximum output size (bytes)
    max_iterations: int = 1000000   # Maximum loop iterations

    # Allowed modules for import
    allowed_modules: Set[str] = field(default_factory=lambda: {
        "math", "json", "datetime", "re", "collections",
        "itertools", "functools", "typing", "dataclasses",
        "decimal", "fractions", "random", "statistics",
        "string", "textwrap", "copy", "operator",
    })

    # Enable network access (dangerous!)
    allow_network: bool = False


# =============================================================================
# Sandbox Executor
# =============================================================================

@dataclass
class SandboxResult:
    """Result from sandbox execution."""

    success: bool
    output: str
    error: Optional[str] = None
    return_value: Any = None
    execution_time: float = 0.0


class Sandbox:
    """
    Safe Python code execution sandbox.

    Features:
    - AST-based security validation
    - Restricted builtins
    - Timeout enforcement
    - Output capture
    - Memory limit advisory

    Usage:
        sandbox = Sandbox()
        result = sandbox.execute("print(sum([1, 2, 3]))")
        print(result.output)  # "6"
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.safe_builtins = create_safe_builtins()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate code for security issues.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check for security violations
        visitor = ASTSecurityVisitor(self.config.allowed_modules)
        violations = visitor.check(tree)

        if violations:
            return False, "Security violations: " + "; ".join(violations)

        return True, None

    def _create_globals(self) -> Dict[str, Any]:
        """Create restricted globals for execution."""
        return {
            "__builtins__": self.safe_builtins,
            "__name__": "__sandbox__",
            "__doc__": None,
        }

    def _execute_code(
        self,
        code: str,
        globals_dict: Dict[str, Any],
        locals_dict: Dict[str, Any],
    ) -> Tuple[str, Any]:
        """Execute code and capture output."""
        # Capture stdout
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        captured_output = io.StringIO()
        sys.stdout = captured_output
        sys.stderr = captured_output

        return_value = None

        try:
            # Compile and execute
            compiled = compile(code, "<sandbox>", "exec")
            exec(compiled, globals_dict, locals_dict)

            # Check for return value (last expression)
            if "_result" in locals_dict:
                return_value = locals_dict["_result"]

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        output = captured_output.getvalue()

        # Truncate if too long
        if len(output) > self.config.max_output_size:
            output = output[:self.config.max_output_size] + "\n... (output truncated)"

        return output, return_value

    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SandboxResult:
        """
        Execute code in sandbox.

        Args:
            code: Python code to execute
            context: Optional variables to inject into scope

        Returns:
            SandboxResult with output and return value
        """
        start_time = time.time()

        # Validate code
        is_valid, error = self._validate_code(code)
        if not is_valid:
            return SandboxResult(
                success=False,
                output="",
                error=error,
            )

        # Create execution environment
        globals_dict = self._create_globals()
        locals_dict = {}

        # Inject context (with validation)
        if context:
            for key, value in context.items():
                if not key.startswith("_"):
                    locals_dict[key] = value

        # Add allowed modules to globals
        for module_name in self.config.allowed_modules:
            try:
                module = __import__(module_name)
                globals_dict[module_name] = module
            except ImportError:
                pass

        try:
            # Execute with timeout
            future = self._executor.submit(
                self._execute_code,
                code,
                globals_dict,
                locals_dict,
            )

            output, return_value = future.result(timeout=self.config.max_time)

            execution_time = time.time() - start_time

            return SandboxResult(
                success=True,
                output=output,
                return_value=return_value,
                execution_time=execution_time,
            )

        except FuturesTimeout:
            return SandboxResult(
                success=False,
                output="",
                error=f"Execution timeout ({self.config.max_time}s)",
                execution_time=self.config.max_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {e}"

            # Include traceback for debugging
            tb = traceback.format_exc()
            if "sandbox" in tb.lower():
                # Clean sandbox-specific parts from traceback
                lines = tb.split("\n")
                tb = "\n".join(
                    line for line in lines
                    if "<sandbox>" in line or "Error" in line
                )

            return SandboxResult(
                success=False,
                output="",
                error=f"{error_msg}\n{tb}".strip(),
                execution_time=execution_time,
            )

    async def execute_async(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SandboxResult:
        """
        Execute code asynchronously.

        Same as execute() but runs in thread pool.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.execute(code, context),
        )

    def close(self) -> None:
        """Clean up executor."""
        self._executor.shutdown(wait=False)


# =============================================================================
# FastAPI Integration
# =============================================================================

# Global sandbox instance
_sandbox: Optional[Sandbox] = None


def get_sandbox() -> Sandbox:
    """Get or create sandbox instance."""
    global _sandbox
    if _sandbox is None:
        _sandbox = Sandbox()
    return _sandbox


# =============================================================================
# Convenience Functions
# =============================================================================

def safe_eval(expression: str, context: Optional[Dict[str, Any]] = None) -> Any:
    """
    Safely evaluate a single expression.

    Args:
        expression: Python expression to evaluate
        context: Optional variables

    Returns:
        Evaluated result

    Raises:
        SandboxError: If evaluation fails
    """
    sandbox = get_sandbox()

    # Wrap expression to capture result
    code = f"_result = {expression}"

    result = sandbox.execute(code, context)

    if not result.success:
        raise SandboxError(result.error)

    return result.return_value


def safe_exec(code: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Safely execute code and return output.

    Args:
        code: Python code to execute
        context: Optional variables

    Returns:
        Captured output

    Raises:
        SandboxError: If execution fails
    """
    sandbox = get_sandbox()
    result = sandbox.execute(code, context)

    if not result.success:
        raise SandboxError(result.error)

    return result.output
