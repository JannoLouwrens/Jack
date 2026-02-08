"""
Sandbox Tests.

Tests for the safe code execution sandbox.
"""

import pytest

from jack.server.sandbox import (
    Sandbox, SandboxConfig, SandboxResult,
    SandboxError, SandboxSecurityError,
    safe_eval, safe_exec,
)


class TestSandbox:
    """Tests for the Sandbox class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sandbox = Sandbox()

    def teardown_method(self):
        """Clean up."""
        if self.sandbox:
            self.sandbox.close()

    def test_simple_expression(self):
        """Test simple arithmetic."""
        result = self.sandbox.execute("_result = 2 + 2")
        assert result.success is True
        assert result.return_value == 4

    def test_print_output(self):
        """Test print output capture."""
        result = self.sandbox.execute("print('Hello, World!')")
        assert result.success is True
        assert "Hello, World!" in result.output

    def test_multiple_statements(self):
        """Test multiple statements."""
        code = """
x = 10
y = 20
print(x + y)
_result = x * y
"""
        result = self.sandbox.execute(code)
        assert result.success is True
        assert "30" in result.output
        assert result.return_value == 200

    def test_list_operations(self):
        """Test list operations."""
        code = """
numbers = [1, 2, 3, 4, 5]
_result = sum(numbers)
"""
        result = self.sandbox.execute(code)
        assert result.success is True
        assert result.return_value == 15

    def test_dict_operations(self):
        """Test dictionary operations."""
        code = """
data = {'a': 1, 'b': 2, 'c': 3}
_result = sum(data.values())
"""
        result = self.sandbox.execute(code)
        assert result.success is True
        assert result.return_value == 6

    def test_context_injection(self):
        """Test injecting context variables."""
        result = self.sandbox.execute(
            "_result = x + y",
            context={"x": 10, "y": 20}
        )
        assert result.success is True
        assert result.return_value == 30

    def test_allowed_imports(self):
        """Test that allowed modules work."""
        code = """
import math
_result = math.sqrt(16)
"""
        result = self.sandbox.execute(code)
        assert result.success is True
        assert result.return_value == 4.0

    def test_json_module(self):
        """Test JSON module."""
        code = """
import json
data = json.dumps({'key': 'value'})
_result = json.loads(data)
"""
        result = self.sandbox.execute(code)
        assert result.success is True
        assert result.return_value == {"key": "value"}


class TestSandboxSecurity:
    """Security tests for the sandbox."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sandbox = Sandbox()

    def teardown_method(self):
        """Clean up."""
        if self.sandbox:
            self.sandbox.close()

    def test_block_import_os(self):
        """Test that os module is blocked."""
        result = self.sandbox.execute("import os")
        assert result.success is False
        assert "not allowed" in result.error.lower()

    def test_block_import_sys(self):
        """Test that sys module is blocked."""
        result = self.sandbox.execute("import sys")
        assert result.success is False

    def test_block_import_subprocess(self):
        """Test that subprocess is blocked."""
        result = self.sandbox.execute("import subprocess")
        assert result.success is False

    def test_block_open(self):
        """Test that open() is blocked."""
        result = self.sandbox.execute("open('/etc/passwd')")
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_block_eval(self):
        """Test that eval() is blocked."""
        result = self.sandbox.execute("eval('1+1')")
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_block_exec(self):
        """Test that exec() is blocked."""
        result = self.sandbox.execute("exec('x=1')")
        assert result.success is False

    def test_block_compile(self):
        """Test that compile() is blocked."""
        result = self.sandbox.execute("compile('x=1', '', 'exec')")
        assert result.success is False

    def test_block_dunder_class(self):
        """Test that __class__ access is blocked."""
        result = self.sandbox.execute("().__class__")
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_block_dunder_bases(self):
        """Test that __bases__ access is blocked."""
        result = self.sandbox.execute("object.__bases__")
        assert result.success is False

    def test_block_dunder_subclasses(self):
        """Test that __subclasses__ is blocked."""
        result = self.sandbox.execute("object.__subclasses__()")
        assert result.success is False

    def test_block_dunder_globals(self):
        """Test that __globals__ is blocked."""
        result = self.sandbox.execute("(lambda: 0).__globals__")
        assert result.success is False

    def test_block_getattr(self):
        """Test that getattr() is blocked."""
        result = self.sandbox.execute("getattr(object, '__bases__')")
        assert result.success is False


class TestSandboxTimeout:
    """Timeout tests for the sandbox."""

    def test_timeout_infinite_loop(self):
        """Test that infinite loops are stopped."""
        sandbox = Sandbox(SandboxConfig(max_time=1.0))

        result = sandbox.execute("while True: pass")
        assert result.success is False
        assert "timeout" in result.error.lower()

        sandbox.close()

    def test_execution_time_tracked(self):
        """Test that execution time is tracked."""
        sandbox = Sandbox()
        result = sandbox.execute("x = sum(range(1000))")

        assert result.success is True
        assert result.execution_time > 0
        assert result.execution_time < 1.0  # Should be fast

        sandbox.close()


class TestSandboxOutputLimit:
    """Output limit tests for the sandbox."""

    def test_output_truncation(self):
        """Test that excessive output is truncated."""
        sandbox = Sandbox(SandboxConfig(max_output_size=100))

        result = sandbox.execute("print('x' * 1000)")
        assert result.success is True
        assert len(result.output) <= 150  # Some overhead allowed
        assert "truncated" in result.output.lower()

        sandbox.close()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_safe_eval_simple(self):
        """Test safe_eval with simple expression."""
        result = safe_eval("2 + 2")
        assert result == 4

    def test_safe_eval_with_context(self):
        """Test safe_eval with context."""
        result = safe_eval("x * y", context={"x": 3, "y": 4})
        assert result == 12

    def test_safe_eval_blocked(self):
        """Test that dangerous code is blocked."""
        with pytest.raises(SandboxError):
            safe_eval("open('/etc/passwd')")

    def test_safe_exec_simple(self):
        """Test safe_exec with simple code."""
        output = safe_exec("print('Hello')")
        assert "Hello" in output

    def test_safe_exec_blocked(self):
        """Test that dangerous code is blocked."""
        with pytest.raises(SandboxError):
            safe_exec("import os")
