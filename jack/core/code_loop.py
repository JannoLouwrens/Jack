"""
Code Generation + Testing + Refinement Loop

The core loop that makes Jack useful:
1. Ask LLM to generate code for a task
2. Verify it's safe (static analysis)
3. Run it in sandbox
4. If failed: build context from error, ask LLM to fix
5. Repeat until success or max attempts

This is how Claude Code, Devin, etc. work.
No magic prediction - just run and learn from errors.
"""

import os
import re
import ast
import sys
import time
import traceback
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from enum import Enum


class CodeLanguage(Enum):
    PYTHON = "python"
    BASH = "bash"
    POWERSHELL = "powershell"
    JAVASCRIPT = "javascript"


@dataclass
class CodeAttempt:
    """Record of a single code generation attempt"""
    attempt_number: int
    code: str
    language: CodeLanguage

    # Static analysis
    syntax_valid: bool = False
    safety_valid: bool = False
    static_errors: List[str] = field(default_factory=list)

    # Execution
    executed: bool = False
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    execution_time_ms: float = 0

    # Result
    success: bool = False
    error_summary: str = ""


@dataclass
class RefinementResult:
    """Final result of the refinement loop"""
    task: str
    success: bool
    final_code: str
    language: CodeLanguage
    attempts: List[CodeAttempt]
    total_time_ms: float
    output: str = ""
    error: str = ""


class LLMInterface:
    """
    Interface for calling LLM to generate/fix code.

    Can be implemented with:
    - Anthropic API (Claude)
    - OpenAI API (GPT-4)
    - Local models (Ollama, llama.cpp)
    - Mock for testing
    """

    def generate_code(
        self,
        task: str,
        language: CodeLanguage,
        context: Optional[str] = None,
    ) -> str:
        """Generate code for a task"""
        raise NotImplementedError

    def fix_code(
        self,
        original_code: str,
        error_output: str,
        task: str,
        language: CodeLanguage,
        previous_attempts: List[CodeAttempt] = None,
    ) -> str:
        """Fix code based on error output"""
        raise NotImplementedError


class MockLLM(LLMInterface):
    """Mock LLM for testing - returns predefined responses"""

    def __init__(self, responses: List[str] = None):
        self.responses = responses or []
        self.call_count = 0

    def generate_code(self, task: str, language: CodeLanguage, context: str = None) -> str:
        if self.call_count < len(self.responses):
            code = self.responses[self.call_count]
            self.call_count += 1
            return code
        return "print('No more mock responses')"

    def fix_code(self, original_code: str, error_output: str, task: str,
                 language: CodeLanguage, previous_attempts: List[CodeAttempt] = None) -> str:
        return self.generate_code(task, language)


class AnthropicLLM(LLMInterface):
    """Anthropic Claude API implementation"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        return self._client

    def generate_code(
        self,
        task: str,
        language: CodeLanguage,
        context: Optional[str] = None,
    ) -> str:
        prompt = f"""Generate {language.value} code to accomplish this task:

TASK: {task}

Requirements:
- Write clean, working code
- Include error handling
- Print results to stdout
- Do NOT include markdown code blocks, just the raw code

{f"CONTEXT: {context}" if context else ""}

Respond with ONLY the code, nothing else."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        code = response.content[0].text
        # Strip markdown if LLM included it anyway
        code = self._strip_markdown(code)
        return code

    def fix_code(
        self,
        original_code: str,
        error_output: str,
        task: str,
        language: CodeLanguage,
        previous_attempts: List[CodeAttempt] = None,
    ) -> str:
        # Build context from previous attempts
        history = ""
        if previous_attempts:
            history = "\n\nPREVIOUS ATTEMPTS:\n"
            for attempt in previous_attempts[-3:]:  # Last 3 attempts
                history += f"\n--- Attempt {attempt.attempt_number} ---\n"
                history += f"Code:\n{attempt.code[:500]}...\n" if len(attempt.code) > 500 else f"Code:\n{attempt.code}\n"
                history += f"Error: {attempt.error_summary}\n"

        prompt = f"""Fix this {language.value} code that failed.

ORIGINAL TASK: {task}

CURRENT CODE:
```
{original_code}
```

ERROR OUTPUT:
```
{error_output}
```
{history}

Analyze the error and provide FIXED code.
- Fix the specific error shown
- Don't change working parts unnecessarily
- Include the complete fixed code

Respond with ONLY the fixed code, no explanations."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        code = response.content[0].text
        code = self._strip_markdown(code)
        return code

    def _strip_markdown(self, code: str) -> str:
        """Remove markdown code blocks if present"""
        # Remove ```python ... ``` or ```bash ... ```
        pattern = r'```(?:python|bash|powershell|javascript|js)?\n?(.*?)```'
        match = re.search(pattern, code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return code.strip()


class StaticAnalyzer:
    """Static code analysis - syntax and safety checks"""

    def __init__(self, verifier=None):
        self.verifier = verifier

    def check_syntax(self, code: str, language: CodeLanguage) -> Tuple[bool, List[str]]:
        """Check if code has valid syntax"""
        errors = []

        if language == CodeLanguage.PYTHON:
            try:
                ast.parse(code)
                return True, []
            except SyntaxError as e:
                errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
                return False, errors

        elif language == CodeLanguage.BASH:
            # Basic bash syntax check using bash -n
            try:
                result = subprocess.run(
                    ["bash", "-n", "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    errors.append(f"Bash syntax error: {result.stderr}")
                    return False, errors
                return True, []
            except FileNotFoundError:
                # Bash not available (Windows without WSL)
                return True, []  # Skip check
            except Exception as e:
                errors.append(f"Syntax check failed: {e}")
                return False, errors

        elif language == CodeLanguage.POWERSHELL:
            # PowerShell syntax check
            try:
                result = subprocess.run(
                    ["powershell", "-Command", f"[System.Management.Automation.PSParser]::Tokenize('{code}', [ref]$null)"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # This is a basic check - PS doesn't have a simple -n flag
                return True, []
            except Exception:
                return True, []  # Skip if PS not available

        return True, []  # Default: assume valid

    def check_safety(self, code: str, language: CodeLanguage) -> Tuple[bool, List[str]]:
        """Check if code is safe to execute"""
        errors = []

        # Dangerous patterns for any language
        dangerous_patterns = [
            (r'rm\s+-rf\s+/', "Recursive delete of root"),
            (r'rm\s+-rf\s+~', "Recursive delete of home"),
            (r':(){ :\|:& };:', "Fork bomb"),
            (r'dd\s+if=.*of=/dev/', "Direct disk write"),
            (r'mkfs\.', "Filesystem format"),
            (r'>\s*/dev/sd', "Overwrite disk"),
            (r'chmod\s+-R\s+777\s+/', "Dangerous permission change"),
            (r'curl.*\|\s*bash', "Pipe to shell"),
            (r'wget.*\|\s*bash', "Pipe to shell"),
            (r'eval\s*\(\s*base64', "Encoded eval"),
        ]

        # Python-specific dangers
        if language == CodeLanguage.PYTHON:
            dangerous_patterns.extend([
                (r'os\.system\s*\(\s*["\']rm\s+-rf', "Dangerous os.system call"),
                (r'subprocess.*shell\s*=\s*True.*rm\s+-rf', "Dangerous subprocess"),
                (r'__import__\s*\(\s*["\']os["\']\s*\)\.system', "Hidden os import"),
                (r'exec\s*\(\s*compile', "Dynamic code execution"),
                (r'eval\s*\(\s*input', "Eval user input"),
            ])

        for pattern, description in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(f"Dangerous pattern: {description}")

        # Use verifier if available
        if self.verifier:
            is_safe, reason = self.verifier.check_code(code, language.value)
            if not is_safe:
                errors.append(reason)

        return len(errors) == 0, errors


class CodeExecutor:
    """Execute code in isolated environment"""

    def __init__(self, sandbox_dir: Optional[str] = None, timeout: int = 30):
        self.timeout = timeout
        if sandbox_dir:
            self.sandbox_dir = Path(sandbox_dir)
        else:
            self.sandbox_dir = Path(tempfile.mkdtemp(prefix="jack_code_"))
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        code: str,
        language: CodeLanguage,
        stdin: str = "",
    ) -> Tuple[str, str, int, float]:
        """
        Execute code and capture output.

        Returns: (stdout, stderr, return_code, execution_time_ms)
        """
        start_time = time.time()

        if language == CodeLanguage.PYTHON:
            return self._execute_python(code, stdin)
        elif language == CodeLanguage.BASH:
            return self._execute_bash(code, stdin)
        elif language == CodeLanguage.POWERSHELL:
            return self._execute_powershell(code, stdin)
        else:
            return "", f"Unsupported language: {language}", 1, 0

    def _execute_python(self, code: str, stdin: str) -> Tuple[str, str, int, float]:
        """Execute Python code"""
        start = time.time()

        # Write code to temp file
        code_file = self.sandbox_dir / "script.py"
        code_file.write_text(code, encoding='utf-8')

        try:
            result = subprocess.run(
                [sys.executable, str(code_file)],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.sandbox_dir),
            )
            elapsed = (time.time() - start) * 1000
            return result.stdout, result.stderr, result.returncode, elapsed
        except subprocess.TimeoutExpired:
            elapsed = (time.time() - start) * 1000
            return "", f"Execution timed out after {self.timeout}s", 124, elapsed
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return "", str(e), 1, elapsed

    def _execute_bash(self, code: str, stdin: str) -> Tuple[str, str, int, float]:
        """Execute Bash code"""
        start = time.time()

        try:
            result = subprocess.run(
                ["bash", "-c", code],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.sandbox_dir),
            )
            elapsed = (time.time() - start) * 1000
            return result.stdout, result.stderr, result.returncode, elapsed
        except FileNotFoundError:
            return "", "Bash not available", 1, 0
        except subprocess.TimeoutExpired:
            elapsed = (time.time() - start) * 1000
            return "", f"Execution timed out after {self.timeout}s", 124, elapsed
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return "", str(e), 1, elapsed

    def _execute_powershell(self, code: str, stdin: str) -> Tuple[str, str, int, float]:
        """Execute PowerShell code"""
        start = time.time()

        try:
            result = subprocess.run(
                ["powershell", "-Command", code],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.sandbox_dir),
            )
            elapsed = (time.time() - start) * 1000
            return result.stdout, result.stderr, result.returncode, elapsed
        except FileNotFoundError:
            return "", "PowerShell not available", 1, 0
        except subprocess.TimeoutExpired:
            elapsed = (time.time() - start) * 1000
            return "", f"Execution timed out after {self.timeout}s", 124, elapsed
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return "", str(e), 1, elapsed

    def cleanup(self):
        """Clean up sandbox directory"""
        import shutil
        try:
            shutil.rmtree(self.sandbox_dir)
        except Exception:
            pass


class Evaluator:
    """Evaluate if code execution was successful"""

    def __init__(self, success_criteria: Optional[Callable[[str, str, int], bool]] = None):
        """
        Args:
            success_criteria: Custom function(stdout, stderr, return_code) -> bool
        """
        self.success_criteria = success_criteria

    def evaluate(
        self,
        stdout: str,
        stderr: str,
        return_code: int,
        task: str,
    ) -> Tuple[bool, str]:
        """
        Evaluate if execution was successful.

        Returns: (success, error_summary)
        """
        # Custom criteria if provided
        if self.success_criteria:
            success = self.success_criteria(stdout, stderr, return_code)
            if success:
                return True, ""

        # Default: non-zero return code = failure
        if return_code != 0:
            error_summary = self._summarize_error(stderr, stdout, return_code)
            return False, error_summary

        # Check for common error patterns in output
        error_patterns = [
            r'Traceback \(most recent call last\)',
            r'Error:',
            r'Exception:',
            r'FAILED',
            r'error:',
            r'fatal:',
        ]

        combined = stdout + stderr
        for pattern in error_patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                # But success if return code was 0 and output looks intentional
                if return_code == 0 and 'error' in task.lower():
                    continue  # Task might be about finding errors
                return False, self._summarize_error(stderr, stdout, return_code)

        return True, ""

    def _summarize_error(self, stderr: str, stdout: str, return_code: int) -> str:
        """Create concise error summary for LLM"""
        summary_parts = []

        if return_code != 0:
            summary_parts.append(f"Exit code: {return_code}")

        # Get last meaningful error line from stderr
        if stderr:
            lines = [l.strip() for l in stderr.strip().split('\n') if l.strip()]
            if lines:
                # Find the actual error message (often last line or line with 'Error')
                error_line = lines[-1]
                for line in reversed(lines):
                    if 'error' in line.lower() or 'exception' in line.lower():
                        error_line = line
                        break
                summary_parts.append(f"Error: {error_line[:200]}")

        # Check stdout for errors too
        if stdout and not stderr:
            if 'Traceback' in stdout:
                lines = stdout.strip().split('\n')
                summary_parts.append(f"Exception: {lines[-1][:200]}")

        return ' | '.join(summary_parts) if summary_parts else "Unknown error"


class CodeRefinementLoop:
    """
    The main loop: Generate → Test → Fix → Repeat

    This is Jack's core capability for code tasks.
    """

    def __init__(
        self,
        llm: LLMInterface,
        analyzer: Optional[StaticAnalyzer] = None,
        executor: Optional[CodeExecutor] = None,
        evaluator: Optional[Evaluator] = None,
        max_attempts: int = 5,
        verbose: bool = True,
    ):
        self.llm = llm
        self.analyzer = analyzer or StaticAnalyzer()
        self.executor = executor or CodeExecutor()
        self.evaluator = evaluator or Evaluator()
        self.max_attempts = max_attempts
        self.verbose = verbose

    def run(
        self,
        task: str,
        language: CodeLanguage = CodeLanguage.PYTHON,
        context: Optional[str] = None,
        success_criteria: Optional[Callable[[str, str, int], bool]] = None,
    ) -> RefinementResult:
        """
        Run the full refinement loop.

        Args:
            task: Natural language description of what code should do
            language: Programming language to use
            context: Additional context (file contents, API docs, etc.)
            success_criteria: Custom success check function

        Returns:
            RefinementResult with final code and all attempts
        """
        start_time = time.time()
        attempts: List[CodeAttempt] = []

        if success_criteria:
            self.evaluator.success_criteria = success_criteria

        current_code = None
        last_error = ""

        for attempt_num in range(1, self.max_attempts + 1):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Attempt {attempt_num}/{self.max_attempts}")
                print('='*50)

            attempt = CodeAttempt(
                attempt_number=attempt_num,
                code="",
                language=language,
            )

            # Step 1: Generate or fix code
            try:
                if current_code is None:
                    if self.verbose:
                        print("Generating initial code...")
                    code = self.llm.generate_code(task, language, context)
                else:
                    if self.verbose:
                        print(f"Fixing code based on error: {last_error[:100]}...")
                    code = self.llm.fix_code(
                        current_code,
                        last_error,
                        task,
                        language,
                        attempts
                    )

                attempt.code = code
                current_code = code

                if self.verbose:
                    print(f"Generated {len(code)} chars of code")

            except Exception as e:
                attempt.error_summary = f"LLM error: {e}"
                attempts.append(attempt)
                continue

            # Step 2: Static analysis
            syntax_ok, syntax_errors = self.analyzer.check_syntax(code, language)
            attempt.syntax_valid = syntax_ok
            attempt.static_errors.extend(syntax_errors)

            if not syntax_ok:
                if self.verbose:
                    print(f"Syntax error: {syntax_errors}")
                attempt.error_summary = f"Syntax: {syntax_errors[0]}"
                last_error = f"SYNTAX ERROR:\n{chr(10).join(syntax_errors)}\n\nCODE:\n{code}"
                attempts.append(attempt)
                continue

            safety_ok, safety_errors = self.analyzer.check_safety(code, language)
            attempt.safety_valid = safety_ok
            attempt.static_errors.extend(safety_errors)

            if not safety_ok:
                if self.verbose:
                    print(f"Safety check failed: {safety_errors}")
                attempt.error_summary = f"Safety: {safety_errors[0]}"
                last_error = f"SAFETY ERROR - Code blocked:\n{chr(10).join(safety_errors)}\n\nRewrite without dangerous operations."
                attempts.append(attempt)
                continue

            if self.verbose:
                print("Static checks passed")

            # Step 3: Execute
            if self.verbose:
                print("Executing code...")

            attempt.executed = True
            stdout, stderr, return_code, exec_time = self.executor.execute(code, language)

            attempt.stdout = stdout
            attempt.stderr = stderr
            attempt.return_code = return_code
            attempt.execution_time_ms = exec_time

            if self.verbose:
                print(f"Execution completed in {exec_time:.0f}ms (exit code: {return_code})")
                if stdout:
                    print(f"STDOUT: {stdout[:200]}{'...' if len(stdout) > 200 else ''}")
                if stderr:
                    print(f"STDERR: {stderr[:200]}{'...' if len(stderr) > 200 else ''}")

            # Step 4: Evaluate
            success, error_summary = self.evaluator.evaluate(stdout, stderr, return_code, task)
            attempt.success = success
            attempt.error_summary = error_summary

            attempts.append(attempt)

            if success:
                if self.verbose:
                    print("SUCCESS!")

                total_time = (time.time() - start_time) * 1000
                return RefinementResult(
                    task=task,
                    success=True,
                    final_code=code,
                    language=language,
                    attempts=attempts,
                    total_time_ms=total_time,
                    output=stdout,
                )

            # Prepare error context for next attempt
            last_error = f"""EXECUTION FAILED

STDOUT:
{stdout}

STDERR:
{stderr}

EXIT CODE: {return_code}

ERROR SUMMARY: {error_summary}
"""

            if self.verbose:
                print(f"Failed: {error_summary}")

        # Max attempts reached
        total_time = (time.time() - start_time) * 1000
        return RefinementResult(
            task=task,
            success=False,
            final_code=current_code or "",
            language=language,
            attempts=attempts,
            total_time_ms=total_time,
            error=f"Failed after {self.max_attempts} attempts. Last error: {last_error[:500]}",
        )

    def cleanup(self):
        """Clean up resources"""
        self.executor.cleanup()


# Convenience function
def refine_code(
    task: str,
    language: str = "python",
    api_key: Optional[str] = None,
    max_attempts: int = 5,
    verbose: bool = True,
) -> RefinementResult:
    """
    Convenience function to run code refinement loop.

    Args:
        task: What the code should do
        language: "python", "bash", or "powershell"
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        max_attempts: Maximum refinement attempts
        verbose: Print progress

    Returns:
        RefinementResult

    Example:
        result = refine_code("Download the Google homepage and count words")
        if result.success:
            print(result.final_code)
    """
    lang = CodeLanguage(language.lower())
    llm = AnthropicLLM(api_key=api_key)

    loop = CodeRefinementLoop(
        llm=llm,
        max_attempts=max_attempts,
        verbose=verbose,
    )

    try:
        return loop.run(task, lang)
    finally:
        loop.cleanup()
