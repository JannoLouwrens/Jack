"""
Phase 0: Digital Physics Training

Learn by doing. Execute real actions, observe real results.

Key insight from research:
- STAR (2024): Action-relevant state representation
- ETrSR (2024): Task-relevant features only
- SWE-Agent: Minimal essential context

State is ACTION-CONDITIONED:
- Only capture what the action needs to know
- Track what CHANGED (delta), not everything
- Remove irrelevant noise (processes, env vars, etc.)
"""

import os
import json
import time
import random
import hashlib
import tempfile
import subprocess
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import shutil


# =============================================================================
# ACTION-CONDITIONED STATE
# =============================================================================

@dataclass
class ActionState:
    """
    Minimal state relevant to an action.

    Instead of capturing EVERYTHING, we capture:
    - Preconditions: What the action needs to succeed
    - Effects: What changed after execution
    """
    # Target info (the file/command being acted on)
    target: str = ""
    target_exists: bool = False
    target_size: int = 0
    target_readable: bool = False
    target_writable: bool = False

    # Parent directory (for file ops)
    parent_exists: bool = False
    parent_writable: bool = False

    # Content hash (to detect changes)
    content_hash: str = ""

    # For shell commands
    cwd: str = ""

    def to_vector(self) -> List[float]:
        """Convert to fixed-size numeric vector for neural network"""
        return [
            float(self.target_exists),
            float(self.target_size) / 1000000,  # Normalize to MB
            float(self.target_readable),
            float(self.target_writable),
            float(self.parent_exists),
            float(self.parent_writable),
            float(int(self.content_hash[:8], 16) / 0xFFFFFFFF) if self.content_hash else 0,
        ]

    @staticmethod
    def vector_size() -> int:
        return 7

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ActionEffect:
    """What changed after an action"""
    success: bool = False
    error_type: str = ""  # none, not_found, permission, timeout, etc.
    error_message: str = ""

    # Size delta
    size_before: int = 0
    size_after: int = 0

    # Content changed?
    content_changed: bool = False

    # Output (for shell/read)
    output_length: int = 0
    output_hash: str = ""

    # Timing
    duration_ms: float = 0

    def to_vector(self) -> List[float]:
        """Convert to fixed-size numeric vector"""
        error_types = ["none", "not_found", "permission", "timeout", "syntax", "unknown"]
        error_idx = error_types.index(self.error_type) if self.error_type in error_types else 5

        return [
            float(self.success),
            float(error_idx) / len(error_types),
            float(self.size_after - self.size_before) / 1000000,
            float(self.content_changed),
            float(self.output_length) / 10000,
            float(self.duration_ms) / 1000,
        ]

    @staticmethod
    def vector_size() -> int:
        return 6

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Transition:
    """
    A single learning example: (state, action) -> effect

    Minimal and focused. No bloat.
    """
    # Action info
    action_type: str  # shell_run, file_read, file_write, http_request, get_state
    action_target: str  # path or command
    action_params: Dict[str, Any] = field(default_factory=dict)

    # State BEFORE (relevant to this action only)
    state_before: ActionState = field(default_factory=ActionState)

    # Effect (what happened)
    effect: ActionEffect = field(default_factory=ActionEffect)

    # Output content (truncated)
    output: str = ""

    def to_dict(self) -> Dict:
        return {
            "action_type": self.action_type,
            "action_target": self.action_target,
            "action_params": self.action_params,
            "state_before": self.state_before.to_dict(),
            "effect": self.effect.to_dict(),
            "output": self.output[:500],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Transition':
        return cls(
            action_type=d["action_type"],
            action_target=d["action_target"],
            action_params=d.get("action_params", {}),
            state_before=ActionState(**d.get("state_before", {})),
            effect=ActionEffect(**d.get("effect", {})),
            output=d.get("output", ""),
        )


# For backwards compatibility
@dataclass
class StateSnapshot:
    """Legacy - kept for compatibility but deprecated"""
    timestamp: float = 0
    files: Dict[str, Dict] = field(default_factory=dict)
    processes: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    memory_usage_mb: float = 0
    cpu_percent: float = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'StateSnapshot':
        return cls(**d)


# =============================================================================
# TRANSITION BUFFER
# =============================================================================

class TransitionBuffer:
    """Store transitions for training"""

    def __init__(self, max_size: int = 100000, save_path: Optional[str] = None):
        self.max_size = max_size
        self.buffer: List[Transition] = []
        self.save_path = save_path

        if save_path and os.path.exists(save_path):
            self.load()

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size: int) -> List[Transition]:
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        return random.sample(self.buffer, batch_size)

    def save(self) -> None:
        if self.save_path:
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
            data = [t.to_dict() for t in self.buffer]
            with open(self.save_path, 'w') as f:
                json.dump(data, f)

    def load(self) -> None:
        if self.save_path and os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            self.buffer = [Transition.from_dict(d) for d in data]

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# SANDBOX
# =============================================================================

class DigitalSandbox:
    """
    Isolated environment for safe experimentation.

    Executes real actions, returns real results.
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir:
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.base_dir = Path(tempfile.mkdtemp(prefix="jack_sandbox_"))

        self.sandbox_dir = self.base_dir / f"sandbox_{datetime.now().strftime('%H%M%S')}"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self._setup_test_files()

    def _setup_test_files(self) -> None:
        """Create test files for experimentation"""
        test_files = {
            "readme.txt": "This is a test file.\nLine 2.\nLine 3.\n",
            "data/config.json": '{"name": "test", "value": 42}',
            "data/numbers.txt": "1\n2\n3\n4\n5\n",
            "scripts/hello.py": 'print("Hello, World!")\n',
            "logs/app.log": "2024-01-01 INFO: Started\n2024-01-02 ERROR: Failed\n",
        }

        for path, content in test_files.items():
            full_path = self.sandbox_dir / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

    def _get_file_state(self, path: str) -> ActionState:
        """Get state relevant to a file operation"""
        full_path = self.sandbox_dir / path
        parent_path = full_path.parent

        state = ActionState(
            target=path,
            cwd=str(self.sandbox_dir),
        )

        try:
            resolved = full_path.resolve()
            if not str(resolved).startswith(str(self.sandbox_dir.resolve())):
                return state  # Path escape - return minimal state
        except:
            pass

        # Target file info
        if full_path.exists():
            state.target_exists = True
            state.target_size = full_path.stat().st_size
            state.target_readable = os.access(full_path, os.R_OK)
            state.target_writable = os.access(full_path, os.W_OK)
            try:
                content = full_path.read_bytes()
                state.content_hash = hashlib.md5(content).hexdigest()
            except:
                pass

        # Parent directory info
        if parent_path.exists():
            state.parent_exists = True
            state.parent_writable = os.access(parent_path, os.W_OK)

        return state

    def _get_shell_state(self, command: str) -> ActionState:
        """Get state relevant to a shell command"""
        return ActionState(
            target=command.split()[0] if command else "",
            cwd=str(self.sandbox_dir),
            parent_exists=True,
            parent_writable=True,
        )

    def read_file(self, path: str) -> Transition:
        """Read file and return transition"""
        state_before = self._get_file_state(path)
        full_path = self.sandbox_dir / path

        start = time.time()
        effect = ActionEffect(size_before=state_before.target_size)
        output = ""

        try:
            # Check path escape
            resolved = full_path.resolve()
            if not str(resolved).startswith(str(self.sandbox_dir.resolve())):
                raise PermissionError("Path escape")

            content = full_path.read_text()
            output = content[:1000]

            effect.success = True
            effect.error_type = "none"
            effect.output_length = len(content)
            effect.output_hash = hashlib.md5(content.encode()).hexdigest()[:16]
            effect.size_after = len(content)

        except FileNotFoundError:
            effect.error_type = "not_found"
            effect.error_message = f"File not found: {path}"
        except PermissionError as e:
            effect.error_type = "permission"
            effect.error_message = str(e)
        except Exception as e:
            effect.error_type = "unknown"
            effect.error_message = str(e)

        effect.duration_ms = (time.time() - start) * 1000

        return Transition(
            action_type="file_read",
            action_target=path,
            state_before=state_before,
            effect=effect,
            output=output,
        )

    def write_file(self, path: str, content: str) -> Transition:
        """Write file and return transition"""
        state_before = self._get_file_state(path)
        full_path = self.sandbox_dir / path

        start = time.time()
        effect = ActionEffect(size_before=state_before.target_size)

        try:
            resolved = full_path.resolve()
            if not str(resolved).startswith(str(self.sandbox_dir.resolve())):
                raise PermissionError("Path escape")

            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

            effect.success = True
            effect.error_type = "none"
            effect.size_after = len(content)
            effect.content_changed = True

        except PermissionError as e:
            effect.error_type = "permission"
            effect.error_message = str(e)
        except Exception as e:
            effect.error_type = "unknown"
            effect.error_message = str(e)

        effect.duration_ms = (time.time() - start) * 1000

        return Transition(
            action_type="file_write",
            action_target=path,
            action_params={"content_length": len(content)},
            state_before=state_before,
            effect=effect,
        )

    def run_shell(self, command: str, timeout: int = 5) -> Transition:
        """Run shell command and return transition"""
        state_before = self._get_shell_state(command)

        start = time.time()
        effect = ActionEffect()
        output = ""

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.sandbox_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = result.stdout[:1000]
            effect.success = result.returncode == 0
            effect.error_type = "none" if effect.success else "syntax"
            effect.error_message = result.stderr[:200] if result.stderr else ""
            effect.output_length = len(result.stdout)
            effect.output_hash = hashlib.md5(result.stdout.encode()).hexdigest()[:16]

        except subprocess.TimeoutExpired:
            effect.error_type = "timeout"
            effect.error_message = f"Command timed out after {timeout}s"
        except Exception as e:
            effect.error_type = "unknown"
            effect.error_message = str(e)

        effect.duration_ms = (time.time() - start) * 1000

        return Transition(
            action_type="shell_run",
            action_target=command,
            state_before=state_before,
            effect=effect,
            output=output,
        )

    def list_files(self) -> List[str]:
        """List all files in sandbox"""
        files = []
        for root, _, filenames in os.walk(self.sandbox_dir):
            for f in filenames:
                rel = os.path.relpath(os.path.join(root, f), self.sandbox_dir)
                files.append(rel.replace("\\", "/"))
        return files

    def reset(self) -> None:
        """Reset sandbox"""
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self._setup_test_files()

    def cleanup(self) -> None:
        """Remove sandbox"""
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)


# =============================================================================
# EXPERIENCE COLLECTOR
# =============================================================================

class ExperienceCollector:
    """Collect experiences through systematic experimentation"""

    def __init__(self, sandbox: DigitalSandbox, buffer: TransitionBuffer):
        self.sandbox = sandbox
        self.buffer = buffer

    def collect_file_read_experiences(self, n: int = 50) -> int:
        """Learn file reading behavior"""
        collected = 0
        files = self.sandbox.list_files()

        # Read existing files
        for _ in range(n // 2):
            if not files:
                break
            path = random.choice(files)
            transition = self.sandbox.read_file(path)
            self.buffer.add(transition)
            collected += 1

        # Try non-existent files (learn errors)
        fake_paths = [
            "nonexistent.txt",
            "missing/file.txt",
            f"random_{random.randint(0,999)}.txt",
            "../escape.txt",
            "data/nope.json",
        ]
        for path in fake_paths[:n // 2]:
            transition = self.sandbox.read_file(path)
            self.buffer.add(transition)
            collected += 1

        return collected

    def collect_file_write_experiences(self, n: int = 50) -> int:
        """Learn file writing behavior"""
        collected = 0

        scenarios = [
            ("new_file.txt", "Hello, World!"),
            ("data/new.json", '{"key": "value"}'),
            ("deep/nested/file.txt", "Nested content"),
            ("readme.txt", "Overwritten content"),
            ("test.py", "print('test')\n"),
        ]

        for path, content in scenarios[:n]:
            transition = self.sandbox.write_file(path, content)
            self.buffer.add(transition)
            collected += 1

        # Random writes
        for i in range(n - len(scenarios)):
            path = f"generated_{i}.txt"
            content = f"Generated content {random.randint(0, 1000)}"
            transition = self.sandbox.write_file(path, content)
            self.buffer.add(transition)
            collected += 1

        return collected

    def collect_shell_experiences(self, n: int = 50) -> int:
        """Learn shell command behavior"""
        collected = 0

        # Safe commands to try
        commands = [
            "echo hello",
            "echo $PATH",
            "pwd",
            "ls",
            "ls -la",
            "cat readme.txt",
            "head -1 readme.txt",
            "wc -l readme.txt",
            "find . -name '*.txt'",
            "grep test readme.txt",
            # Error cases
            "nonexistent_command",
            "cat nonexistent.txt",
            "ls /nonexistent",
            "invalid syntax {{{}}}",
        ]

        for cmd in commands[:n]:
            transition = self.sandbox.run_shell(cmd)
            self.buffer.add(transition)
            collected += 1

        return collected

    def collect_sequence_experiences(self, n: int = 50) -> int:
        """Learn action sequences (write then read, etc.)"""
        collected = 0

        for i in range(n):
            path = f"sequence_{i}.txt"
            content = f"Sequence test {random.randint(0, 1000)}"

            # Write
            t1 = self.sandbox.write_file(path, content)
            self.buffer.add(t1)
            collected += 1

            # Read back
            t2 = self.sandbox.read_file(path)
            self.buffer.add(t2)
            collected += 1

            if collected >= n * 2:
                break

        return collected


# =============================================================================
# TRAINER (Legacy compatibility)
# =============================================================================

class Phase0Trainer:
    """Legacy trainer - use JackTrainer from train_transformer.py instead"""

    def __init__(
        self,
        sandbox: Optional[DigitalSandbox] = None,
        buffer_path: Optional[str] = None,
        world_model: Optional[Any] = None,
    ):
        self.sandbox = sandbox or DigitalSandbox()
        self.buffer = TransitionBuffer(save_path=buffer_path)
        self.collector = ExperienceCollector(self.sandbox, self.buffer)
        self.world_model = world_model  # JackBrain

        self.training_stats = {
            'total_transitions': 0,
            'successful_transitions': 0,
            'failed_transitions': 0,
        }

    def collect_phase(self, num_samples: int = 500) -> Dict[str, int]:
        """Collect experiences"""
        n = num_samples // 4

        stats = {
            'file_read': self.collector.collect_file_read_experiences(n),
            'file_write': self.collector.collect_file_write_experiences(n),
            'shell': self.collector.collect_shell_experiences(n),
            'sequence': self.collector.collect_sequence_experiences(n),
        }

        self.buffer.save()
        return stats

    def cleanup(self):
        self.sandbox.cleanup()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Phase 0: Action-Conditioned State")
    print("=" * 50)

    sandbox = DigitalSandbox()
    buffer = TransitionBuffer()
    collector = ExperienceCollector(sandbox, buffer)

    print(f"\nSandbox: {sandbox.sandbox_dir}")
    print(f"Files: {sandbox.list_files()}")

    # Test file read
    print("\n--- File Read ---")
    t = sandbox.read_file("readme.txt")
    print(f"Action: {t.action_type} {t.action_target}")
    print(f"State before: exists={t.state_before.target_exists}, size={t.state_before.target_size}")
    print(f"Effect: success={t.effect.success}, output_len={t.effect.output_length}")
    print(f"State vector: {t.state_before.to_vector()}")
    print(f"Effect vector: {t.effect.to_vector()}")

    # Test file read error
    print("\n--- File Read (Error) ---")
    t = sandbox.read_file("nonexistent.txt")
    print(f"Effect: success={t.effect.success}, error={t.effect.error_type}")

    # Test file write
    print("\n--- File Write ---")
    t = sandbox.write_file("new_file.txt", "Hello!")
    print(f"Effect: success={t.effect.success}, size_after={t.effect.size_after}")

    # Test shell
    print("\n--- Shell ---")
    t = sandbox.run_shell("echo hello")
    print(f"Effect: success={t.effect.success}, output_len={t.effect.output_length}")
    print(f"Output: {t.output}")

    # Collect batch
    print("\n--- Collect 100 experiences ---")
    collector.collect_file_read_experiences(25)
    collector.collect_file_write_experiences(25)
    collector.collect_shell_experiences(25)
    collector.collect_sequence_experiences(25)
    print(f"Buffer size: {len(buffer)}")

    # Show stats
    success = sum(1 for t in buffer.buffer if t.effect.success)
    print(f"Success rate: {success}/{len(buffer)} = {100*success/len(buffer):.1f}%")

    sandbox.cleanup()
    print("\nDone.")
