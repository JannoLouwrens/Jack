"""
EXECUTOR - The 5 Primitive Actions

Jack's entire capability comes from these 5 actions:
1. shell_run(command)       - Run any shell command
2. file_read(path)          - Read any file
3. file_write(path, content) - Write any file (including code!)
4. http_request(...)        - Call any API
5. get_state()              - Observe system state

+ Code generation = UNLIMITED CAPABILITY

Like robot joints - you define the primitives, Jack learns how to use them.
"""

import os
import subprocess
import shutil
import platform
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class ActionResult:
    """Result of any action Jack takes"""
    success: bool
    action_type: str
    data: Any = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "action_type": self.action_type,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SystemState:
    """Observable state of the system"""
    platform: str
    hostname: str
    cwd: str
    user: str

    # File system
    files_in_cwd: List[str]
    disk_usage: Dict[str, Any]

    # Processes
    process_count: int
    top_processes: List[Dict]

    # Memory
    memory_percent: float

    # Network
    network_connected: bool

    # Environment
    env_vars: Dict[str, str]
    path_dirs: List[str]

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "platform": self.platform,
            "hostname": self.hostname,
            "cwd": self.cwd,
            "user": self.user,
            "files_in_cwd": self.files_in_cwd,
            "disk_usage": self.disk_usage,
            "process_count": self.process_count,
            "top_processes": self.top_processes,
            "memory_percent": self.memory_percent,
            "network_connected": self.network_connected,
            "timestamp": self.timestamp.isoformat(),
        }


class Executor:
    """
    Executes Jack's 5 primitive actions.

    These 5 actions + code generation = unlimited capability.
    """

    def __init__(self, verifier=None, working_dir: str = None):
        self.verifier = verifier
        self.working_dir = working_dir or os.getcwd()
        self.platform = platform.system()  # 'Windows', 'Linux', 'Darwin'

        # History of actions (for learning)
        self.history: List[ActionResult] = []

    # ═══════════════════════════════════════════════════════════════════════
    # ACTION 1: SHELL RUN
    # ═══════════════════════════════════════════════════════════════════════

    def shell_run(
        self,
        command: str,
        timeout: int = 60,
        capture_output: bool = True,
        cwd: str = None
    ) -> ActionResult:
        """
        Run any shell command.

        This single action gives Jack access to:
        - Every command line tool installed
        - Package managers (apt, pip, npm, brew...)
        - Git, docker, kubectl...
        - Python, node, any interpreter
        - System administration
        - ANYTHING the shell can do

        Args:
            command: The command to run
            timeout: Max seconds to wait
            capture_output: Whether to capture stdout/stderr
            cwd: Working directory

        Returns:
            ActionResult with stdout, stderr, return code
        """
        # Verify if verifier is attached
        if self.verifier:
            verify_result = self.verifier.check_shell(command)
            if not verify_result.allowed:
                return ActionResult(
                    success=False,
                    action_type="shell_run",
                    error=f"BLOCKED: {verify_result.reason}",
                    data={"command": command, "blocked": True}
                )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                cwd=cwd or self.working_dir,
            )

            action_result = ActionResult(
                success=(result.returncode == 0),
                action_type="shell_run",
                data={
                    "command": command,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                },
                error=result.stderr if result.returncode != 0 else None,
            )

        except subprocess.TimeoutExpired:
            action_result = ActionResult(
                success=False,
                action_type="shell_run",
                error=f"Timeout after {timeout}s",
                data={"command": command}
            )
        except Exception as e:
            action_result = ActionResult(
                success=False,
                action_type="shell_run",
                error=str(e),
                data={"command": command}
            )

        self.history.append(action_result)
        return action_result

    # ═══════════════════════════════════════════════════════════════════════
    # ACTION 2: FILE READ
    # ═══════════════════════════════════════════════════════════════════════

    def file_read(
        self,
        path: str,
        encoding: str = "utf-8",
        binary: bool = False
    ) -> ActionResult:
        """
        Read any file.

        Args:
            path: Path to file
            encoding: Text encoding
            binary: Read as bytes

        Returns:
            ActionResult with file contents
        """
        path = os.path.expanduser(path)

        # Verify
        if self.verifier:
            verify_result = self.verifier.check_file_read(path)
            if not verify_result.allowed:
                return ActionResult(
                    success=False,
                    action_type="file_read",
                    error=f"BLOCKED: {verify_result.reason}",
                    data={"path": path, "blocked": True}
                )

        try:
            mode = "rb" if binary else "r"
            with open(path, mode, encoding=None if binary else encoding) as f:
                content = f.read()

            action_result = ActionResult(
                success=True,
                action_type="file_read",
                data={
                    "path": path,
                    "content": content,
                    "size": len(content),
                }
            )

        except FileNotFoundError:
            action_result = ActionResult(
                success=False,
                action_type="file_read",
                error=f"File not found: {path}",
                data={"path": path}
            )
        except PermissionError:
            action_result = ActionResult(
                success=False,
                action_type="file_read",
                error=f"Permission denied: {path}",
                data={"path": path}
            )
        except Exception as e:
            action_result = ActionResult(
                success=False,
                action_type="file_read",
                error=str(e),
                data={"path": path}
            )

        self.history.append(action_result)
        return action_result

    # ═══════════════════════════════════════════════════════════════════════
    # ACTION 3: FILE WRITE
    # ═══════════════════════════════════════════════════════════════════════

    def file_write(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        append: bool = False,
        make_executable: bool = False
    ) -> ActionResult:
        """
        Write any file - INCLUDING CODE.

        This is where Jack's unlimited capability comes from.
        Jack can write Python, Bash, JavaScript, anything.
        Then run it with shell_run().

        Args:
            path: Path to file
            content: Content to write
            encoding: Text encoding
            append: Append instead of overwrite
            make_executable: chmod +x after writing

        Returns:
            ActionResult
        """
        path = os.path.expanduser(path)

        # Verify
        if self.verifier:
            verify_result = self.verifier.check_file_write(path, content)
            if not verify_result.allowed:
                return ActionResult(
                    success=False,
                    action_type="file_write",
                    error=f"BLOCKED: {verify_result.reason}",
                    data={"path": path, "blocked": True}
                )

        try:
            # Create parent directories if needed
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

            mode = "a" if append else "w"
            with open(path, mode, encoding=encoding) as f:
                f.write(content)

            # Make executable if requested (Unix)
            if make_executable and self.platform != "Windows":
                os.chmod(path, 0o755)

            action_result = ActionResult(
                success=True,
                action_type="file_write",
                data={
                    "path": path,
                    "size": len(content),
                    "mode": "append" if append else "write",
                }
            )

        except PermissionError:
            action_result = ActionResult(
                success=False,
                action_type="file_write",
                error=f"Permission denied: {path}",
                data={"path": path}
            )
        except Exception as e:
            action_result = ActionResult(
                success=False,
                action_type="file_write",
                error=str(e),
                data={"path": path}
            )

        self.history.append(action_result)
        return action_result

    # ═══════════════════════════════════════════════════════════════════════
    # ACTION 4: HTTP REQUEST
    # ═══════════════════════════════════════════════════════════════════════

    def http_request(
        self,
        method: str,
        url: str,
        body: Dict = None,
        headers: Dict = None,
        timeout: int = 30
    ) -> ActionResult:
        """
        Make any HTTP request - access any API.

        This gives Jack access to:
        - REST APIs
        - GraphQL
        - Webhooks
        - Cloud services (AWS, GCP, Azure)
        - AI APIs (OpenAI, Anthropic)
        - Any web service

        Args:
            method: GET, POST, PUT, DELETE, etc.
            url: The URL
            body: JSON body for POST/PUT
            headers: HTTP headers
            timeout: Request timeout

        Returns:
            ActionResult with response
        """
        if not HAS_REQUESTS:
            return ActionResult(
                success=False,
                action_type="http_request",
                error="requests library not installed. Run: pip install requests",
                data={"url": url}
            )

        # Verify
        if self.verifier:
            verify_result = self.verifier.check_http_request(method, url, body, headers)
            if not verify_result.allowed:
                return ActionResult(
                    success=False,
                    action_type="http_request",
                    error=f"BLOCKED: {verify_result.reason}",
                    data={"url": url, "blocked": True}
                )

        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                json=body,
                headers=headers,
                timeout=timeout
            )

            # Try to parse JSON response
            try:
                response_data = response.json()
            except:
                response_data = response.text

            action_result = ActionResult(
                success=(200 <= response.status_code < 300),
                action_type="http_request",
                data={
                    "url": url,
                    "method": method.upper(),
                    "status_code": response.status_code,
                    "response": response_data,
                    "headers": dict(response.headers),
                },
                error=None if response.ok else f"HTTP {response.status_code}"
            )

        except requests.Timeout:
            action_result = ActionResult(
                success=False,
                action_type="http_request",
                error=f"Timeout after {timeout}s",
                data={"url": url}
            )
        except Exception as e:
            action_result = ActionResult(
                success=False,
                action_type="http_request",
                error=str(e),
                data={"url": url}
            )

        self.history.append(action_result)
        return action_result

    # ═══════════════════════════════════════════════════════════════════════
    # ACTION 5: GET STATE
    # ═══════════════════════════════════════════════════════════════════════

    def get_state(self) -> SystemState:
        """
        Observe the current system state.

        This is Jack's "eyes" - seeing what's happening on the computer.
        Essential for the world model to predict outcomes.

        Returns:
            SystemState with full system observation
        """
        # Basic info
        state = {
            "platform": self.platform,
            "hostname": platform.node(),
            "cwd": os.getcwd(),
            "user": os.environ.get("USER") or os.environ.get("USERNAME", "unknown"),
        }

        # Files in current directory
        try:
            state["files_in_cwd"] = os.listdir(".")[:100]  # Limit for performance
        except:
            state["files_in_cwd"] = []

        # Disk usage
        try:
            if HAS_PSUTIL:
                disk = psutil.disk_usage("/")
                state["disk_usage"] = {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": disk.percent,
                }
            else:
                state["disk_usage"] = {}
        except:
            state["disk_usage"] = {}

        # Processes
        try:
            if HAS_PSUTIL:
                procs = list(psutil.process_iter(['pid', 'name', 'cpu_percent']))
                state["process_count"] = len(procs)
                state["top_processes"] = sorted(
                    [{"pid": p.info['pid'], "name": p.info['name'], "cpu": p.info.get('cpu_percent', 0)}
                     for p in procs[:20]],
                    key=lambda x: x.get('cpu', 0),
                    reverse=True
                )[:10]
            else:
                state["process_count"] = 0
                state["top_processes"] = []
        except:
            state["process_count"] = 0
            state["top_processes"] = []

        # Memory
        try:
            if HAS_PSUTIL:
                mem = psutil.virtual_memory()
                state["memory_percent"] = mem.percent
            else:
                state["memory_percent"] = 0.0
        except:
            state["memory_percent"] = 0.0

        # Network
        try:
            if HAS_REQUESTS:
                requests.get("https://google.com", timeout=2)
                state["network_connected"] = True
            else:
                state["network_connected"] = False
        except:
            state["network_connected"] = False

        # Environment (filtered for safety)
        safe_vars = ["PATH", "HOME", "USER", "SHELL", "LANG", "TERM"]
        state["env_vars"] = {k: os.environ.get(k, "") for k in safe_vars}

        # PATH directories
        state["path_dirs"] = os.environ.get("PATH", "").split(os.pathsep)

        return SystemState(**state)

    # ═══════════════════════════════════════════════════════════════════════
    # HELPER: Discover commands
    # ═══════════════════════════════════════════════════════════════════════

    def discover_commands(self) -> List[str]:
        """
        Discover what commands are available on this system.
        Jack uses this to learn what's possible.
        """
        commands = set()

        for path_dir in os.environ.get("PATH", "").split(os.pathsep):
            if os.path.isdir(path_dir):
                try:
                    for item in os.listdir(path_dir):
                        item_path = os.path.join(path_dir, item)
                        if os.access(item_path, os.X_OK):
                            commands.add(item)
                except PermissionError:
                    continue

        return sorted(list(commands))

    def get_command_help(self, command: str) -> str:
        """
        Get help for a command.
        Jack uses this to learn what commands do.
        """
        # Try --help first
        result = self.shell_run(f"{command} --help", timeout=5)
        if result.success:
            return result.data.get("stdout", "")

        # Try -h
        result = self.shell_run(f"{command} -h", timeout=5)
        if result.success:
            return result.data.get("stdout", "")

        # Try man page (Unix)
        if self.platform != "Windows":
            result = self.shell_run(f"man {command} 2>/dev/null | head -100", timeout=5)
            if result.success:
                return result.data.get("stdout", "")

        return ""


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_executor(with_verifier: bool = True) -> Executor:
    """Create an executor with optional verifier"""
    if with_verifier:
        from .verifier import Verifier
        verifier = Verifier()
    else:
        verifier = None

    return Executor(verifier=verifier)


# ═══════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("JACK EXECUTOR - 5 Primitive Actions")
    print("="*60)

    executor = Executor()

    # Test get_state
    print("\n[TEST] get_state()")
    state = executor.get_state()
    print(f"  Platform: {state.platform}")
    print(f"  User: {state.user}")
    print(f"  CWD: {state.cwd}")
    print(f"  Files: {len(state.files_in_cwd)}")
    print(f"  Processes: {state.process_count}")

    # Test shell_run
    print("\n[TEST] shell_run('echo hello')")
    result = executor.shell_run("echo hello")
    print(f"  Success: {result.success}")
    print(f"  Output: {result.data.get('stdout', '').strip()}")

    # Test file operations
    print("\n[TEST] file_write + file_read")
    test_path = "/tmp/jack_test.txt" if executor.platform != "Windows" else "C:\\temp\\jack_test.txt"
    executor.file_write(test_path, "Hello from Jack!")
    result = executor.file_read(test_path)
    print(f"  Written and read: {result.data.get('content', '')}")

    # Test discover commands
    print("\n[TEST] discover_commands()")
    commands = executor.discover_commands()
    print(f"  Found {len(commands)} commands")
    print(f"  Sample: {commands[:10]}")

    print("\n" + "="*60)
    print("[OK] Executor working")
    print("="*60)
