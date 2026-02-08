"""
Base OS Adapter - Abstract interface for OS-specific operations

Philosophy:
- Jack's core brain is the same across all platforms
- Adapters translate intent into OS-specific commands
- Like how JackTheWalker has motor controllers for different robot bodies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import subprocess


@dataclass
class ProcessInfo:
    """Information about a running process"""
    pid: int
    name: str
    cpu_percent: float
    memory_mb: float
    status: str
    cmdline: Optional[str] = None
    username: Optional[str] = None


@dataclass
class DiskInfo:
    """Disk/partition information"""
    path: str
    total_gb: float
    used_gb: float
    free_gb: float
    percent_used: float


@dataclass
class NetworkInterface:
    """Network interface information"""
    name: str
    ip_address: Optional[str]
    mac_address: Optional[str]
    is_up: bool
    speed_mbps: Optional[int] = None


@dataclass
class ServiceInfo:
    """System service information"""
    name: str
    status: str  # running, stopped, disabled
    startup_type: Optional[str] = None  # auto, manual, disabled


class OSAdapter(ABC):
    """
    Abstract base class for OS-specific operations.

    Design principle: Same interface, different implementations.
    Jack's brain doesn't need to know if it's on Windows/Linux/Mac.
    """

    @property
    @abstractmethod
    def os_name(self) -> str:
        """Return OS name: 'windows', 'linux', or 'macos'"""
        pass

    @property
    @abstractmethod
    def shell(self) -> str:
        """Return default shell path"""
        pass

    @property
    @abstractmethod
    def shell_args(self) -> List[str]:
        """Return shell arguments for command execution"""
        pass

    @property
    @abstractmethod
    def path_separator(self) -> str:
        """Return path separator for this OS"""
        pass

    @property
    @abstractmethod
    def env_var_syntax(self) -> Tuple[str, str]:
        """Return (prefix, suffix) for environment variables. E.g., ('$', '') for Linux, ('%', '%') for Windows"""
        pass

    # === Process Management ===

    @abstractmethod
    def list_processes(self) -> List[ProcessInfo]:
        """List all running processes"""
        pass

    @abstractmethod
    def kill_process(self, pid: int, force: bool = False) -> bool:
        """Kill a process by PID"""
        pass

    @abstractmethod
    def get_process_tree(self, pid: int) -> List[ProcessInfo]:
        """Get process and all its children"""
        pass

    # === File System ===

    @abstractmethod
    def get_home_dir(self) -> Path:
        """Get user's home directory"""
        pass

    @abstractmethod
    def get_temp_dir(self) -> Path:
        """Get system temp directory"""
        pass

    @abstractmethod
    def get_config_dir(self) -> Path:
        """Get OS-appropriate config directory for Jack"""
        pass

    @abstractmethod
    def normalize_path(self, path: str) -> str:
        """Normalize path for this OS"""
        pass

    @abstractmethod
    def get_disk_info(self) -> List[DiskInfo]:
        """Get disk/partition information"""
        pass

    @abstractmethod
    def is_path_safe(self, path: str) -> Tuple[bool, str]:
        """Check if path is safe to access. Returns (is_safe, reason)"""
        pass

    # === System Information ===

    @abstractmethod
    def get_cpu_count(self) -> int:
        """Get number of CPU cores"""
        pass

    @abstractmethod
    def get_memory_info(self) -> Dict[str, float]:
        """Get memory info: total_gb, available_gb, used_gb, percent"""
        pass

    @abstractmethod
    def get_network_interfaces(self) -> List[NetworkInterface]:
        """List network interfaces"""
        pass

    @abstractmethod
    def get_hostname(self) -> str:
        """Get system hostname"""
        pass

    @abstractmethod
    def get_username(self) -> str:
        """Get current username"""
        pass

    # === Services ===

    @abstractmethod
    def list_services(self) -> List[ServiceInfo]:
        """List system services"""
        pass

    @abstractmethod
    def get_service_status(self, name: str) -> Optional[ServiceInfo]:
        """Get status of a specific service"""
        pass

    @abstractmethod
    def start_service(self, name: str) -> bool:
        """Start a service (may require admin)"""
        pass

    @abstractmethod
    def stop_service(self, name: str) -> bool:
        """Stop a service (may require admin)"""
        pass

    # === Command Discovery ===

    @abstractmethod
    def get_path_dirs(self) -> List[str]:
        """Get directories in PATH"""
        pass

    @abstractmethod
    def find_executable(self, name: str) -> Optional[str]:
        """Find executable by name"""
        pass

    @abstractmethod
    def list_installed_packages(self) -> List[Dict[str, str]]:
        """List installed packages/programs"""
        pass

    # === Environment ===

    @abstractmethod
    def get_env_var(self, name: str) -> Optional[str]:
        """Get environment variable"""
        pass

    @abstractmethod
    def set_env_var(self, name: str, value: str, permanent: bool = False) -> bool:
        """Set environment variable"""
        pass

    # === Utility Methods (implemented in base) ===

    def run_command(
        self,
        command: str,
        timeout: int = 60,
        capture_output: bool = True,
        cwd: Optional[str] = None
    ) -> subprocess.CompletedProcess:
        """
        Run a command using OS-appropriate shell.
        This is a convenience wrapper - Executor.shell_run is preferred.
        """
        return subprocess.run(
            [self.shell] + self.shell_args + [command],
            timeout=timeout,
            capture_output=capture_output,
            text=True,
            cwd=cwd
        )

    def translate_command(self, generic_command: str) -> str:
        """
        Translate generic/Linux-style command to OS-specific equivalent.
        Override in subclasses for complex translations.

        Examples:
        - 'ls' -> 'dir' on Windows
        - 'cat file.txt' -> 'type file.txt' on Windows
        """
        return generic_command  # Default: no translation

    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""
        memory = self.get_memory_info()
        return {
            "os": self.os_name,
            "hostname": self.get_hostname(),
            "username": self.get_username(),
            "cpu_cores": self.get_cpu_count(),
            "memory_total_gb": memory.get("total_gb", 0),
            "memory_available_gb": memory.get("available_gb", 0),
            "home_dir": str(self.get_home_dir()),
            "temp_dir": str(self.get_temp_dir()),
        }
