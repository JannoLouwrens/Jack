"""
Windows OS Adapter

Provides Windows-specific implementations for Jack.
Uses PowerShell for advanced operations, cmd for simple ones.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import platform

from .base import (
    OSAdapter, ProcessInfo, DiskInfo, NetworkInterface, ServiceInfo
)


class WindowsAdapter(OSAdapter):
    """Windows-specific adapter for Jack"""

    # Command translations: Linux -> Windows
    COMMAND_MAP = {
        "ls": "dir",
        "cat": "type",
        "rm": "del",
        "rm -rf": "rmdir /s /q",
        "cp": "copy",
        "mv": "move",
        "mkdir -p": "mkdir",
        "pwd": "cd",
        "clear": "cls",
        "which": "where",
        "grep": "findstr",
        "ps": "tasklist",
        "kill": "taskkill /pid",
        "touch": "type nul >",
        "chmod": "icacls",  # Different syntax
        "df": "wmic logicaldisk get size,freespace,caption",
        "free": "systeminfo | findstr Memory",
        "uname": "ver",
        "whoami": "whoami",
        "env": "set",
        "export": "set",
    }

    # Protected system paths
    PROTECTED_PATHS = [
        "C:\\Windows\\System32",
        "C:\\Windows\\SysWOW64",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        "C:\\ProgramData",
        "C:\\$Recycle.Bin",
        "C:\\System Volume Information",
    ]

    @property
    def os_name(self) -> str:
        return "windows"

    @property
    def shell(self) -> str:
        return "cmd.exe"

    @property
    def shell_args(self) -> List[str]:
        return ["/c"]

    @property
    def path_separator(self) -> str:
        return "\\"

    @property
    def env_var_syntax(self) -> Tuple[str, str]:
        return ("%", "%")  # %VAR%

    def list_processes(self) -> List[ProcessInfo]:
        """List all running processes using tasklist and wmic"""
        processes = []
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'status', 'cmdline', 'username']):
                try:
                    info = proc.info
                    processes.append(ProcessInfo(
                        pid=info['pid'],
                        name=info['name'],
                        cpu_percent=info['cpu_percent'] or 0.0,
                        memory_mb=(info['memory_info'].rss / 1024 / 1024) if info['memory_info'] else 0.0,
                        status=info['status'],
                        cmdline=' '.join(info['cmdline']) if info['cmdline'] else None,
                        username=info['username']
                    ))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            # Fallback to tasklist
            result = subprocess.run(
                ["tasklist", "/fo", "csv", "/nh"],
                capture_output=True, text=True
            )
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.strip('"').split('","')
                    if len(parts) >= 5:
                        processes.append(ProcessInfo(
                            pid=int(parts[1]),
                            name=parts[0],
                            cpu_percent=0.0,
                            memory_mb=float(parts[4].replace(' K', '').replace(',', '')) / 1024,
                            status="running"
                        ))
        return processes

    def kill_process(self, pid: int, force: bool = False) -> bool:
        """Kill process using taskkill"""
        cmd = ["taskkill", "/pid", str(pid)]
        if force:
            cmd.append("/f")
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def get_process_tree(self, pid: int) -> List[ProcessInfo]:
        """Get process and children using wmic"""
        processes = []
        try:
            import psutil
            parent = psutil.Process(pid)
            processes.append(ProcessInfo(
                pid=parent.pid,
                name=parent.name(),
                cpu_percent=parent.cpu_percent(),
                memory_mb=parent.memory_info().rss / 1024 / 1024,
                status=parent.status()
            ))
            for child in parent.children(recursive=True):
                processes.append(ProcessInfo(
                    pid=child.pid,
                    name=child.name(),
                    cpu_percent=child.cpu_percent(),
                    memory_mb=child.memory_info().rss / 1024 / 1024,
                    status=child.status()
                ))
        except Exception:
            pass
        return processes

    def get_home_dir(self) -> Path:
        return Path(os.environ.get("USERPROFILE", "C:\\Users\\Default"))

    def get_temp_dir(self) -> Path:
        return Path(os.environ.get("TEMP", "C:\\Windows\\Temp"))

    def get_config_dir(self) -> Path:
        """Windows: Use AppData/Local/Jack"""
        appdata = os.environ.get("LOCALAPPDATA", str(self.get_home_dir() / "AppData" / "Local"))
        return Path(appdata) / "Jack"

    def normalize_path(self, path: str) -> str:
        """Normalize path for Windows"""
        # Convert forward slashes to backslashes
        path = path.replace("/", "\\")
        # Expand environment variables
        path = os.path.expandvars(path)
        # Expand user
        path = os.path.expanduser(path)
        # Get absolute path
        return str(Path(path).resolve())

    def get_disk_info(self) -> List[DiskInfo]:
        """Get disk information"""
        disks = []
        try:
            import psutil
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disks.append(DiskInfo(
                        path=partition.mountpoint,
                        total_gb=usage.total / (1024**3),
                        used_gb=usage.used / (1024**3),
                        free_gb=usage.free / (1024**3),
                        percent_used=usage.percent
                    ))
                except (PermissionError, OSError):
                    continue
        except ImportError:
            # Fallback
            result = subprocess.run(
                ["wmic", "logicaldisk", "get", "size,freespace,caption"],
                capture_output=True, text=True
            )
            # Parse wmic output (simplified)
        return disks

    def is_path_safe(self, path: str) -> Tuple[bool, str]:
        """Check if path is safe to access"""
        normalized = self.normalize_path(path)

        # Check protected paths
        for protected in self.PROTECTED_PATHS:
            if normalized.lower().startswith(protected.lower()):
                return False, f"Protected system path: {protected}"

        # Check for registry access attempts
        if normalized.lower().startswith("hk"):
            return False, "Registry access not allowed"

        return True, "Path is safe"

    def get_cpu_count(self) -> int:
        return os.cpu_count() or 1

    def get_memory_info(self) -> Dict[str, float]:
        """Get memory information"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
                "used_gb": mem.used / (1024**3),
                "percent": mem.percent
            }
        except ImportError:
            return {"total_gb": 0, "available_gb": 0, "used_gb": 0, "percent": 0}

    def get_network_interfaces(self) -> List[NetworkInterface]:
        """List network interfaces"""
        interfaces = []
        try:
            import psutil
            addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()
            for name, addr_list in addrs.items():
                ip = None
                mac = None
                for addr in addr_list:
                    if addr.family.name == 'AF_INET':
                        ip = addr.address
                    elif addr.family.name == 'AF_LINK':
                        mac = addr.address
                stat = stats.get(name)
                interfaces.append(NetworkInterface(
                    name=name,
                    ip_address=ip,
                    mac_address=mac,
                    is_up=stat.isup if stat else False,
                    speed_mbps=stat.speed if stat else None
                ))
        except ImportError:
            # Fallback to ipconfig
            result = subprocess.run(["ipconfig", "/all"], capture_output=True, text=True)
            # Basic parsing would go here
        return interfaces

    def get_hostname(self) -> str:
        return platform.node()

    def get_username(self) -> str:
        return os.environ.get("USERNAME", "unknown")

    def list_services(self) -> List[ServiceInfo]:
        """List Windows services using sc query"""
        services = []
        try:
            result = subprocess.run(
                ["sc", "query", "state=", "all"],
                capture_output=True, text=True
            )
            current_service = None
            for line in result.stdout.split('\n'):
                if "SERVICE_NAME:" in line:
                    current_service = line.split(":")[-1].strip()
                elif "STATE" in line and current_service:
                    status = "running" if "RUNNING" in line else "stopped"
                    services.append(ServiceInfo(
                        name=current_service,
                        status=status
                    ))
                    current_service = None
        except Exception:
            pass
        return services

    def get_service_status(self, name: str) -> Optional[ServiceInfo]:
        """Get specific service status"""
        result = subprocess.run(
            ["sc", "query", name],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            status = "running" if "RUNNING" in result.stdout else "stopped"
            return ServiceInfo(name=name, status=status)
        return None

    def start_service(self, name: str) -> bool:
        """Start a Windows service"""
        result = subprocess.run(
            ["sc", "start", name],
            capture_output=True, text=True
        )
        return result.returncode == 0

    def stop_service(self, name: str) -> bool:
        """Stop a Windows service"""
        result = subprocess.run(
            ["sc", "stop", name],
            capture_output=True, text=True
        )
        return result.returncode == 0

    def get_path_dirs(self) -> List[str]:
        """Get PATH directories"""
        path = os.environ.get("PATH", "")
        return [p.strip() for p in path.split(";") if p.strip()]

    def find_executable(self, name: str) -> Optional[str]:
        """Find executable using where command"""
        result = subprocess.run(
            ["where", name],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
        return None

    def list_installed_packages(self) -> List[Dict[str, str]]:
        """List installed programs from registry"""
        packages = []
        try:
            result = subprocess.run(
                ["wmic", "product", "get", "name,version"],
                capture_output=True, text=True, timeout=60
            )
            for line in result.stdout.split('\n')[1:]:
                parts = line.strip().rsplit(None, 1)
                if len(parts) >= 2:
                    packages.append({
                        "name": parts[0].strip(),
                        "version": parts[1].strip()
                    })
        except Exception:
            pass
        return packages

    def get_env_var(self, name: str) -> Optional[str]:
        return os.environ.get(name)

    def set_env_var(self, name: str, value: str, permanent: bool = False) -> bool:
        """Set environment variable"""
        os.environ[name] = value
        if permanent:
            # Use setx for permanent variables
            result = subprocess.run(
                ["setx", name, value],
                capture_output=True, text=True
            )
            return result.returncode == 0
        return True

    def translate_command(self, generic_command: str) -> str:
        """Translate Linux-style commands to Windows equivalents"""
        # Check for direct mappings
        for linux_cmd, windows_cmd in self.COMMAND_MAP.items():
            if generic_command.startswith(linux_cmd + " ") or generic_command == linux_cmd:
                return generic_command.replace(linux_cmd, windows_cmd, 1)

        # Handle specific patterns
        if generic_command.startswith("ls -la"):
            return "dir /a"
        if generic_command.startswith("rm -r "):
            path = generic_command[6:]
            return f"rmdir /s /q {path}"

        return generic_command
