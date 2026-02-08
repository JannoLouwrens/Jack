"""
Linux OS Adapter

Provides Linux-specific implementations for Jack.
Native environment - most commands work directly.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import platform

from .base import (
    OSAdapter, ProcessInfo, DiskInfo, NetworkInterface, ServiceInfo
)


class LinuxAdapter(OSAdapter):
    """Linux-specific adapter for Jack"""

    # Protected system paths
    PROTECTED_PATHS = [
        "/bin",
        "/sbin",
        "/usr/bin",
        "/usr/sbin",
        "/boot",
        "/dev",
        "/proc",
        "/sys",
        "/etc/passwd",
        "/etc/shadow",
        "/etc/sudoers",
    ]

    @property
    def os_name(self) -> str:
        return "linux"

    @property
    def shell(self) -> str:
        return os.environ.get("SHELL", "/bin/bash")

    @property
    def shell_args(self) -> List[str]:
        return ["-c"]

    @property
    def path_separator(self) -> str:
        return "/"

    @property
    def env_var_syntax(self) -> Tuple[str, str]:
        return ("$", "")  # $VAR

    def list_processes(self) -> List[ProcessInfo]:
        """List all running processes"""
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
            # Fallback to ps
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True, text=True
            )
            for line in result.stdout.strip().split('\n')[1:]:
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    processes.append(ProcessInfo(
                        pid=int(parts[1]),
                        name=parts[10].split()[0],
                        cpu_percent=float(parts[2]),
                        memory_mb=float(parts[5]) / 1024,  # RSS in KB
                        status="running",
                        username=parts[0]
                    ))
        return processes

    def kill_process(self, pid: int, force: bool = False) -> bool:
        """Kill process using kill command"""
        signal = "-9" if force else "-15"
        result = subprocess.run(
            ["kill", signal, str(pid)],
            capture_output=True, text=True
        )
        return result.returncode == 0

    def get_process_tree(self, pid: int) -> List[ProcessInfo]:
        """Get process and children"""
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
        return Path(os.environ.get("HOME", "/home"))

    def get_temp_dir(self) -> Path:
        return Path(os.environ.get("TMPDIR", "/tmp"))

    def get_config_dir(self) -> Path:
        """Linux: Use ~/.config/jack or XDG_CONFIG_HOME"""
        xdg_config = os.environ.get("XDG_CONFIG_HOME", str(self.get_home_dir() / ".config"))
        return Path(xdg_config) / "jack"

    def normalize_path(self, path: str) -> str:
        """Normalize path for Linux"""
        # Expand user
        path = os.path.expanduser(path)
        # Expand environment variables
        path = os.path.expandvars(path)
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
            # Fallback to df
            result = subprocess.run(
                ["df", "-BG"],
                capture_output=True, text=True
            )
            for line in result.stdout.split('\n')[1:]:
                parts = line.split()
                if len(parts) >= 6:
                    disks.append(DiskInfo(
                        path=parts[5],
                        total_gb=float(parts[1].rstrip('G')),
                        used_gb=float(parts[2].rstrip('G')),
                        free_gb=float(parts[3].rstrip('G')),
                        percent_used=float(parts[4].rstrip('%'))
                    ))
        return disks

    def is_path_safe(self, path: str) -> Tuple[bool, str]:
        """Check if path is safe to access"""
        normalized = self.normalize_path(path)

        # Check protected paths
        for protected in self.PROTECTED_PATHS:
            if normalized == protected or normalized.startswith(protected + "/"):
                # Allow reading some paths, just not writing
                if "shadow" in normalized or "sudoers" in normalized:
                    return False, f"Sensitive system file: {protected}"

        # Check for system directories
        if normalized.startswith("/root") and os.getuid() != 0:
            return False, "Cannot access root home directory"

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
            # Fallback to /proc/meminfo
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split()
                if parts[0] in ['MemTotal:', 'MemFree:', 'MemAvailable:']:
                    info[parts[0].rstrip(':')] = float(parts[1]) / (1024**2)  # KB to GB
            return {
                "total_gb": info.get("MemTotal", 0),
                "available_gb": info.get("MemAvailable", info.get("MemFree", 0)),
                "used_gb": info.get("MemTotal", 0) - info.get("MemAvailable", info.get("MemFree", 0)),
                "percent": 0
            }

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
                    elif addr.family.name == 'AF_PACKET':
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
            # Fallback to ip command
            result = subprocess.run(["ip", "addr"], capture_output=True, text=True)
            # Basic parsing would go here
        return interfaces

    def get_hostname(self) -> str:
        return platform.node()

    def get_username(self) -> str:
        return os.environ.get("USER", "unknown")

    def list_services(self) -> List[ServiceInfo]:
        """List systemd services"""
        services = []
        try:
            result = subprocess.run(
                ["systemctl", "list-units", "--type=service", "--all", "--no-pager", "--plain"],
                capture_output=True, text=True
            )
            for line in result.stdout.split('\n')[1:]:
                parts = line.split()
                if len(parts) >= 4 and parts[0].endswith('.service'):
                    status = "running" if parts[3] == "running" else "stopped"
                    services.append(ServiceInfo(
                        name=parts[0].replace('.service', ''),
                        status=status
                    ))
        except Exception:
            pass
        return services

    def get_service_status(self, name: str) -> Optional[ServiceInfo]:
        """Get specific service status"""
        result = subprocess.run(
            ["systemctl", "is-active", name],
            capture_output=True, text=True
        )
        status = result.stdout.strip()
        return ServiceInfo(
            name=name,
            status="running" if status == "active" else "stopped"
        )

    def start_service(self, name: str) -> bool:
        """Start a systemd service"""
        result = subprocess.run(
            ["sudo", "systemctl", "start", name],
            capture_output=True, text=True
        )
        return result.returncode == 0

    def stop_service(self, name: str) -> bool:
        """Stop a systemd service"""
        result = subprocess.run(
            ["sudo", "systemctl", "stop", name],
            capture_output=True, text=True
        )
        return result.returncode == 0

    def get_path_dirs(self) -> List[str]:
        """Get PATH directories"""
        path = os.environ.get("PATH", "")
        return [p.strip() for p in path.split(":") if p.strip()]

    def find_executable(self, name: str) -> Optional[str]:
        """Find executable using which command"""
        result = subprocess.run(
            ["which", name],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None

    def list_installed_packages(self) -> List[Dict[str, str]]:
        """List installed packages (apt/dpkg for Debian-based)"""
        packages = []

        # Try dpkg first (Debian/Ubuntu)
        result = subprocess.run(
            ["dpkg", "-l"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n')[5:]:
                parts = line.split()
                if len(parts) >= 3 and parts[0].startswith('ii'):
                    packages.append({
                        "name": parts[1],
                        "version": parts[2]
                    })
            return packages

        # Try rpm (RHEL/Fedora)
        result = subprocess.run(
            ["rpm", "-qa", "--queryformat", "%{NAME} %{VERSION}\n"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                parts = line.split()
                if len(parts) >= 2:
                    packages.append({
                        "name": parts[0],
                        "version": parts[1]
                    })

        return packages

    def get_env_var(self, name: str) -> Optional[str]:
        return os.environ.get(name)

    def set_env_var(self, name: str, value: str, permanent: bool = False) -> bool:
        """Set environment variable"""
        os.environ[name] = value
        if permanent:
            # Append to .bashrc
            bashrc = self.get_home_dir() / ".bashrc"
            try:
                with open(bashrc, 'a') as f:
                    f.write(f'\nexport {name}="{value}"\n')
                return True
            except Exception:
                return False
        return True

    def translate_command(self, generic_command: str) -> str:
        """No translation needed for Linux - native environment"""
        return generic_command
