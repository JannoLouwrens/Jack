"""
OS Adapter Factory

Automatically selects and creates the appropriate adapter for the current OS.
"""

import platform
from typing import Optional

from .base import OSAdapter


class AdapterFactory:
    """Factory for creating OS-specific adapters"""

    _instance: Optional[OSAdapter] = None

    @classmethod
    def create(cls, os_override: Optional[str] = None) -> OSAdapter:
        """
        Create an OS adapter.

        Args:
            os_override: Force specific OS adapter ('windows', 'linux', 'macos')

        Returns:
            Appropriate OSAdapter instance
        """
        os_name = os_override or platform.system().lower()

        if os_name in ('windows', 'win32', 'nt'):
            from .windows import WindowsAdapter
            return WindowsAdapter()

        elif os_name in ('linux', 'linux2'):
            from .linux import LinuxAdapter
            return LinuxAdapter()

        elif os_name in ('darwin', 'macos', 'macosx'):
            from .macos import MacOSAdapter
            return MacOSAdapter()

        else:
            # Default to Linux adapter for Unix-like systems
            from .linux import LinuxAdapter
            return LinuxAdapter()

    @classmethod
    def get_singleton(cls, os_override: Optional[str] = None) -> OSAdapter:
        """Get or create singleton adapter instance"""
        if cls._instance is None:
            cls._instance = cls.create(os_override)
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton (useful for testing)"""
        cls._instance = None


def get_adapter(os_override: Optional[str] = None) -> OSAdapter:
    """
    Convenience function to get the OS adapter.

    Usage:
        from jack.adapters import get_adapter

        adapter = get_adapter()
        print(adapter.os_name)  # 'windows', 'linux', or 'macos'
    """
    return AdapterFactory.get_singleton(os_override)
