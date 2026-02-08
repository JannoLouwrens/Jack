"""
Jack OS Adapters - Cross-platform abstraction layer

Each adapter provides OS-specific implementations for:
- Process management
- File system operations
- System information
- Network configuration
"""

from .base import OSAdapter
from .factory import get_adapter, AdapterFactory

__all__ = [
    "OSAdapter",
    "get_adapter",
    "AdapterFactory",
]
