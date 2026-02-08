"""
Jack Network - Distributed Agent Mesh

Enables multiple Jack instances to communicate and collaborate:
- IoT devices with Jack brains
- Desktop computers
- Servers
- Robot bodies (JackTheWalker)

Architecture:
- Each Jack is a Node with unique ID
- Nodes discover each other via mDNS/broadcast
- Messages use simple JSON protocol over TCP
- Tasks can be delegated to specialized nodes
"""

from .node import JackNode, NodeInfo
from .discovery import NodeDiscovery
from .protocol import Message, MessageType

__all__ = [
    "JackNode",
    "NodeInfo",
    "NodeDiscovery",
    "Message",
    "MessageType",
]
