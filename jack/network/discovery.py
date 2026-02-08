"""
Jack Node Discovery

Discovers other Jack nodes on the network using:
1. mDNS/Bonjour (local network)
2. UDP broadcast (fallback)
3. Static configuration (known nodes)

Like how robots find each other in swarm robotics.
"""

import socket
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta


@dataclass
class DiscoveredNode:
    """Information about a discovered Jack node"""
    node_id: str
    address: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    last_seen: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    substrate: str = "digital"  # 'digital', 'physical', 'iot'

    def is_alive(self, timeout_seconds: int = 60) -> bool:
        """Check if node has been seen recently"""
        return datetime.now() - self.last_seen < timedelta(seconds=timeout_seconds)


class NodeDiscovery:
    """
    Discovers and tracks Jack nodes on the network.

    Discovery methods:
    1. UDP broadcast on local network
    2. Response to announce messages
    3. Manual registration

    Future: mDNS, DHT for internet-scale
    """

    BROADCAST_PORT = 51337  # Jack discovery port
    ANNOUNCE_INTERVAL = 30  # Seconds between announcements

    def __init__(
        self,
        node_id: str,
        listen_port: int = 51338,
        capabilities: Optional[List[str]] = None,
    ):
        self.node_id = node_id
        self.listen_port = listen_port
        self.capabilities = capabilities or ["digital", "shell", "file", "http"]

        # Known nodes
        self.nodes: Dict[str, DiscoveredNode] = {}

        # Callbacks
        self.on_node_discovered: Optional[Callable[[DiscoveredNode], None]] = None
        self.on_node_lost: Optional[Callable[[str], None]] = None

        # Threading
        self._running = False
        self._listener_thread: Optional[threading.Thread] = None
        self._announcer_thread: Optional[threading.Thread] = None
        self._cleaner_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start discovery service"""
        self._running = True

        # Start listener for discovery messages
        self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listener_thread.start()

        # Start periodic announcer
        self._announcer_thread = threading.Thread(target=self._announce_loop, daemon=True)
        self._announcer_thread.start()

        # Start cleaner for stale nodes
        self._cleaner_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleaner_thread.start()

    def stop(self) -> None:
        """Stop discovery service"""
        self._running = False

    def _listen_loop(self) -> None:
        """Listen for discovery broadcasts"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(1.0)
            sock.bind(('', self.BROADCAST_PORT))

            while self._running:
                try:
                    data, addr = sock.recvfrom(1024)
                    self._handle_discovery_message(data, addr)
                except socket.timeout:
                    continue
                except Exception:
                    continue

        except Exception:
            pass  # Failed to bind, might be in use

    def _announce_loop(self) -> None:
        """Periodically announce presence"""
        while self._running:
            self._broadcast_announce()
            time.sleep(self.ANNOUNCE_INTERVAL)

    def _cleanup_loop(self) -> None:
        """Remove stale nodes"""
        while self._running:
            time.sleep(60)
            self._cleanup_stale_nodes()

    def _broadcast_announce(self) -> None:
        """Broadcast node announcement"""
        message = {
            "type": "announce",
            "node_id": self.node_id,
            "port": self.listen_port,
            "capabilities": self.capabilities,
            "timestamp": time.time(),
        }

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(
                json.dumps(message).encode(),
                ('<broadcast>', self.BROADCAST_PORT)
            )
            sock.close()
        except Exception:
            pass  # Network might not support broadcast

    def _handle_discovery_message(self, data: bytes, addr: tuple) -> None:
        """Handle incoming discovery message"""
        try:
            message = json.loads(data.decode())

            if message.get("type") == "announce":
                node_id = message.get("node_id")
                if node_id and node_id != self.node_id:
                    self._register_node(
                        node_id=node_id,
                        address=addr[0],
                        port=message.get("port", self.listen_port),
                        capabilities=message.get("capabilities", []),
                    )

            elif message.get("type") == "discover":
                # Someone is looking for nodes, respond
                self._broadcast_announce()

        except Exception:
            pass

    def _register_node(
        self,
        node_id: str,
        address: str,
        port: int,
        capabilities: List[str],
    ) -> None:
        """Register a discovered node"""
        is_new = node_id not in self.nodes

        self.nodes[node_id] = DiscoveredNode(
            node_id=node_id,
            address=address,
            port=port,
            capabilities=capabilities,
            last_seen=datetime.now(),
        )

        if is_new and self.on_node_discovered:
            self.on_node_discovered(self.nodes[node_id])

    def _cleanup_stale_nodes(self, timeout_seconds: int = 120) -> None:
        """Remove nodes not seen recently"""
        stale = []
        for node_id, node in self.nodes.items():
            if not node.is_alive(timeout_seconds):
                stale.append(node_id)

        for node_id in stale:
            del self.nodes[node_id]
            if self.on_node_lost:
                self.on_node_lost(node_id)

    def discover_now(self) -> None:
        """Trigger immediate discovery request"""
        message = {
            "type": "discover",
            "node_id": self.node_id,
            "timestamp": time.time(),
        }

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(
                json.dumps(message).encode(),
                ('<broadcast>', self.BROADCAST_PORT)
            )
            sock.close()
        except Exception:
            pass

    def register_static_node(
        self,
        node_id: str,
        address: str,
        port: int,
        capabilities: Optional[List[str]] = None,
    ) -> None:
        """Manually register a known node"""
        self._register_node(
            node_id=node_id,
            address=address,
            port=port,
            capabilities=capabilities or [],
        )

    def get_nodes(self) -> List[DiscoveredNode]:
        """Get list of all known nodes"""
        return list(self.nodes.values())

    def get_nodes_by_capability(self, capability: str) -> List[DiscoveredNode]:
        """Get nodes that have a specific capability"""
        return [
            node for node in self.nodes.values()
            if capability in node.capabilities
        ]

    def get_node(self, node_id: str) -> Optional[DiscoveredNode]:
        """Get specific node by ID"""
        return self.nodes.get(node_id)

    def find_best_node_for_task(self, task_type: str) -> Optional[DiscoveredNode]:
        """
        Find best node for a given task type.

        Considers:
        - Capabilities
        - Latency
        - Last seen time
        """
        candidates = self.get_nodes_by_capability(task_type)
        if not candidates:
            candidates = self.get_nodes()

        if not candidates:
            return None

        # Sort by latency and freshness
        candidates.sort(key=lambda n: (n.latency_ms, -n.last_seen.timestamp()))
        return candidates[0]
