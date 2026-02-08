"""
Jack Node - Network-enabled Jack Instance

Each Jack runs as a Node that can:
- Accept tasks from other Jacks
- Delegate tasks to specialized Jacks
- Share learned knowledge
- Coordinate multi-device operations

Like swarm robotics but for AI agents.
"""

import socket
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path

from .protocol import Message, MessageType, TaskRequest, TaskResponse
from .discovery import NodeDiscovery, DiscoveredNode


@dataclass
class NodeInfo:
    """Information about this Jack node"""
    node_id: str
    hostname: str
    capabilities: List[str]
    substrate: str  # 'digital', 'physical', 'iot'
    version: str = "0.1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "capabilities": self.capabilities,
            "substrate": self.substrate,
            "version": self.version,
        }


class JackNode:
    """
    Network-enabled Jack instance.

    Responsibilities:
    1. Listen for incoming tasks
    2. Execute tasks locally or delegate
    3. Participate in discovery
    4. Share knowledge with other nodes

    Future: Support for JackTheWalker physical nodes
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        listen_port: int = 51338,
        capabilities: Optional[List[str]] = None,
        substrate: str = "digital",
    ):
        """
        Initialize Jack node.

        Args:
            node_id: Unique node identifier (generated if None)
            listen_port: Port to listen for messages
            capabilities: List of capabilities (e.g., ['shell', 'python', 'gpu'])
            substrate: Type of substrate ('digital', 'physical', 'iot')
        """
        self.node_id = node_id or f"jack-{uuid.uuid4().hex[:8]}"
        self.listen_port = listen_port
        self.substrate = substrate

        # Default capabilities
        self.capabilities = capabilities or self._detect_capabilities()

        # Get hostname
        try:
            self.hostname = socket.gethostname()
        except Exception:
            self.hostname = "unknown"

        # Node info
        self.info = NodeInfo(
            node_id=self.node_id,
            hostname=self.hostname,
            capabilities=self.capabilities,
            substrate=self.substrate,
        )

        # Discovery service
        self.discovery = NodeDiscovery(
            node_id=self.node_id,
            listen_port=self.listen_port,
            capabilities=self.capabilities,
        )

        # Message handlers
        self._handlers: Dict[MessageType, Callable[[Message], Optional[Message]]] = {}
        self._register_default_handlers()

        # Task handler (set externally)
        self.on_task_received: Optional[Callable[[TaskRequest], TaskResponse]] = None

        # Server state
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._server_thread: Optional[threading.Thread] = None

        # Pending responses
        self._pending_responses: Dict[str, threading.Event] = {}
        self._responses: Dict[str, Message] = {}

    def _detect_capabilities(self) -> List[str]:
        """Auto-detect node capabilities"""
        caps = ["digital"]

        # Check for Python
        try:
            import sys
            caps.append("python")
            caps.append(f"python{sys.version_info.major}.{sys.version_info.minor}")
        except Exception:
            pass

        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                caps.append("cuda")
                caps.append("gpu")
        except ImportError:
            pass

        # Check for Docker
        try:
            import shutil
            if shutil.which("docker"):
                caps.append("docker")
        except Exception:
            pass

        # Basic capabilities
        caps.extend(["shell", "file", "http"])

        return list(set(caps))

    def _register_default_handlers(self) -> None:
        """Register default message handlers"""
        self._handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self._handlers[MessageType.TASK_REQUEST] = self._handle_task_request
        self._handlers[MessageType.STATE_QUERY] = self._handle_state_query

    def start(self) -> None:
        """Start the node (discovery + server)"""
        self._running = True

        # Start discovery
        self.discovery.start()

        # Start message server
        self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self._server_thread.start()

        print(f"Jack node {self.node_id} started on port {self.listen_port}")
        print(f"Capabilities: {', '.join(self.capabilities)}")

    def stop(self) -> None:
        """Stop the node"""
        self._running = False
        self.discovery.stop()

        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass

    def _server_loop(self) -> None:
        """Main server loop - accept connections"""
        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.settimeout(1.0)
            self._server_socket.bind(('0.0.0.0', self.listen_port))
            self._server_socket.listen(10)

            while self._running:
                try:
                    conn, addr = self._server_socket.accept()
                    # Handle in thread
                    threading.Thread(
                        target=self._handle_connection,
                        args=(conn, addr),
                        daemon=True
                    ).start()
                except socket.timeout:
                    continue
                except Exception:
                    if self._running:
                        continue

        except Exception as e:
            print(f"Server error: {e}")

    def _handle_connection(self, conn: socket.socket, addr: tuple) -> None:
        """Handle a single connection"""
        try:
            conn.settimeout(30.0)

            # Read message length
            length_bytes = conn.recv(4)
            if not length_bytes:
                return
            length = int.from_bytes(length_bytes, 'big')

            # Read message
            data = b''
            while len(data) < length:
                chunk = conn.recv(min(4096, length - len(data)))
                if not chunk:
                    break
                data += chunk

            # Parse and handle
            message = Message.from_json(data.decode('utf-8'))
            response = self._handle_message(message)

            # Send response if any
            if response:
                response_bytes = response.to_bytes()
                conn.sendall(response_bytes)

        except Exception:
            pass
        finally:
            conn.close()

    def _handle_message(self, message: Message) -> Optional[Message]:
        """Route message to appropriate handler"""
        # Check if this is a response to pending request
        if message.reply_to and message.reply_to in self._pending_responses:
            self._responses[message.reply_to] = message
            self._pending_responses[message.reply_to].set()
            return None

        # Get handler
        handler = self._handlers.get(message.type)
        if handler:
            return handler(message)

        return None

    def _handle_heartbeat(self, message: Message) -> Message:
        """Handle heartbeat message"""
        return message.create_response({
            "status": "alive",
            "timestamp": time.time(),
        })

    def _handle_task_request(self, message: Message) -> Message:
        """Handle task request"""
        try:
            task = TaskRequest.from_dict(message.payload)

            if self.on_task_received:
                response = self.on_task_received(task)
            else:
                response = TaskResponse(
                    success=False,
                    error="No task handler configured",
                )

            return message.create_response(response.to_dict())

        except Exception as e:
            return message.create_response(
                TaskResponse(success=False, error=str(e)).to_dict()
            )

    def _handle_state_query(self, message: Message) -> Message:
        """Handle state query"""
        return message.create_response({
            "node_info": self.info.to_dict(),
            "connected_nodes": len(self.discovery.nodes),
            "uptime": time.time(),  # Should track actual uptime
        })

    def send_message(
        self,
        target_node_id: str,
        message: Message,
        timeout: float = 30.0,
    ) -> Optional[Message]:
        """
        Send message to another node and wait for response.

        Args:
            target_node_id: Target node ID
            message: Message to send
            timeout: Timeout in seconds

        Returns:
            Response message or None
        """
        target = self.discovery.get_node(target_node_id)
        if not target:
            return None

        try:
            # Connect to target
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((target.address, target.port))

            # Send message
            message.sender_id = self.node_id
            message.recipient_id = target_node_id
            sock.sendall(message.to_bytes())

            # Wait for response
            length_bytes = sock.recv(4)
            if length_bytes:
                length = int.from_bytes(length_bytes, 'big')
                data = b''
                while len(data) < length:
                    chunk = sock.recv(min(4096, length - len(data)))
                    if not chunk:
                        break
                    data += chunk
                return Message.from_json(data.decode('utf-8'))

        except Exception:
            pass
        finally:
            sock.close()

        return None

    def request_task(
        self,
        target_node_id: str,
        task: TaskRequest,
        timeout: float = 60.0,
    ) -> Optional[TaskResponse]:
        """
        Request another node to execute a task.

        Args:
            target_node_id: Node to send task to
            task: Task request
            timeout: Timeout in seconds

        Returns:
            Task response or None
        """
        message = Message(
            type=MessageType.TASK_REQUEST,
            sender_id=self.node_id,
            recipient_id=target_node_id,
            payload=task.to_dict(),
        )

        response = self.send_message(target_node_id, message, timeout)
        if response:
            return TaskResponse.from_dict(response.payload)

        return None

    def broadcast_task(
        self,
        task: TaskRequest,
        capability_filter: Optional[str] = None,
    ) -> List[TaskResponse]:
        """
        Broadcast task to all capable nodes.

        Args:
            task: Task to broadcast
            capability_filter: Only send to nodes with this capability

        Returns:
            List of responses
        """
        responses = []

        if capability_filter:
            nodes = self.discovery.get_nodes_by_capability(capability_filter)
        else:
            nodes = self.discovery.get_nodes()

        for node in nodes:
            response = self.request_task(node.node_id, task)
            if response:
                responses.append(response)

        return responses

    def delegate_task(self, task: TaskRequest) -> Optional[TaskResponse]:
        """
        Delegate task to the best available node.

        Automatically finds node with appropriate capabilities.
        """
        # Find best node for task type
        best_node = self.discovery.find_best_node_for_task(task.task_type)
        if best_node:
            return self.request_task(best_node.node_id, task)

        return None

    def get_network_status(self) -> Dict[str, Any]:
        """Get network status summary"""
        nodes = self.discovery.get_nodes()
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "listen_port": self.listen_port,
            "capabilities": self.capabilities,
            "connected_nodes": len(nodes),
            "nodes": [
                {
                    "id": n.node_id,
                    "address": n.address,
                    "capabilities": n.capabilities,
                    "alive": n.is_alive(),
                }
                for n in nodes
            ],
        }
