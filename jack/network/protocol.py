"""
Jack Network Protocol

Simple message protocol for Jack-to-Jack communication.
Designed for:
- Low latency (embedded devices)
- Reliability (retry logic)
- Security (future: encryption)
"""

import json
import time
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List


class MessageType(Enum):
    """Types of messages in Jack network"""

    # Discovery
    ANNOUNCE = "announce"           # Node announcing itself
    DISCOVER = "discover"           # Request for node info
    HEARTBEAT = "heartbeat"         # Keep-alive ping

    # Tasks
    TASK_REQUEST = "task_request"   # Request to execute task
    TASK_RESPONSE = "task_response" # Response with result
    TASK_DELEGATE = "task_delegate" # Delegate task to another node
    TASK_CANCEL = "task_cancel"     # Cancel a running task

    # State
    STATE_QUERY = "state_query"     # Request node state
    STATE_UPDATE = "state_update"   # Broadcast state change

    # Memory
    MEMORY_SHARE = "memory_share"   # Share learned knowledge
    MEMORY_QUERY = "memory_query"   # Query for specific knowledge

    # Control
    SHUTDOWN = "shutdown"           # Request node shutdown
    RESTART = "restart"             # Request node restart
    UPDATE = "update"               # Request code update


@dataclass
class Message:
    """
    Network message structure.

    All Jack-to-Jack communication uses this format.
    """
    type: MessageType
    sender_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    recipient_id: Optional[str] = None  # None = broadcast
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None  # For responses
    ttl: int = 3  # Time-to-live for forwarding

    def to_json(self) -> str:
        """Serialize to JSON"""
        data = asdict(self)
        data['type'] = self.type.value
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Deserialize from JSON"""
        data = json.loads(json_str)
        data['type'] = MessageType(data['type'])
        return cls(**data)

    def to_bytes(self) -> bytes:
        """Convert to bytes for transmission"""
        json_str = self.to_json()
        # Simple framing: 4-byte length prefix + JSON
        length = len(json_str)
        return length.to_bytes(4, 'big') + json_str.encode('utf-8')

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Message':
        """Parse from bytes"""
        if len(data) < 4:
            raise ValueError("Message too short")
        length = int.from_bytes(data[:4], 'big')
        json_str = data[4:4+length].decode('utf-8')
        return cls.from_json(json_str)

    def create_response(self, payload: Dict[str, Any]) -> 'Message':
        """Create response to this message"""
        return Message(
            type=MessageType.TASK_RESPONSE,
            sender_id=self.recipient_id or "unknown",
            recipient_id=self.sender_id,
            payload=payload,
            reply_to=self.message_id,
        )


@dataclass
class TaskRequest:
    """Structured task request payload"""
    task_description: str
    task_type: str = "generic"  # 'shell', 'file', 'http', 'generic'
    priority: int = 5  # 1-10, higher = more urgent
    timeout_seconds: int = 60
    require_verification: bool = True
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TaskRequest':
        return cls(**d)


@dataclass
class TaskResponse:
    """Structured task response payload"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0
    actions_taken: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TaskResponse':
        return cls(**d)


# Helper functions for common messages

def create_announce_message(
    node_id: str,
    capabilities: List[str],
    address: str,
    port: int,
) -> Message:
    """Create node announcement message"""
    return Message(
        type=MessageType.ANNOUNCE,
        sender_id=node_id,
        payload={
            "capabilities": capabilities,
            "address": address,
            "port": port,
        }
    )


def create_task_request_message(
    sender_id: str,
    task: TaskRequest,
    recipient_id: Optional[str] = None,
) -> Message:
    """Create task request message"""
    return Message(
        type=MessageType.TASK_REQUEST,
        sender_id=sender_id,
        recipient_id=recipient_id,
        payload=task.to_dict(),
    )


def create_heartbeat_message(node_id: str) -> Message:
    """Create heartbeat message"""
    return Message(
        type=MessageType.HEARTBEAT,
        sender_id=node_id,
        payload={"timestamp": time.time()},
    )
