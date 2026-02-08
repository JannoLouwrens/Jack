"""
MEMORY - Jack's Learning Memory

Three-tier memory system:
1. Short-term: Current conversation/task context
2. Long-term: Persistent knowledge, preferences, patterns
3. Episodic: What happened (action history, outcomes)

Plus: Tool memory - saved code/scripts Jack creates

Research: "Memory should be treated as three different problems"
- Short-term: in-task context needed to finish the current job
- Long-term: consented preferences, defaults
- Organizational: policies, playbooks, SOPs
"""

import os
import json
import sqlite3
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class MemoryItem:
    """A single memory"""
    content: str
    memory_type: str  # "short", "long", "episodic", "tool"
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5  # 0-1, for prioritization
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryItem":
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ActionMemory:
    """Memory of an action and its outcome"""
    action_type: str
    action_data: Dict
    result_success: bool
    result_data: Any
    state_before: Dict
    state_after: Dict
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "action_type": self.action_type,
            "action_data": self.action_data,
            "result_success": self.result_success,
            "result_data": str(self.result_data)[:1000],  # Truncate
            "state_before": self.state_before,
            "state_after": self.state_after,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Tool:
    """A saved tool (code) Jack created"""
    name: str
    description: str
    code: str
    language: str  # "python", "bash", etc.
    created: datetime = field(default_factory=datetime.now)
    last_used: datetime = None
    use_count: int = 0
    success_rate: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "language": self.language,
            "created": self.created.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "use_count": self.use_count,
            "success_rate": self.success_rate,
        }


class Memory:
    """
    Jack's memory system.

    Stores:
    - Short-term: Current context (in memory)
    - Long-term: Persistent knowledge (SQLite)
    - Episodic: Action history (SQLite)
    - Tools: Saved code (files + SQLite index)
    """

    def __init__(self, jack_dir: str = None):
        self.jack_dir = Path(jack_dir or os.path.expanduser("~/.jack"))
        self.jack_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.jack_dir / "memory.db"
        self.tools_dir = self.jack_dir / "tools"
        self.tools_dir.mkdir(exist_ok=True)

        # Initialize database
        self._init_db()

        # Short-term memory (in-memory)
        self.short_term: List[MemoryItem] = []
        self.short_term_max = 100

        # Context window for current task
        self.context: List[Dict] = []
        self.context_max = 50

    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Long-term memories
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                tags TEXT,
                metadata TEXT,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Episodic memories (action history)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                action_data TEXT,
                result_success INTEGER,
                result_data TEXT,
                state_before TEXT,
                state_after TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Tools index
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                language TEXT,
                file_path TEXT,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                use_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 1.0
            )
        """)

        # Command knowledge (what Jack learned about commands)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT UNIQUE NOT NULL,
                description TEXT,
                usage TEXT,
                examples TEXT,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                last_used TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    # ═══════════════════════════════════════════════════════════════════════
    # SHORT-TERM MEMORY
    # ═══════════════════════════════════════════════════════════════════════

    def add_short_term(self, content: str, importance: float = 0.5, tags: List[str] = None):
        """Add to short-term memory"""
        item = MemoryItem(
            content=content,
            memory_type="short",
            importance=importance,
            tags=tags or []
        )
        self.short_term.append(item)

        # Trim if too long (keep important ones)
        if len(self.short_term) > self.short_term_max:
            self.short_term.sort(key=lambda x: x.importance, reverse=True)
            self.short_term = self.short_term[:self.short_term_max]

    def get_short_term(self, n: int = 10) -> List[MemoryItem]:
        """Get recent short-term memories"""
        return self.short_term[-n:]

    def clear_short_term(self):
        """Clear short-term memory (end of task)"""
        # Optionally promote important items to long-term
        for item in self.short_term:
            if item.importance > 0.8:
                self.add_long_term(item.content, item.importance, item.tags)
        self.short_term = []

    # ═══════════════════════════════════════════════════════════════════════
    # LONG-TERM MEMORY
    # ═══════════════════════════════════════════════════════════════════════

    def add_long_term(self, content: str, importance: float = 0.5, tags: List[str] = None, metadata: Dict = None):
        """Add to long-term memory (persistent)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO long_term (content, importance, tags, metadata)
            VALUES (?, ?, ?, ?)
        """, (
            content,
            importance,
            json.dumps(tags or []),
            json.dumps(metadata or {})
        ))

        conn.commit()
        conn.close()

    def search_long_term(self, query: str, limit: int = 10) -> List[Dict]:
        """Search long-term memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Simple text search (could be upgraded to vector search with ChromaDB)
        cursor.execute("""
            SELECT content, importance, tags, metadata, created
            FROM long_term
            WHERE content LIKE ?
            ORDER BY importance DESC, created DESC
            LIMIT ?
        """, (f"%{query}%", limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                "content": row[0],
                "importance": row[1],
                "tags": json.loads(row[2]),
                "metadata": json.loads(row[3]),
                "created": row[4],
            })

        conn.close()
        return results

    # ═══════════════════════════════════════════════════════════════════════
    # EPISODIC MEMORY (Action History)
    # ═══════════════════════════════════════════════════════════════════════

    def add_episode(self, action: ActionMemory):
        """Remember an action and its outcome"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO episodic (action_type, action_data, result_success, result_data, state_before, state_after)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            action.action_type,
            json.dumps(action.action_data),
            1 if action.result_success else 0,
            str(action.result_data)[:1000],
            json.dumps(action.state_before),
            json.dumps(action.state_after),
        ))

        conn.commit()
        conn.close()

    def get_similar_episodes(self, action_type: str, limit: int = 10) -> List[Dict]:
        """Find similar past actions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT action_type, action_data, result_success, result_data, timestamp
            FROM episodic
            WHERE action_type = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (action_type, limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                "action_type": row[0],
                "action_data": json.loads(row[1]),
                "success": bool(row[2]),
                "result": row[3],
                "timestamp": row[4],
            })

        conn.close()
        return results

    def get_success_rate(self, action_type: str) -> float:
        """Get success rate for an action type"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                SUM(CASE WHEN result_success = 1 THEN 1 ELSE 0 END) as successes,
                COUNT(*) as total
            FROM episodic
            WHERE action_type = ?
        """, (action_type,))

        row = cursor.fetchone()
        conn.close()

        if row and row[1] > 0:
            return row[0] / row[1]
        return 0.5  # No data, assume 50%

    # ═══════════════════════════════════════════════════════════════════════
    # TOOL MEMORY (Saved Code)
    # ═══════════════════════════════════════════════════════════════════════

    def save_tool(self, name: str, description: str, code: str, language: str = "python"):
        """Save a tool Jack created"""
        # Determine file extension
        ext = {"python": ".py", "bash": ".sh", "javascript": ".js"}.get(language, ".txt")
        file_path = self.tools_dir / f"{name}{ext}"

        # Write code to file
        with open(file_path, "w") as f:
            f.write(code)

        # Index in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO tools (name, description, language, file_path)
            VALUES (?, ?, ?, ?)
        """, (name, description, language, str(file_path)))

        conn.commit()
        conn.close()

        return file_path

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a saved tool"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name, description, language, file_path, created, last_used, use_count, success_rate
            FROM tools
            WHERE name = ?
        """, (name,))

        row = cursor.fetchone()
        conn.close()

        if row:
            file_path = Path(row[3])
            if file_path.exists():
                with open(file_path, "r") as f:
                    code = f.read()

                return Tool(
                    name=row[0],
                    description=row[1],
                    code=code,
                    language=row[2],
                    created=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
                    last_used=datetime.fromisoformat(row[5]) if row[5] else None,
                    use_count=row[6],
                    success_rate=row[7]
                )
        return None

    def list_tools(self) -> List[Dict]:
        """List all saved tools"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name, description, language, use_count, success_rate
            FROM tools
            ORDER BY use_count DESC
        """)

        results = []
        for row in cursor.fetchall():
            results.append({
                "name": row[0],
                "description": row[1],
                "language": row[2],
                "use_count": row[3],
                "success_rate": row[4],
            })

        conn.close()
        return results

    def record_tool_use(self, name: str, success: bool):
        """Record that a tool was used"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current stats
        cursor.execute("SELECT use_count, success_rate FROM tools WHERE name = ?", (name,))
        row = cursor.fetchone()

        if row:
            use_count = row[0] + 1
            # Exponential moving average for success rate
            old_rate = row[1]
            new_rate = 0.9 * old_rate + 0.1 * (1.0 if success else 0.0)

            cursor.execute("""
                UPDATE tools
                SET use_count = ?, success_rate = ?, last_used = CURRENT_TIMESTAMP
                WHERE name = ?
            """, (use_count, new_rate, name))

        conn.commit()
        conn.close()

    # ═══════════════════════════════════════════════════════════════════════
    # COMMAND KNOWLEDGE
    # ═══════════════════════════════════════════════════════════════════════

    def learn_command(self, command: str, description: str = "", usage: str = "", examples: str = ""):
        """Learn about a command"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO commands (command, description, usage, examples)
            VALUES (?, ?, ?, ?)
        """, (command, description, usage, examples))

        conn.commit()
        conn.close()

    def get_command_knowledge(self, command: str) -> Optional[Dict]:
        """Get knowledge about a command"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT command, description, usage, examples, success_count, fail_count
            FROM commands
            WHERE command = ?
        """, (command,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "command": row[0],
                "description": row[1],
                "usage": row[2],
                "examples": row[3],
                "success_count": row[4],
                "fail_count": row[5],
            }
        return None

    def record_command_result(self, command: str, success: bool):
        """Record command success/failure"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if success:
            cursor.execute("""
                UPDATE commands SET success_count = success_count + 1, last_used = CURRENT_TIMESTAMP
                WHERE command = ?
            """, (command,))
        else:
            cursor.execute("""
                UPDATE commands SET fail_count = fail_count + 1, last_used = CURRENT_TIMESTAMP
                WHERE command = ?
            """, (command,))

        conn.commit()
        conn.close()

    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════

    def add_context(self, role: str, content: str):
        """Add to current context"""
        self.context.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Trim if too long
        if len(self.context) > self.context_max:
            self.context = self.context[-self.context_max:]

    def get_context(self) -> List[Dict]:
        """Get current context"""
        return self.context

    def clear_context(self):
        """Clear context (new conversation)"""
        self.context = []

    # ═══════════════════════════════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM long_term")
        stats["long_term_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM episodic")
        stats["episode_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM tools")
        stats["tool_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM commands")
        stats["command_count"] = cursor.fetchone()[0]

        stats["short_term_count"] = len(self.short_term)
        stats["context_length"] = len(self.context)

        conn.close()
        return stats


# ═══════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile

    print("="*60)
    print("JACK MEMORY - Three-Tier Memory System")
    print("="*60)

    # Use temp directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = Memory(jack_dir=tmpdir)

        # Test short-term
        print("\n[TEST] Short-term memory")
        memory.add_short_term("User asked to organize files", importance=0.7)
        memory.add_short_term("Found 50 files in Downloads", importance=0.5)
        print(f"  Items: {len(memory.short_term)}")

        # Test long-term
        print("\n[TEST] Long-term memory")
        memory.add_long_term("User prefers dark mode", importance=0.9, tags=["preference"])
        results = memory.search_long_term("dark mode")
        print(f"  Found: {len(results)} matches")

        # Test episodic
        print("\n[TEST] Episodic memory")
        memory.add_episode(ActionMemory(
            action_type="shell_run",
            action_data={"command": "ls -la"},
            result_success=True,
            result_data="file1.txt file2.txt",
            state_before={},
            state_after={},
        ))
        episodes = memory.get_similar_episodes("shell_run")
        print(f"  Episodes: {len(episodes)}")

        # Test tools
        print("\n[TEST] Tool memory")
        memory.save_tool(
            name="organize_files",
            description="Organize files by type",
            code="import os\nprint('organizing...')",
            language="python"
        )
        tool = memory.get_tool("organize_files")
        print(f"  Saved tool: {tool.name if tool else 'None'}")

        # Test stats
        print("\n[TEST] Stats")
        stats = memory.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("[OK] Memory system working")
    print("="*60)
