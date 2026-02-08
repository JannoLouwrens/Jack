# Jack - Universal Intelligent Agent

**An AI agent that truly understands computers - built on the same principles as JackTheWalker.**

> Jack learns digital "physics" first, then adds language. Not an LLM wrapper - a real learning system.

---

## Philosophy

Like JackTheWalker learns physics (F=ma, torque, energy) before walking, Jack learns **digital physics** (file operations, processes, APIs, networks) before controlling computers.

```
JACKTHEWALKER                          JACK
───────────────                        ────
Phase 0: Learn physics (SymPy)    →    Phase 0: Learn digital physics (sandbox)
Phase 1: Learn to walk (RL)       →    Phase 1: Learn to achieve goals (RL)
Phase 2: Learn from demos         →    Phase 2: Learn from user feedback
Phase 3: Add language             →    Phase 3: Add language (train LLM into brain)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              JACK                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│   │   BRAIN     │  │  VERIFIER   │  │  EXECUTOR   │  │   MEMORY    │   │
│   │             │  │             │  │             │  │             │   │
│   │ World Model │  │ Safety      │  │ 5 Actions:  │  │ Short-term  │   │
│   │ (learned)   │  │ Rules       │  │ • shell     │  │ Long-term   │   │
│   │             │  │ (symbolic)  │  │ • file_read │  │ Episodic    │   │
│   │ Planner     │  │             │  │ • file_write│  │             │   │
│   │ (goals)     │  │ Blocks      │  │ • http      │  │ Tools       │   │
│   │             │  │ dangerous   │  │ • get_state │  │ (saved code)│   │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│          │                │                │                │          │
│          └────────────────┴────────────────┴────────────────┘          │
│                                    │                                    │
│                         ┌──────────┴──────────┐                        │
│                         │    OS ADAPTERS      │                        │
│                         │ Windows/Linux/Mac   │                        │
│                         └──────────┬──────────┘                        │
│                                    │                                    │
│                         ┌──────────┴──────────┐                        │
│                         │      NETWORK        │                        │
│                         │  (future: IoT/mesh) │                        │
│                         └─────────────────────┘                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5 Actions = Unlimited Capability

Jack has only 5 primitive actions, but can do ANYTHING:

| Action | What It Does |
|--------|--------------|
| `shell_run(cmd)` | Run any command |
| `file_read(path)` | Read any file |
| `file_write(path, content)` | Write any file (including code) |
| `http_request(url, ...)` | Call any API |
| `get_state()` | Observe system |

**+ Code generation = Jack can write Python/Bash to do anything**

---

## Training Pipeline

```
Phase 0: Digital Physics
├── Sandbox (Docker) generates state-action-outcome data
├── Jack tries random commands, observes results
├── World model learns to predict outcomes
└── Like Phase0_Physics.py but for computers

Phase 1: Goal Achievement
├── RL in sandboxed environments
├── Learn to achieve: organize files, find things, automate tasks
└── Like Phase1_Locomotion.py but for computer tasks

Phase 2: User Feedback
├── Deploy to real system
├── Learn from success/failure
├── Continual improvement
└── Like Phase2_Imitation.py but from real interaction

Phase 3: Language Integration
├── Fine-tune code LLM into Jack's brain
├── Natural language interface
├── No external API dependency
└── Runs fully local
```

---

## Quick Start

```bash
# Install
pip install -e ".[full]"    # Full install with ML and CLI
pip install -e ".[cli]"     # Just CLI (lighter)
pip install -e .            # Minimal (core only)

# Run a task
jack run "list all python files in current directory"

# Interactive mode
jack interactive
# or shorthand
jack i

# Direct shell command (verified for safety)
jack shell "dir /a"

# Training (Phase 0: Digital Physics)
jack train --samples 1000

# Check status
jack status
```

### Python API

```python
from jack import Brain, Executor, Verifier, Memory

# Initialize components
memory = Memory()
executor = Executor()
verifier = Verifier()
brain = Brain(memory=memory, executor=executor, verifier=verifier)

# Execute a task
result = brain.execute("find all log files over 10MB")
print(result)

# Direct action (with safety check)
is_safe, reason = verifier.check_shell("rm -rf /tmp/cache")
if is_safe:
    result = executor.shell_run("rm -rf /tmp/cache")
```

---

## Project Structure

```
jack/
├── core/
│   ├── jack_brain.py         # 85K param transformer neural network
│   ├── executor.py           # 5 primitive actions
│   ├── verifier.py           # Safety rules
│   └── memory.py             # Short/long-term memory
│
├── foundation/               # Production-grade modules (SOTA research)
│   ├── loop.py              # Main agent orchestration loop
│   ├── reason.py            # 8 advanced reasoning patterns
│   ├── perceive.py          # LLM-first perception
│   ├── retrieve.py          # Agentic RAG (multi-source)
│   ├── verify.py            # Constitutional AI verification
│   ├── memory.py            # A-MEM Zettelkasten memory
│   ├── robust.py            # Production infrastructure
│   ├── llm.py               # OpenAI-compatible LLM integration
│   ├── brain_reasoner.py    # Brain-LLM hybrid reasoner
│   └── types.py             # Rust-inspired Result/Option types
│
├── training/
│   ├── phase0_digital.py     # Learn digital physics (sandbox)
│   └── train_transformer.py  # Train JackBrain
│
├── server/                   # FastAPI server
│   ├── app.py               # Main application
│   ├── auth.py              # JWT + API key authentication
│   └── sandbox.py           # Safe code execution
│
├── adapters/
│   ├── windows.py            # Windows-specific
│   ├── linux.py              # Linux-specific
│   └── macos.py              # macOS-specific
│
└── network/                  # Future: IoT mesh
    ├── node.py
    └── protocol.py
```

---

## Research Papers Implemented

Jack implements 10+ state-of-the-art patterns from AI research:

| Pattern | Paper | Description |
|---------|-------|-------------|
| Chain of Thought | [Wei et al. 2022](https://arxiv.org/abs/2201.11903) | Step-by-step reasoning |
| Self-Consistency | [Wang et al. 2022](https://arxiv.org/abs/2203.11171) | Multiple reasoning paths + voting |
| Tree of Thoughts | [Yao et al. 2023](https://arxiv.org/abs/2305.10601) | Tree search with BFS/DFS/MCTS |
| Reflexion | [Shinn et al. 2023](https://arxiv.org/abs/2303.11366) | Self-reflection loop |
| Least-to-Most | [Zhou et al. 2022](https://arxiv.org/abs/2205.10625) | Problem decomposition |
| ReAct | [Yao et al. 2022](https://arxiv.org/abs/2210.03629) | Think-Act-Observe |
| LATS | [Zhou et al. 2023](https://arxiv.org/abs/2310.04406) | Language Agent Tree Search |
| Constitutional AI | [Anthropic 2022](https://arxiv.org/abs/2212.08073) | Principle-based safety |
| A-MEM | Zettelkasten | Agentic memory with linking |
| OpenClaw | [Docs](https://docs.openclaw.ai) | Lifecycle hooks, context management |

---

## Reasoner Options

| Reasoner | Use Case |
|----------|----------|
| `BrainReasoner` | Fast actions using trained neural network |
| `SyncLLMReasoner` | Complex reasoning with LLM |
| `HybridReasoner` | Best of both - brain for speed, LLM for complexity |

```python
from jack.foundation.brain_reasoner import create_hybrid_reasoner
from jack.foundation.loop import Loop

reasoner = create_hybrid_reasoner()
loop = Loop(reasoner=reasoner)
```

---

## Safety (Verifier)

Every action passes through the Verifier before execution:

```python
# Dangerous patterns blocked
"rm -rf /"           # Would destroy system
"DROP TABLE"         # Would delete database
"> /dev/sda"         # Would corrupt disk
"curl | bash"        # Piping to shell

# Secrets protected
passwords.txt        # Blocked from reading
.env files           # Blocked from exposing
API keys in URLs     # Blocked
```

---

## Network (Future: IoT)

Jack is designed to run on multiple devices:

```
jack@laptop ←──────→ jack@server
     ↕                    ↕
jack@raspberry-pi ←──→ jack@arduino
     ↕
jack@robot (JackTheWalker!)
```

Same brain. Different bodies. Coordinated action.

---

## Author

**Janno Louwrens**
- BSc Computing (UNISA 2024)
- Honours AI (in progress)
- Creator of JackTheWalker

---

## License

MIT
