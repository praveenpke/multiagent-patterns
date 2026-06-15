# Reusable LangGraph Multi-Agent Patterns Library

Modular library of production-ready LangGraph orchestration patterns.

## Overview

A curated collection of well-documented, reusable LangGraph graph patterns extracted from real production agents. Each pattern includes a graph definition, Mermaid diagram, state schema (Pydantic), guardrails, and observability hooks.

## Patterns Included

### Orchestration Patterns
| Pattern | Description |
|---------|-------------|
| **Supervisor** | Central router delegates to specialist subagents based on task type |
| **Hierarchical Delegation** | Multi-level agent trees with scoped authority |
| **Reflection** | Agent critiques and rewrites its own output iteratively |
| **Plan-and-Execute** | Planner generates step list; executor runs each step |
| **ReAct** | Reason-Act-Observe loop for tool-using agents |

### Control Flow Patterns
| Pattern | Description |
|---------|-------------|
| **HITL Handoff** | Pause execution and route to human for approval |
| **Conditional Routing** | Dynamic edge selection based on agent output |
| **Parallel Fan-out** | Run multiple subagents concurrently, merge results |

### Memory Patterns
| Pattern | Description |
|---------|-------------|
| **Episodic Memory** | Store and retrieve past interactions by similarity |
| **Shared State** | Typed state schema passed across all nodes |
| **Checkpointing** | Persist graph state for resumable long-running tasks |

## Repository Structure

```
langgraph-multiagent-patterns/
├── patterns/
│   ├── supervisor/
│   ├── hierarchical/
│   ├── reflection/
│   ├── hitl_handoff/
│   ├── plan_and_execute/
│   └── react/
├── memory/
│   ├── episodic/
│   └── checkpointing/
├── schemas/          # Pydantic state schemas
├── diagrams/         # Mermaid graph diagrams
└── examples/         # End-to-end usage examples
```

## Tech Stack

- **Runtime:** LangGraph, LangChain
- **Validation:** Pydantic v2
- **Observability:** LangSmith
- **Diagrams:** Mermaid

## Getting Started

```bash
git clone https://github.com/praveenpke/langgraph-multiagent-patterns.git
cd langgraph-multiagent-patterns
pip install -r requirements.txt

# Run a pattern example
python examples/supervisor_example.py
```

## Usage

Import any pattern directly into your agent:

```python
from patterns.supervisor import build_supervisor_graph
from schemas.email import EmailAgentState

graph = build_supervisor_graph(agents=[...], state_schema=EmailAgentState)
```

## Status

Work in progress — patterns being extracted and documented from production agents.
