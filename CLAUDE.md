# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent debate and deliberation system using local Ollama models with Langfuse tracing. The project has three main modes:

1. **Interactive Chatbot** (`main.py`) - Streaming chat with conversation memory
2. **Council Debate** (`council.cli debate`) - Multi-agent competitive debate with voting and consensus
3. **Council of Consciousness** (`council.cli consciousness`) - High-context governance council with episodic memory

## Environment Setup

### Prerequisites
- Ollama running locally at `http://localhost:11434`
- Python 3.9+ with `uv` package manager
- Langfuse project credentials for tracing

### Configuration
1. Copy `.env.example` to `.env`
2. Set required variables:
   - `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` (required for tracing)
   - `LANGFUSE_BASE_URL` (EU: `https://cloud.langfuse.com`, US: `https://us.cloud.langfuse.com`)
3. Default models used:
   - Chat: `gemma3:1b`
   - Embeddings: `granite-embedding:latest`
   - Various models for different agent personalities

### Install Required Ollama Models
```bash
ollama pull gemma3:1b
ollama pull gemma3:latest
ollama pull qwen2.5:3b
ollama pull qwen3:1.7b
ollama pull granite-embedding:latest
```

## Running the Project

### Interactive Chatbot (Streaming)
```bash
uv run main.py
```
Commands in REPL: `exit`, `quit`, or `:q` to quit

### Council Debate
```bash
# Basic debate
uv run -m council.cli debate "Your question here"

# With options
uv run -m council.cli debate "Question" \
  --title "Debate Title" \
  --judge gemma3:1b \
  --min-it 2 \
  --max-it 5 \
  --consensus majority \
  --eliminate
```

Consensus options:
- `majority` (>50%)
- `supermajority` (>66%, default)
- `unanimity` (100%)

### Council of Consciousness
```bash
# Direct question
uv run -m council.cli consciousness --question "Your question"

# Interactive wizard
uv run -m council.cli interactive
```

## Architecture

### Core Debate Engine (`council/engine.py`)

The debate system uses an iterative argument-vote-consensus loop:

1. **Argument Generation** (`_prompt_for_argument`): Each personality generates arguments based on their traits, perspective, and prior arguments
2. **Voting** (`_prompt_for_vote`): Agents rank all arguments from best to worst
3. **Consensus Check** (`_consensus_from_votes`): Determines if threshold met by counting first-place votes
4. **Elimination** (optional): Removes worst-performing agent based on aggregate ranking
5. **Judge Decision** (`_prompt_for_judge`): Final synthesis after iterations complete

All streaming uses `_stream_completion` with callbacks for real-time token display.

### Council of Consciousness (`council/consciousness.py`)

Multi-phase governance deliberation with persistent memory:

1. **Memory Retrieval**: Fetches recent + semantically similar episodes from SQLite
2. **Moderator Opening**: Frames the question and structure
3. **Argument Round**: Each role (philosopher, humanist, critic, spiritualist, technocrat) contributes
4. **Critique Phase**: Dedicated critic analyzes all contributions
5. **Reflection Phase**: Each speaker reflects on critique and others' points
6. **Moderator Closing**: Synthesis and summary
7. **Elimination** (optional): Evaluator suggests weakest contributor

All phases are logged to Markdown (`debates/` folder) and SQLite (`memory/council_memory.db`).

### Memory System (`council/memory.py`)

**CouncilMemory** class provides episodic memory with semantic search:
- **Storage**: SQLite with columns: timestamp, question, agent, role, phase, content, embedding
- **Embedding**: Uses `granite-embedding:latest` via `embed_text()`
- **Retrieval**:
  - `fetch_recent()` - chronological recent episodes
  - `fetch_similar()` - cosine similarity search on embeddings
- **Summarization**: `summarize_memory()` condenses context for agents

### Personality System

Two frameworks:

1. **Debate Personalities** (`council/personalities.py`):
   - Base agents: Strategist, Dreamer, Engineer, Observer
   - Specialized: Bias Auditor, Compliance Legal, Ethics Reviewer, Risk Assessor, Performance Analyst, Retrieval Agent, Planning Agent
   - Each has: model, traits, perspective, persistence, reasoning_depth, truth_seeking

2. **Council Roles** (`council/roles.py`):
   - Fixed archetypes: Moderator, Rationalist, Humanist, Critic, Spiritualist, Technocrat
   - Each has: archetype, perspective, signature (speaking style), color

### Client Configuration (`council/clients.py`)

`get_ollama_client()` returns Langfuse-wrapped OpenAI client:
- Base URL: `http://localhost:11434/v1` (Ollama's OpenAI-compatible endpoint)
- API key: placeholder "ollama" (required by SDK, ignored by Ollama)
- Automatic tracing via `langfuse.openai.OpenAI`

### Storage (`council/storage.py`)

`autosave_json()` serializes DebateState to `debates/` folder with timestamp-based filenames.

## Development Commands

### Run Tests
No test framework currently configured.

### Code Style
Project uses Python 3.9+ with type hints. Key conventions:
- Pydantic models for data structures (`types.py`)
- Rich library for terminal output
- Streaming completions for real-time user feedback

### Project Structure
```
.
├── main.py                    # Interactive chatbot entry point
├── council/
│   ├── cli.py                 # Typer CLI with debate/consciousness/interactive commands
│   ├── engine.py              # Core debate loop logic
│   ├── consciousness.py       # Council of Consciousness multi-phase engine
│   ├── memory.py              # SQLite + embeddings memory system
│   ├── personalities.py       # Debate agent definitions
│   ├── roles.py               # Council archetype definitions
│   ├── clients.py             # Ollama client factory
│   ├── types.py               # Pydantic models
│   ├── storage.py             # JSON serialization
│   └── interactive.py         # Terminal wizard UI
├── debates/                   # Auto-generated Markdown logs
├── memory/                    # SQLite database for episodic memory
└── pyproject.toml             # uv dependencies
```

## Key Implementation Details

### Adding New Personalities
Edit `council/personalities.py` `default_personalities()`:
```python
Personality(
    name="Your Agent Name",
    model="ollama-model-name",
    traits="Key characteristics",
    perspective="Viewpoint and approach",
    persistence=0.5,        # 0-1: resistance to changing position
    reasoning_depth=2,      # Complexity of reasoning requested
    truth_seeking=0.8,      # 0-1: prioritization of truth vs other goals
)
```

### Adding New Council Roles
Edit `council/roles.py` `council_of_consciousness_roles()`:
```python
CouncilRole(
    key="unique_key",
    title="Display Name",
    model="ollama-model",
    archetype="Philosophical framework",
    perspective="Analytical lens",
    signature="Speaking style",
    color="rich_color_name",
    reasoning_depth=2,
    truth_seeking=0.8,
)
```

### Modifying Consensus Logic
Edit `council/engine.py` `_consensus_from_votes()`. Currently uses first-place vote counting. Alternative approaches could use Borda count, approval voting, or ranked-choice algorithms.

### Memory Retrieval Customization
In `council/consciousness.py` `run_council_of_consciousness()`:
- Adjust `limit` parameters for `fetch_similar()` and `fetch_recent()`
- Modify `summarize_memory()` prompt in `council/memory.py` to change summarization style

### Color Customization
Colors are assigned deterministically in `council/engine.py` `_color_for()` using hash-based palette selection. Council roles have explicit colors in `council/roles.py`.

## Langfuse Tracing

All LLM calls are automatically traced because:
1. Client created via `langfuse.openai.OpenAI` instead of standard `openai.OpenAI`
2. Environment variables `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` are set
3. Each call appears as a trace in Langfuse dashboard with input/output/latency/cost

To disable tracing: unset Langfuse env vars or switch to standard OpenAI client.

## Notes

- **Language**: All system prompts and UI are in Indonesian (Bahasa Indonesia)
- **Streaming**: Used extensively for real-time user feedback; implement via `_stream_completion()` pattern
- **Debate Outputs**: Auto-saved to `debates/` after each iteration as checkpoint
- **Memory Persistence**: SQLite database persists across sessions; context automatically retrieved
- **Elimination Mode**: Optional competitive element where worst-performing agents are removed each iteration
