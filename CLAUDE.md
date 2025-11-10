# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an advanced multi-agent debate and deliberation system using local Ollama models with Langfuse tracing, ChromaDB vector memory, and focus scoring mechanisms. The project has three main modes:

1. **Interactive Chatbot** (`main.py`) - Streaming chat with conversation memory
2. **Council Debate** (`council.cli debate`) - Multi-agent competitive debate with voting, consensus, and real-time focus scoring
3. **Council of Consciousness** (`council.cli consciousness`) - High-context governance council with semantic memory retrieval via ChromaDB

### Key Enhancements (v2)
- **Focus Scoring**: Automatic evaluation of argument relevance to keep debates on-topic
- **ChromaDB Memory**: Advanced vector database for semantic memory retrieval
- **Improved Prompts**: Stricter system prompts enforcing focus and depth
- **Enhanced Display**: Rich terminal UI with panels, progress indicators, and color-coded agents
- **Better Personalities**: More diverse and professional agent archetypes with higher reasoning depth
- **Improved Moderator**: Grand Moderator role with stronger facilitation and focus enforcement

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
ollama pull kimi-k2:1t-cloud  # Model hakim (judge)
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
  --judge kimi-k2:1t-cloud \  # Default judge model (powerful)
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
5. **Judge Decision** (`_prompt_for_judge`): Final synthesis after iterations complete using **kimi-k2:1t-cloud** model (powerful judge with enhanced reasoning)

All streaming uses `_stream_completion` with callbacks for real-time token display.

**Judge Model**: Default is `kimi-k2:1t-cloud`, a high-quality model specifically chosen for:
- Superior reasoning and synthesis capabilities
- Better understanding of complex multi-agent debates
- More nuanced and balanced final decisions
- Higher context window for processing full debate transcripts

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

### Memory System (`council/chroma_memory.py`)

**ChromaCouncilMemory** class provides advanced vector memory with ChromaDB:
- **Storage**: ChromaDB persistent client with HNSW indexing and cosine similarity
- **Embedding**: Uses `granite-embedding:latest` via `embed_text()`
- **Retrieval**:
  - `fetch_recent()` - chronological recent episodes with optional filters
  - `fetch_similar()` - semantic similarity search with configurable threshold
  - `search_by_metadata()` - filter by agent, role, phase, question
- **Summarization**: `summarize_memory()` condenses both recent and semantically similar memories
- **Benefits over SQLite**:
  - Optimized vector indexing (HNSW algorithm)
  - Better performance for large memory collections
  - Native similarity search without manual cosine calculation
  - Persistent storage with automatic optimization
  - Metadata filtering combined with vector search

**Legacy SQLite memory** (`council/memory.py`) is still available but deprecated in favor of ChromaDB.

### Focus Scoring System (`council/focus_scorer.py`)

Automatic evaluation system to keep debates on-topic:

- **`score_argument_focus()`**: Scores individual arguments (0.0-1.0) for relevance to the question
  - Uses LLM evaluation with strict rubric
  - Returns `FocusScore` with score, reasoning, and boolean focus flag
  - Configurable threshold (default: 0.7)

- **`batch_score_arguments()`**: Scores multiple arguments efficiently

- **`generate_focus_report()`**: Creates markdown report with scores and status for all participants

- **`get_focus_warnings()`**: Returns warnings for off-topic arguments

**Integration**: Focus scoring runs automatically after each debate iteration, providing real-time feedback on argument quality.

### Personality System

Two frameworks:

1. **Debate Personalities** (`council/personalities.py`) - Enhanced v2:
   - **Base agents** (high-quality, diverse perspectives):
     - `Strategist Prime`: Deep systems thinking, game theory, cost-benefit analysis (depth=3, truth=0.85)
     - `Humanist Voice`: Empathy-driven, social justice focus, human welfare prioritization (depth=2, truth=0.8)
     - `Technical Architect`: Pragmatic implementer, engineering rigor, feasibility-focused (depth=3, truth=0.9)
     - `Systems Thinker`: Holistic view, feedback loops, cascade effects analysis (depth=2, truth=0.85)
     - `Devil's Advocate`: Contrarian challenger, assumption breaker, extreme critical thinking (depth=3, truth=0.95)
     - `Data Empiricist`: Evidence-only, rejects speculation, methodological rigor (depth=2, truth=0.95)

   - **Specialized agents**:
     - `Equity Guardian`: Bias detection, DEI champion, intersectional analysis
     - `Legal Sentinel`: Compliance expert, liability mapper, risk-averse
     - `Ethics Philosopher`: Kantian/utilitarian balance, moral reasoning depth
     - `Risk Strategist`: Paranoid productive, scenario planning, probabilistic thinking
     - `Metrics Oracle`: OKR-driven, KPI designer, measurement obsessed
     - `Knowledge Synthesizer`: Cross-domain connector, pattern recognizer
     - `Execution Planner`: Implementation-first, resource optimizer, critical path

   - Each has: model, traits, perspective, persistence (0-1), reasoning_depth (1-3), truth_seeking (0-1)

2. **Council Roles** (`council/roles.py`) - Enhanced v2:
   - **Grand Moderator**: Master facilitator with Socratic method, enforces focus, decisive synthesis (depth=3, truth=0.95)
   - **Chief Logic Officer**: Aristotelian syllogism, Bayesian reasoning, fallacy detection (depth=3, truth=0.95)
   - **Voice of Humanity**: Rawlsian justice, lived experience focus, welfare prioritization (depth=2, truth=0.85)
   - **Radical Skeptic**: Critical theory, power structure analysis, assumption challenger (depth=3, truth=0.9)
   - **Wisdom Keeper**: Stoic/Buddhist integration, existential meaning, long-term civilizational view (depth=2, truth=0.85)
   - **Future Architect**: Techno-optimist, scalability focus, exponential thinking (depth=3, truth=0.85)

   - Each has: archetype, perspective, signature (speaking style), color, reasoning_depth, truth_seeking

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
│   ├── engine.py              # Core debate loop logic + focus scoring integration
│   ├── consciousness.py       # Council of Consciousness multi-phase engine
│   ├── chroma_memory.py       # ChromaDB vector memory system (NEW)
│   ├── memory.py              # SQLite memory system (DEPRECATED)
│   ├── focus_scorer.py        # Focus evaluation system (NEW)
│   ├── personalities.py       # Enhanced debate agent definitions (v2)
│   ├── roles.py               # Enhanced council archetypes (v2)
│   ├── clients.py             # Ollama client factory
│   ├── types.py               # Pydantic models
│   ├── storage.py             # JSON serialization
│   └── interactive.py         # Terminal wizard UI
├── debates/                   # Auto-generated Markdown logs
├── memory/
│   ├── chroma_db/             # ChromaDB persistent storage (NEW)
│   └── council_memory.db      # Legacy SQLite database
└── pyproject.toml             # uv dependencies (includes chromadb)
```

## Key Implementation Details

### Prompt Engineering for Focus

The system uses strict prompts to enforce focus and prevent topic drift:

**Debate Prompts** (`council/engine.py`):
- 7-point strict rule system in every agent prompt
- Explicit instructions: "FOKUS MUTLAK", "jangan melebar", max 3-4 poin
- Truth-seeking and reasoning depth parameters
- Evidence-based argumentation requirements
- Specific engagement with prior arguments

**Judge Prompts**:
- Professional judge persona with structured evaluation framework
- 5-step decision format (summary, strongest argument, weaknesses, decision, recommendations)
- Emphasis on objective truth over popularity

**Council Prompts** (`council/consciousness.py`):
- Enhanced system prompts with ARKETIPE, PERSPEKTIF, GAYA sections
- 7 strict rules including focus enforcement
- Limited context window (last 5 messages) to maintain clarity
- Phase-specific instructions

### Focus Scoring Workflow

1. **After Arguments**: Each debate iteration, all arguments are scored
2. **LLM Evaluation**: Gemma3 model scores 0.0-1.0 with reasoning
3. **Threshold Check**: Default 0.65-0.7 for "focused" classification
4. **Real-time Feedback**: Warnings displayed immediately for off-topic arguments
5. **Visual Indicators**: ✓ green for focused, ⚠ yellow for off-topic

### ChromaDB Integration

**Migration from SQLite**:
- Replace `CouncilMemory()` with `ChromaCouncilMemory()`
- Embeddings now required for all `record_episode()` calls
- `fetch_similar()` returns scored tuples `(similarity, MemoryRecord)`
- Optional `min_similarity` threshold for filtering

**Best Practices**:
- Always generate embeddings before recording
- Use `fetch_similar()` for semantic queries
- Use `fetch_recent()` for chronological history
- Combine both in `summarize_memory()` for comprehensive context

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

### Enhanced Display Features

**Debate UI Improvements** (`council/engine.py`):
- **Header Panel**: Shows question, participant count, consensus threshold, elimination status
- **Agent Headers**: Each argument displays agent name, position, reasoning depth, truth-seeking score
- **Focus Feedback**: Real-time warnings for off-topic arguments with scores
- **Voting Table**: Enhanced with Top 3 rankings and first-choice highlights
- **Consensus Panel**: Visual status (✅ green for success, ⚠️ yellow for failure)
- **Judge Panel**: Bordered final decision with emoji header

**Color System**:
- Deterministic hash-based color assignment in `_color_for()`
- 10-color palette for debate personalities
- Fixed colors for council roles (cyan, magenta, yellow, green, bright_blue, bright_white)
- Consistent coloring across arguments, votes, and tables

**Progress Indicators**:
- Dim text for metadata and progress info
- Bold for speaker names and key information
- Color-coded streaming output per agent
- Iteration counters and phase announcements

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
- **Memory Persistence**: ChromaDB persists across sessions; semantic retrieval automatic
- **Elimination Mode**: Optional competitive element where worst-performing agents are removed each iteration

## Version 2 Changes Summary

### Major Enhancements

1. **Focus Enforcement System**
   - New `focus_scorer.py` module for automatic relevance evaluation
   - Real-time scoring (0.0-1.0) with reasoning for each argument
   - Visual warnings for off-topic contributions
   - Integrated into debate loop after each iteration

2. **ChromaDB Vector Memory**
   - Replaced SQLite with ChromaDB for optimized vector search
   - HNSW indexing for fast similarity queries
   - Configurable similarity thresholds
   - Metadata filtering combined with semantic search
   - Better performance on large memory collections

3. **Improved Agent Quality**
   - **Debate personalities**: 6 base + 7 specialized (was 4 + 7)
   - Higher reasoning depth (2-3 vs 1-2)
   - Higher truth-seeking scores (0.8-0.95 vs 0.7-0.85)
   - More diverse professional archetypes
   - Better-defined perspectives and traits

4. **Enhanced Council Roles**
   - Grand Moderator with stronger facilitation (was generic Moderator)
   - Chief Logic Officer with formal logic emphasis (was Rationalist)
   - More specific philosophical frameworks for each role
   - Higher reasoning depth across all roles

5. **Stricter Prompts**
   - 7-point rule system for focus enforcement
   - Explicit "FOKUS MUTLAK" and "jangan melebar" instructions
   - Structured format requirements (max 3-4 points)
   - Evidence and citation requirements
   - Limited context windows to prevent drift

6. **Better Display & UX**
   - Debate header panel with config summary
   - Agent metadata display (depth, truth-seeking)
   - Enhanced voting table with Top 3 and first choice columns
   - Consensus status panels with emoji indicators
   - Color-coded streaming with progress indicators
   - Focus warnings displayed in real-time

7. **Upgraded Judge Model**
   - Changed from `gemma3:1b` to `kimi-k2:1t-cloud`
   - Superior reasoning and synthesis capabilities
   - Better handling of complex multi-agent debates
   - Larger context window for full transcript processing

### File Changes

**New Files**:
- `council/chroma_memory.py` - ChromaDB memory implementation
- `council/focus_scorer.py` - Focus evaluation system

**Major Updates**:
- `council/engine.py` - Focus scoring integration, enhanced display
- `council/personalities.py` - Complete redesign of all personalities
- `council/roles.py` - Enhanced council archetypes
- `council/consciousness.py` - ChromaDB integration, improved prompts
- `council/types.py` - Updated default judge model to `kimi-k2:1t-cloud`
- `council/cli.py` - Updated judge default parameter
- `pyproject.toml` - Added chromadb dependency

**Deprecated**:
- `council/memory.py` - Use `chroma_memory.py` instead

### Breaking Changes

- `CouncilMemory` → `ChromaCouncilMemory` (API mostly compatible)
- Embeddings now required for all `record_episode()` calls
- `fetch_similar()` returns `List[tuple[float, MemoryRecord]]` instead of `List[MemoryRecord]`
- Some personality names changed (e.g., "Qwen2.5 Strategist" → "Strategist Prime")
- **Default judge model changed**: `gemma3:1b` → `kimi-k2:1t-cloud` (must pull this model via `ollama pull kimi-k2:1t-cloud`)

### Performance Improvements

- Faster semantic search with ChromaDB's native HNSW indexing
- Reduced off-topic arguments via stricter prompts
- Better debate quality through higher truth-seeking agents
- More efficient memory summarization with scored records
