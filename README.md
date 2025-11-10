## Ollama + Langfuse (uv + Python)

Trace local Ollama models with Langfuse using the OpenAI-compatible SDK. This example:
- Chat: `gemma3:1b`
- Embeddings: `granite-embedding:latest`

### Prerequisites
- Ollama running locally (`http://localhost:11434`)
  - `ollama pull gemma3:1b`
  - `ollama pull kimi-k2:1t-cloud` (untuk hakim)
  - `ollama pull granite-embedding:latest`
- Python 3.9+
- `uv` package manager (`pip install uv` or see `https://github.com/astral-sh/uv`)
- Langfuse project with API keys

### Setup
1. Copy environment template and fill values:
   ```bash
   cp .env.example .env
   # set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
   # adjust LANGFUSE_BASE_URL if using US region
   ```
2. Run with uv (no venv needed):
   ```bash
   uv run main.py
   ```

### Streaming Chatbot (Interactive)
- `main.py` starts an interactive chatbot with streaming token output by default.
- Commands:
  - Type your message and press Enter
  - Type `exit`, `quit`, or `:q` to leave
- The session maintains conversation context until you exit.

## 🌐 Web Dashboard v2.0 (NEW - ENHANCED!)

Launch the **stunning new terminal-style dashboard**:

```bash
uv run -m council.cli web
```

Access at: `http://localhost:8000`

### ✨ New v2.0 Features:
- 🎨 **Modern Glassmorphism UI**: Beautiful dark theme dengan gradient accents
- 💻 **Terminal-Style Output**: Live debate streaming dengan syntax highlighting
- ⚡ **Real-time WebSocket**: Watch debates unfold instantly
- 🎭 **Smooth Animations**: Loading states, transitions, progress bars
- 📊 **Live Status Bar**: Connection status, metrics, dan statistics
- 🎯 **5 Interactive Tabs**: Start, Live Stream, History, Analytics, Agents
- 🔔 **Toast Notifications**: Beautiful slide-in notifications
- 📱 **Fully Responsive**: Works perfectly on all devices
- 🌈 **Visual Effects**: Hover animations, glow effects, shimmer
- ⌨️ **Monospace Font**: JetBrains Mono untuk terminal feel

### Screenshots:
- **Start Tab**: Configure agents, questions, and parameters
- **Live Stream Tab**: Real-time terminal output dengan color-coded agents
- **History Tab**: Browse past debates dengan status badges
- **Analytics Tab**: Stats dashboard dengan metrics cards
- **Agents Tab**: Explore available agents dan their properties

## AI Council (Debate Engine)

Run a multi-agent debate with modular personalities and Langfuse tracing. Debates are autosaved to `debates/`.

### Usage
```bash
uv run -m council.cli debate "Haruskah kita mengakselerasi pengembangan AGI?"
```

Options:
- `--title "AGI Acceleration Debate"`
- `--judge kimi-k2:1t-cloud` (default)
- `--min-it 2 --max-it 5`
- `--threshold 0.6`
- `--consensus {majority|supermajority|unanimity}`
- `--eliminate` - Enable agent elimination per iteration
- `--rag` - Enable RAG (Retrieval Augmented Generation)
- `--rag-memory` - Use ChromaDB debate memory for context retrieval
- `--rag-docs` - Use external documents from `docs/` folder

### RAG Example
```bash
# Enable RAG with memory and external documents
uv run -m council.cli debate "AI Ethics and Safety" --rag --rag-memory --rag-docs
```

Enhanced Agents (v0.2.0):
- Strategist Prime, Humanist Voice, Technical Architect
- Systems Thinker, Devil's Advocate, Data Empiricist
- Equity Guardian, Legal Sentinel, Ethics Philosopher
- Risk Strategist, Metrics Oracle, Knowledge Synthesizer, Execution Planner

## Council of Consciousness

High-context governance-style council with moderator + 5 archetypes, episodic memory (SQLite), semantic recall (embeddings), and Markdown logging.

```bash
uv run -m council.cli consciousness --question "Apakah AI dapat menjadi pemimpin moral umat manusia?"
```

Interactive wizard:
- `uv run -m council.cli interactive`
  - Pilih mode `council` atau `debate`
  - Pilih agen, konsensus preset, eliminasi

Artifacts:
- Markdown log per sesi di `debates/`
- Episodic memory ChromaDB di `memory/enhanced_chroma/`
- Memori digunakan sebagai konteks otomatis

## 🧠 Enhanced Memory System (NEW!)

Advanced vector memory with learning capabilities:

### Features:
- **Tags & Categories**: Organize memories with custom tags
- **Memory Decay**: Older memories automatically weighted less
- **Cross-Debate Learning**: Agents learn from past debate patterns
- **Export/Import**: Backup and restore memory database

### Memory Operations:

```python
from council.enhanced_memory import EnhancedCouncilMemory
from council.clients import get_ollama_client

client = get_ollama_client()
memory = EnhancedCouncilMemory()

# Export memory
memory.export_memory(Path("backup/memory_export.json"))

# Import memory
memory.import_memory(Path("backup/memory_export.json"), client)

# Get memory stats
stats = memory.get_memory_stats()
print(f"Total memories: {stats['total_memories']}")
print(f"Tags: {stats['all_tags']}")

# Extract learning insights
insights = memory.extract_learning_insights(
    client=client,
    topic="AI Ethics",
    min_memories=5
)
print(insights)
```

## 🎯 RAG (Retrieval Augmented Generation) (NEW!)

Enhance agent arguments with relevant context from past debates and external documents.

### Features:
- **Memory Retrieval**: Pull relevant context from ChromaDB debate history
- **Document Integration**: Load external knowledge from `.txt`, `.md`, `.json` files
- **Similarity Scoring**: Only retrieve highly relevant context (configurable threshold)
- **Memory Decay**: Recent debates weighted more than older ones
- **Configurable**: Enable/disable per debate via CLI or Web UI

### Web UI Configuration:
1. Navigate to "Start Debate" tab
2. Check "Enable RAG (Retrieval Augmented Generation)"
3. Configure options:
   - Use Debate Memory (ChromaDB)
   - Use External Documents
   - Retrieval Limit (1-10 memories)
   - Minimum Similarity (0.0-1.0)

### CLI Usage:
```bash
# Basic RAG with memory
uv run -m council.cli debate "AGI Safety" --rag

# RAG with memory and external documents
uv run -m council.cli debate "Climate Policy" --rag --rag-memory --rag-docs

# Disable memory, use only external docs
uv run -m council.cli debate "Legal Framework" --rag --no-rag-memory --rag-docs
```

### Adding External Documents:
Create a `docs/` folder and add files:
```bash
mkdir -p docs
echo "AI safety principles: ..." > docs/ai_safety.txt
echo "# Ethics Framework" > docs/ethics.md
```

RAG will automatically load and index these documents for retrieval during debates.

### How RAG Works:
1. **Query Embedding**: Debate question converted to vector
2. **Similarity Search**: ChromaDB finds relevant past arguments
3. **Context Injection**: Top-k relevant memories added to agent prompt
4. **Enhanced Arguments**: Agents cite past debates and documents

### RAG Stats:
```python
from council.rag_system import RAGSystem, RAGConfig
from council.enhanced_memory import EnhancedCouncilMemory

memory = EnhancedCouncilMemory()
rag = RAGSystem(RAGConfig(enabled=True), memory, client)

stats = rag.get_rag_stats()
print(stats)
# Output: {'enabled': True, 'memory_enabled': True, 'external_docs_count': 5, ...}
```

### Notes
- The OpenAI client is configured to `http://localhost:11434/v1`. The `OPENAI_API_KEY` is required by the SDK but not used by Ollama; we set a placeholder value.
- Langfuse tracing is enabled automatically via `from langfuse.openai import OpenAI`.
- ChromaDB provides optimized vector search with HNSW indexing.


