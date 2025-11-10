## Ollama + Langfuse (uv + Python)

Trace local Ollama models with Langfuse using the OpenAI-compatible SDK. This example:
- Chat: `gemma3:1b`
- Embeddings: `granite-embedding:latest`

### Prerequisites
- Ollama running locally (`http://localhost:11434`)
  - `ollama pull gemma3:1b`
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

## AI Council (Debate Engine)

Run a multi-agent debate with modular personalities and Langfuse tracing. Debates are autosaved to `debates/`.

### Usage
```bash
uv run -m council.cli debate "Haruskah kita mengakselerasi pengembangan AGI?"
```

Options:
- `--title "AGI Acceleration Debate"`
- `--judge gemma3:1b`
- `--min-it 2 --max-it 5`
- `--threshold 0.6`

Default debaters:
- Qwen2.5 Strategist (`qwen2.5:3b`)
- Gemma Dreamer (`gemma3:1b`)
- Qwen3 Engineer (`qwen3:1.7b`)
- Qwen3-VL Observer (`qwen3-vl:2b`)

### Notes
- The OpenAI client is configured to `http://localhost:11434/v1`. The `OPENAI_API_KEY` is required by the SDK but not used by Ollama; we set a placeholder value.
- Langfuse tracing is enabled automatically via `from langfuse.openai import OpenAI`.


