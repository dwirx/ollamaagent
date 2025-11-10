from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from council.types import DebateConfig, Personality
from council.personalities import default_personalities
from council.storage import autosave_json
from council.engine import run_debate
from council.clients import get_ollama_client
from council.rag_system import RAGSystem, RAGConfig
from council.enhanced_memory import EnhancedCouncilMemory


# Global state
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


# API Models
class RAGConfigRequest(BaseModel):
    use_memory: bool = True
    use_external_docs: bool = False
    retrieval_limit: int = 3
    min_similarity: float = 0.6


class DebateStartRequest(BaseModel):
    question: str
    title: Optional[str] = None
    judge_model: str = "kimi-k2:1t-cloud"
    min_iterations: int = 2
    max_iterations: int = 5
    consensus_threshold: float = 0.66
    elimination: bool = False
    selected_agents: Optional[List[str]] = None
    mode: str = "debate"  # debate, council, collaboration, oxford, socratic, etc.
    rag_enabled: bool = False
    rag_config: Optional[RAGConfigRequest] = None


class DebateHistoryItem(BaseModel):
    id: str
    title: Optional[str]
    question: str
    timestamp: str
    iterations: int
    consensus_reached: bool
    file_path: str


# Lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Council Debate Server starting...")
    yield
    # Shutdown
    print("👋 Council Debate Server shutting down...")


# FastAPI app
app = FastAPI(
    title="Council Debate Dashboard",
    description="Real-time multi-agent debate system with analytics",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (will create later)
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve main dashboard page"""
    # Try v2 first, fallback to v1
    html_v2 = Path(__file__).parent / "templates" / "dashboard_v2.html"
    html_v1 = Path(__file__).parent / "templates" / "dashboard.html"

    if html_v2.exists():
        return FileResponse(html_v2)
    elif html_v1.exists():
        return FileResponse(html_v1)

    return HTMLResponse(
        """
        <html>
            <head><title>Council Debate Dashboard</title></head>
            <body style="font-family: sans-serif; padding: 40px; background: #0a0e27; color: white;">
                <h1>🏛️ Council Debate Dashboard</h1>
                <p>Dashboard files are being set up...</p>
                <p><a href="/docs" style="color: #00d4ff;">API Documentation</a></p>
            </body>
        </html>
        """
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/personalities")
async def get_personalities():
    """Get all available personalities"""
    personalities = default_personalities()
    return {
        "personalities": [
            {
                "name": p.name,
                "model": p.model,
                "traits": p.traits,
                "perspective": p.perspective,
                "reasoning_depth": p.reasoning_depth,
                "truth_seeking": p.truth_seeking,
                "persistence": p.persistence,
            }
            for p in personalities
        ]
    }


@app.get("/api/debates/history")
async def get_debate_history(limit: int = 50) -> List[DebateHistoryItem]:
    """Get history of past debates"""
    debates_dir = Path("debates")
    if not debates_dir.exists():
        return []

    debate_files = sorted(
        debates_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    history = []
    for file_path in debate_files:
        try:
            with open(file_path) as f:
                data = json.load(f)
                config = data.get("config", {})
                iterations = data.get("iterations", [])

                last_iter = iterations[-1] if iterations else {}

                history.append(
                    DebateHistoryItem(
                        id=file_path.stem,
                        title=config.get("title"),
                        question=config.get("question", ""),
                        timestamp=file_path.stem.split("_")[0] if "_" in file_path.stem else "",
                        iterations=len(iterations),
                        consensus_reached=last_iter.get("consensus_reached", False),
                        file_path=str(file_path),
                    )
                )
        except Exception as e:
            print(f"Error loading debate {file_path}: {e}")
            continue

    return history


@app.get("/api/debates/{debate_id}")
async def get_debate_detail(debate_id: str):
    """Get detailed debate information"""
    debates_dir = Path("debates")
    matching_files = list(debates_dir.glob(f"{debate_id}*.json"))

    if not matching_files:
        raise HTTPException(status_code=404, detail="Debate not found")

    file_path = matching_files[0]
    try:
        with open(file_path) as f:
            data = json.load(f)
            return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading debate: {str(e)}")


@app.websocket("/ws/debate")
async def websocket_debate(websocket: WebSocket):
    """WebSocket endpoint for real-time debate streaming"""
    await manager.connect(websocket)

    try:
        # Wait for debate start command
        data = await websocket.receive_json()

        if data.get("type") == "start_debate":
            config_data = data.get("config", {})

            # Broadcast debate start
            await manager.broadcast(
                {
                    "type": "debate_started",
                    "question": config_data.get("question"),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            # Create debate config
            config = DebateConfig(
                title=config_data.get("title"),
                question=config_data["question"],
                judge_model=config_data.get("judge_model", "kimi-k2:1t-cloud"),
                min_iterations=config_data.get("min_iterations", 2),
                max_iterations=config_data.get("max_iterations", 5),
                consensus_threshold=config_data.get("consensus_threshold", 0.66),
            )

            # Get personalities
            all_personalities = default_personalities()
            selected_names = config_data.get("selected_agents")

            if selected_names:
                personalities = [p for p in all_personalities if p.name in selected_names]
            else:
                personalities = all_personalities[:6]  # Default: first 6

            # Define streaming callback for WebSocket
            async def stream_callback(event_type: str, data: Any):
                await manager.broadcast(
                    {
                        "type": event_type,
                        "data": data,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            # Run debate in background task
            # Note: This is simplified - in production, use background tasks properly
            try:
                # TODO: Implement async version of run_debate or use threading
                await manager.broadcast(
                    {
                        "type": "debate_info",
                        "message": "Debate engine starting (sync mode - streaming limited)",
                    }
                )

                # For now, send a message that debate is running
                # Full integration requires async debate engine
                await manager.broadcast(
                    {
                        "type": "debate_complete",
                        "message": "Debate completed (check CLI for full results)",
                        "note": "Full real-time streaming requires async debate engine - coming soon!",
                    }
                )

            except Exception as e:
                await manager.broadcast(
                    {
                        "type": "error",
                        "message": f"Debate error: {str(e)}",
                    }
                )

        # Keep connection alive
        while True:
            try:
                message = await websocket.receive_text()
                # Handle other message types if needed
                if message == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.post("/api/debates/start")
async def start_debate_api(request: DebateStartRequest):
    """Start a new debate (synchronous API)"""
    try:
        # Create debate config
        config = DebateConfig(
            title=request.title,
            question=request.question,
            judge_model=request.judge_model,
            min_iterations=request.min_iterations,
            max_iterations=request.max_iterations,
            consensus_threshold=request.consensus_threshold,
        )

        # Get personalities
        all_personalities = default_personalities()
        if request.selected_agents:
            personalities = [p for p in all_personalities if p.name in request.selected_agents]
        else:
            personalities = all_personalities[:6]

        # Initialize RAG system if enabled
        rag_system = None
        if request.rag_enabled and request.rag_config:
            client = get_ollama_client()
            memory = EnhancedCouncilMemory() if request.rag_config.use_memory else None

            rag_config = RAGConfig(
                enabled=True,
                use_memory=request.rag_config.use_memory,
                use_external_docs=request.rag_config.use_external_docs,
                external_docs_path=Path("docs") if request.rag_config.use_external_docs else None,
                retrieval_limit=request.rag_config.retrieval_limit,
                min_similarity=request.rag_config.min_similarity,
            )

            rag_system = RAGSystem(rag_config, memory, client)

            # Load external docs if enabled
            if request.rag_config.use_external_docs:
                docs_path = Path("docs")
                if docs_path.exists():
                    rag_system.load_external_documents(docs_path)

        # Handle different debate modes
        # TODO: Implement mode routing for council, collaboration, oxford, etc.
        if request.mode != "debate":
            # For now, we only support standard debate mode
            # Other modes will be implemented in future updates
            print(f"Warning: Mode '{request.mode}' not yet implemented, using standard debate")

        # Run debate (blocking - in production, use background task)
        state = run_debate(
            config=config,
            personalities=personalities,
            save_callback=autosave_json,
            elimination=request.elimination,
            rag_system=rag_system,
        )

        # Return summary
        return {
            "status": "completed",
            "iterations": len(state.iterations),
            "consensus_reached": state.iterations[-1].consensus_reached if state.iterations else False,
            "judge_decision": state.judge_decision,
            "timestamp": datetime.utcnow().isoformat(),
            "mode": request.mode,
            "rag_enabled": request.rag_enabled,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debate error: {str(e)}")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the web server"""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
