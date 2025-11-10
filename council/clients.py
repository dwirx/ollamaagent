import os
from dotenv import load_dotenv
from langfuse.openai import OpenAI


def _ensure_langfuse_env() -> None:
    required = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing Langfuse environment variables: {names}. Set them in your .env.")


def get_ollama_client() -> OpenAI:
    load_dotenv(override=False)
    _ensure_langfuse_env()
    # Ensure Langfuse keys exist (validation done in main entrypoints typically)
    base_url = os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1")
    return OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "ollama"))


