import os
from typing import List

from dotenv import load_dotenv
from langfuse.openai import OpenAI
import sys


def require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value


def get_openai_client_for_ollama() -> OpenAI:
    """
    Configure the OpenAI-compatible client to talk to Ollama at localhost:11434
    via the Langfuse drop-in, enabling tracing automatically.
    """
    base_url = os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1")
    # api_key required by SDK but not used by Ollama; keep a placeholder.
    return OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "ollama"))


def run_chat(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def get_embeddings(client: OpenAI, model: str, inputs: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model=model,
        input=inputs,
    )
    # OpenAI embeddings API returns a list with one embedding per input
    return [d.embedding for d in response.data]


def chat_stream_repl(client: OpenAI, model: str, system_prompt: str) -> None:
    """
    Simple interactive REPL with streaming token output and conversation memory.
    Type 'exit' or Ctrl+C to quit.
    """
    print(f"Starting streaming chat with model '{model}'. Type 'exit' to quit.")
    messages = [{"role": "system", "content": system_prompt}]
    while True:
        try:
            user = input("\nYou: ").strip()
            if user.lower() in {"exit", "quit", ":q"}:
                print("Bye!")
                return
            if not user:
                continue
            messages.append({"role": "user", "content": user})

            # Stream assistant response
            print("Assistant: ", end="", flush=True)
            full_content_parts: List[str] = []
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            for event in stream:
                # Each chunk may contain delta content
                try:
                    delta = event.choices[0].delta.content or ""
                except Exception:
                    # Some SDK versions may use a slightly different schema; fallback gracefully
                    delta = ""
                if delta:
                    full_content_parts.append(delta)
                    sys.stdout.write(delta)
                    sys.stdout.flush()
            assistant_content = "".join(full_content_parts).strip()
            print("")  # newline
            messages.append({"role": "assistant", "content": assistant_content})
        except KeyboardInterrupt:
            print("\nInterrupted. Bye!")
            return
        except EOFError:
            print("\nEOF. Bye!")
            return


def main() -> None:
    # Load .env first to make local development easy
    load_dotenv(override=False)

    # Required for Langfuse tracing
    require_env("LANGFUSE_PUBLIC_KEY")
    require_env("LANGFUSE_SECRET_KEY")
    # Optional: set "LANGFUSE_BASE_URL" if using US region:
    # os.environ["LANGFUSE_BASE_URL"] = "https://us.cloud.langfuse.com"

    client = get_openai_client_for_ollama()

    # By default, start interactive streaming chatbot
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "gemma3:1b")
    chat_stream_repl(
        client=client,
        model=chat_model,
        system_prompt="Anda adalah asisten yang ringkas, membantu, dan sopan. Jawab dalam Bahasa Indonesia secara singkat.",
    )


if __name__ == "__main__":
    main()


