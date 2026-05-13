import time
import uvicorn
from typing import Optional
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# Inizializzazione del server FastMCP
mcp = FastMCP("LlamaCpp_MCP_Server")

# ==========================================
# TOOLS
# ==========================================

@mcp.tool
def add(a: int, b: int) -> int:
    """Somma due numeri interi."""
    return a + b

@mcp.tool
def analyze_text(text: str, max_tokens: int = 100, language: Optional[str] = None) -> dict:
    """Analizza il testo fornito. Restituisce metadati strutturati."""
    return {
        "text_length": len(text),
        "max_tokens_allowed": max_tokens,
        "language_detected": language or "unknown"
    }

@mcp.tool
def slow_tool(x: int) -> int:
    """Esempio di tool bloccante. Eseguito in threadpool per non bloccare l'event loop."""
    time.sleep(2)
    return x * 2

@mcp.tool
async def async_fetch_mock(query: str) -> str:
    """Esempio di tool asincrono per I/O non bloccante."""
    return f"Risultati asincroni mock per: {query}"


# ==========================================
# RESOURCES
# ==========================================

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Restituisce un saluto personalizzato."""
    return f"Hello, {name}!"

@mcp.resource("file://system/info")
def get_system_info() -> str:
    """Info di sistema fornite come risorsa statica."""
    return "Server MCP abilitato con CORS via Uvicorn."


# ==========================================
# PROMPTS
# ==========================================

@mcp.prompt()
def greet_user_prompt(name: str, style: str = "friendly") -> str:
    """Genera un template di prompt da iniettare nel contesto dell'LLM."""
    styles = {
        "friendly": "Scrivi un saluto caldo e amichevole",
        "formal": "Scrivi un saluto formale e professionale"
    }
    return f"{styles.get(style, styles['friendly'])} per un utente di nome {name}."


# ==========================================
# CONFIGURAZIONE CORS E APP ASGI
# ==========================================
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# Configurazione del middleware CORS per consentire l'accesso dal browser
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["mcp-session-id"]
    )
]

# Modifica cruciale: Specifichiamo "streamable-http" per la compatibilità con llama.cpp
# e impostiamo il path su "/sse" in modo che corrisponda all'URL richiesto dal client.
app = mcp.http_app(
    path="/sse",
    transport="streamable-http",
    middleware=middleware
)

if __name__ == "__main__":
    import uvicorn
    # Esecuzione nativa tramite Uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8081)