#!/usr/bin/env python3.11
"""
TurboQuant MCP Server — Agent Coding Tools

Server FastMCP che espone tool realistici per agent coding:
- web_fetch: fetch reale di URL
- file_read / file_write: operazioni su filesystem con sandbox
- shell_execute: esecuzione comandi shell
- code_search: ricerca codice con ripgrep
- list_files: listing directory

Tutte le operazioni su filesystem sono sandboxate
all'interno di WORKSPACE_DIR (default: directory corrente).
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# ==========================================
# CONFIGURAZIONE
# ==========================================

# Sandbox: tutti i file operations sono limitati a questa directory
WORKSPACE_DIR = Path(
    os.getenv("WORKSPACE_DIR", ".")
).resolve()

# Timeout per shell execution (secondi)
SHELL_TIMEOUT = int(os.getenv("SHELL_TIMEOUT", "30"))

# Timeout per web fetch (secondi)
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "15"))

# User-Agent per web fetch
FETCH_USER_AGENT = os.getenv(
    "FETCH_USER_AGENT",
    "TurboQuant-MCP-Agent/1.0"
)

# ==========================================
# GITNEXUS — HELPERS & TOOLS
# ==========================================

def _run_gitnexus(
    args: list[str],
    timeout: int = 30,
    cwd: Optional[str] = None,
) -> tuple[str, str]:
    """Run a gitnexus CLI command and return (stdout, stderr)."""
    try:
        result = subprocess.run(
            ["gitnexus"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return (result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return ("", f"[ERROR] gitnexus timeout after {timeout}s")
    except FileNotFoundError:
        return ("", "[ERROR] gitnexus CLI non trovato. Installa: npm i -g gitnexus")
    except Exception as e:
        return ("", f"[ERROR] {type(e).__name__}: {e}")


def _parse_gitnexus_json(stdout: str, stderr: str = "") -> str:
    """Try to parse JSON from stdout, return pretty-printed or raw text."""
    stdout = stdout.strip()
    if not stdout:
        return stderr.strip() if stderr else "[OK] Nessuna risposta da gitnexus."

    # Prova a parsare come JSON
    try:
        obj = json.loads(stdout)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        pass

    # Se c'è stderr (es. warning FTS), includilo
    parts = []
    if stderr:
        stderr_lines = stderr.strip().split("\n")
        # Filtra i warning non critici (FTS retry)
        filtered = [
            l for l in stderr_lines
            if not l.startswith("[gitnexus] FTS index ensure failed")
        ]
        if filtered:
            parts.append("stderr:" + "\n".join(filtered))
    parts.append(stdout)
    return "\n".join(parts)


# ==========================================
# UTILITÀ DI SICUREZZA
# ==========================================

def _resolve_in_sandbox(path: str) -> Path:
    """
    Risolve un path relativo alla sandbox.
    Raise ValueError se il path tenta di uscire dalla sandbox.
    """
    # Gestisci path assoluti: se fuori dalla sandbox, rifiuta
    absolute = Path(path)
    if absolute.is_absolute():
        resolved = absolute.resolve()
    else:
        resolved = (WORKSPACE_DIR / path).resolve()

    # Verifica che il path risieda dentro la sandbox
    try:
        resolved.relative_to(WORKSPACE_DIR)
    except ValueError:
        raise ValueError(
            f"Accesso negato: il percorso '{path}' esce dalla sandbox "
            f"({WORKSPACE_DIR}). Non è possibile navigare con '../' fuori "
            f"dalla directory workspace."
        )

    return resolved


# ==========================================
# INIZIALIZZAZIONE SERVER
# ==========================================

mcp = FastMCP("TurboQuant_Agent_Coding")

# ==========================================
# TOOLS
# ==========================================

@mcp.tool
def web_fetch(
    url: str,
    as_markdown: bool = False,
    max_length: int = 10000,
) -> str:
    """
    Fetch reale da un URL. Restituisce il contenuto della pagina.
    Imposta as_markdown=True per provare a estrarre testo strutturato.
    Limita l'output a max_length caratteri.
    """
    try:
        with httpx.Client(
            timeout=httpx.Timeout(FETCH_TIMEOUT),
            follow_redirects=True,
        ) as client:
            headers = {"User-Agent": FETCH_USER_AGENT}
            response = client.get(url, headers=headers)
            response.raise_for_status()

            # Cerca encoding nei headers o nel contenuto
            encoding = response.headers.get("content-type", "")
            if "utf-8" in encoding or "text" in encoding:
                content = response.text
            else:
                # Non è testo, restituisci info binarie
                return (
                    f"[BINARY CONTENT] URL {url} restituisce "
                    f"content-type: {response.headers.get('content-type')}. "
                    f"Size: {len(response.content)} bytes."
                )

            if as_markdown:
                # Estrae testo rimuovendo script/style e tag HTML base
                content = re.sub(
                    r"<(script|style|noscript)[^>]*>.*?</\1>",
                    "", content, flags=re.DOTALL | re.IGNORECASE
                )
                content = re.sub(r"<br\s*/?>", "\n", content, flags=re.IGNORECASE)
                content = re.sub(r"<p[^>]*>", "\n\n", content, flags=re.IGNORECASE)
                content = re.sub(r"<h[1-6][^>]*>", "\n## ", content, flags=re.IGNORECASE)
                content = re.sub(r"</h[1-6]>", " ##\n", content, flags=re.IGNORECASE)
                content = re.sub(r"<li>", "\n- ", content, flags=re.IGNORECASE)
                content = re.sub(r"<[^>]+>", "", content)
                content = re.sub(r"\n{3,}", "\n\n", content)

            content = content.strip()
            if len(content) > max_length:
                content = content[:max_length] + "\n...[truncated]"

            return content

    except httpx.HTTPError as e:
        return f"[ERROR] Failed to fetch {url}: {e}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@mcp.tool
def file_read(path: str, encoding: str = "utf-8") -> str:
    """
    Legge il contenuto di un file dalla filesystem.
    Il percorso è sandboxato dentro WORKSPACE_DIR.
    Non è possibile uscire con '../'.
    """
    try:
        resolved = _resolve_in_sandbox(path)
    except ValueError as e:
        return f"[ERROR] {e}"

    if not resolved.exists():
        return f"[ERROR] File non trovato: {path}"

    if resolved.is_dir():
        return f"[ERROR] '{path}' è una directory, non un file."

    try:
        content = resolved.read_text(encoding=encoding)
        # Limita a 5000 righe per evitare OOM
        lines = content.split("\n")
        if len(lines) > 5000:
            return (
                f"[TRUNCATED] File con {len(lines)} righe. "
                f"Mostrando prime 5000.\n" + "\n".join(lines[:5000])
            )
        return content
    except UnicodeDecodeError:
        return f"[ERROR] File '{path}' non sembra un file di testo (encoding {encoding})."
    except PermissionError:
        return f"[ERROR] Permesso negato: {path}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@mcp.tool
def file_write(
    path: str,
    content: str,
    mode: str = "overwrite",
    encoding: str = "utf-8",
) -> str:
    """
    Scrive contenuto in un file.
    - mode='overwrite': sovrascrive il file esistente
    - mode='append': aggiunge in coda
    Crea le directory intermedie se necessario.
    Sandboxato dentro WORKSPACE_DIR.
    """
    try:
        resolved = _resolve_in_sandbox(path)
    except ValueError as e:
        return f"[ERROR] {e}"

    try:
        # Crea directory intermedie se necessario
        resolved.parent.mkdir(parents=True, exist_ok=True)

        if mode == "overwrite":
            resolved.write_text(content, encoding=encoding)
        elif mode == "append":
            resolved.write_text(
                resolved.read_text(encoding=encoding) + content,
                encoding=encoding,
            )
        else:
            return f"[ERROR] mode deve essere 'overwrite' o 'append', ricevuto: {mode}"

        return f"[OK] Scritto {len(content)} caratteri in {path} ({mode})."

    except PermissionError:
        return f"[ERROR] Permesso negato: {path}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@mcp.tool
def shell_execute(
    command: str,
    timeout: Optional[int] = None,
) -> str:
    """
    Esegue un comando nella shell (bash).
    Restituisce stdout, stderr e codice di uscita.
    Timeout di default: 30s (configurabile con SHELL_TIMEOUT env o parametro).
    """
    timeout = timeout or SHELL_TIMEOUT

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WORKSPACE_DIR),
        )

        output = f"Exit code: {result.returncode}\n"
        if result.stdout:
            output += f"stdout:\n{result.stdout}\n"
        if result.stderr:
            output += f"stderr:\n{result.stderr}"

        if len(output) > 10000:
            output = output[:10000] + "\n...[truncated]"

        return output

    except subprocess.TimeoutExpired:
        return f"[ERROR] Comando timeout dopo {timeout}s: {command}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@mcp.tool
def code_search(
    pattern: str,
    path: Optional[str] = None,
    file_glob: Optional[str] = None,
    max_results: int = 50,
    context_lines: int = 2,
) -> str:
    """
    Cerca pattern nel codice usando ripgrep (rg).
    - pattern: regex o stringa letterale da cercare
    - path: directory base (default: WORKSPACE_DIR, sandboxato)
    - file_glob: filtro estensione es. '*.py' o '*.{js,ts}'
    - max_results: numero massimo di match
    - context_lines: linee di contesto prima/dopo ogni match

    Usa rg se disponibile, altrimenti grep.
    """
    try:
        resolved_base = _resolve_in_sandbox(path or ".")
    except ValueError as e:
        return f"[ERROR] {e}"

    args = ["-n", "-C", str(context_lines), "--color=never", pattern]

    # Usa ripgrep se disponibile
    rg_available = subprocess.run(
        ["which", "rg"], capture_output=True
    ).returncode == 0

    if rg_available:
        args.insert(0, "rg")
        if file_glob:
            args.extend(["--glob", file_glob])
        args.extend(["--max-count", str(max_results)])
        args.append(str(resolved_base))
    else:
        # Fallback su grep
        args.insert(0, "grep")
        args = ["-n", "-B", str(context_lines), "-A", str(context_lines),
                "--color=never"] + args
        if file_glob:
            # grep non supporta glob direttamente, filtra dopo
            pass
        args.append(str(resolved_base))

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=15,
        )

        output = result.stdout.strip()
        if not output:
            return f"[OK] Nessun match trovato per '{pattern}'."
        if len(output) > 10000:
            output = output[:10000] + "\n...[truncated]"
        return output

    except subprocess.TimeoutExpired:
        return f"[ERROR] code_search timeout dopo 15s."
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


@mcp.tool
def list_files(
    path: Optional[str] = None,
    recursive: bool = False,
    max_items: int = 200,
) -> str:
    """
    Lista file e directory.
    - path: directory da listare (default: WORKSPACE_DIR, sandboxato)
    - recursive: se True, lista ricorsivamente
    - max_items: numero massimo di elementi restituiti

    Ogni riga: [DIR/FIL] relative/path
    """
    try:
        resolved = _resolve_in_sandbox(path or ".")
    except ValueError as e:
        return f"[ERROR] {e}"

    if not resolved.exists():
        return f"[ERROR] Percorso non trovato: {path}"

    if resolved.is_file():
        return f"[FIL] {resolved.relative_to(WORKSPACE_DIR)}"

    items = []
    try:
        if recursive:
            for root, dirs, files in os.walk(resolved):
                rel = Path(root).relative_to(WORKSPACE_DIR)
                for d in sorted(dirs):
                    items.append(f"[DIR] {rel / d}")
                for f in sorted(files):
                    items.append(f"[FIL] {rel / f}")
        else:
            for entry in sorted(resolved.iterdir()):
                prefix = "[DIR]" if entry.is_dir() else "[FIL]"
                rel = entry.relative_to(WORKSPACE_DIR)
                items.append(f"{prefix} {rel}")
    except PermissionError:
        return f"[ERROR] Permesso negato: {path}"

    if len(items) > max_items:
        return (
            f"[TRUNCATED] {len(items)} items totali, mostrando primi {max_items}.\n"
            + "\n".join(items[:max_items])
        )
    return "\n".join(items) if items else f"[OK] Directory vuota: {path}"


@mcp.tool
def file_info(path: str) -> dict:
    """
    Restituisce metadata su un file: dimensione, modifica, tipo.
    Sandboxato dentro WORKSPACE_DIR.
    """
    try:
        resolved = _resolve_in_sandbox(path)
    except ValueError as e:
        return {"error": str(e)}

    if not resolved.exists():
        return {"error": f"File non trovato: {path}"}

    stat = resolved.stat()
    rel = resolved.relative_to(WORKSPACE_DIR)

    return {
        "path": str(rel),
        "type": "directory" if resolved.is_dir() else "file",
        "size_bytes": stat.st_size,
        "modified": stat.st_mtime,
        "modified_human": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
        ),
        "is_executable": resolved.stat().st_mode & 0o111 != 0,
    }


# ==========================================
# GITNEXUS TOOLS — Knowledge Graph & Code Intelligence
# ==========================================

@mcp.tool
def gitnexus_list_repos() -> str:
    """
    Lista tutti i repository indicizzati da GitNexus.
    Mostra per ciascun repo: nome, path, data indicizzazione, commit,
    statistiche (file, simboli, edges, clusters, flows).
    """
    stdout, stderr = _run_gitnexus(["list"])
    return _parse_gitnexus_json(stdout, stderr)


@mcp.tool
def gitnexus_analyze(
    path: str,
    name: Optional[str] = None,
    force: bool = False,
    embeddings: bool = False,
    skip_agents_md: bool = True,
    no_stats: bool = False,
    skip_git: bool = False,
) -> str:
    """
    Indizza un repository con GitNexus (full analysis).
    - path: percorso della repo da indicizzare
    - name: alias personalizzato nel registry (opzionale)
    - force: forza re-indicizzazione anche se già presente
    - embeddings: genera embeddings per ricerca semantica
    - skip_agents_md: non aggiornare AGENTS.md/CLAUDE.md
    - no_stats: ometti conteggi volatili nei file
    - skip_git: indicizza anche cartelle senza .git

    Timeout: 300s (analisi completa può richiedere tempo).
    """
    args = ["analyze"]
    if name:
        args.extend(["--name", name])
    if force:
        args.append("--force")
    if embeddings:
        args.append("--embeddings")
    if skip_agents_md:
        args.append("--skip-agents-md")
    if no_stats:
        args.append("--no-stats")
    if skip_git:
        args.append("--skip-git")
    args.append(path)

    stdout, stderr = _run_gitnexus(args, timeout=300)
    return _parse_gitnexus_json(stdout, stderr)


@mcp.tool
def gitnexus_query(
    query: str,
    repo: Optional[str] = None,
    limit: int = 5,
    context: Optional[str] = None,
    goal: Optional[str] = None,
    include_content: bool = False,
) -> str:
    """
    Cerca flussi di esecuzione nel knowledge graph di GitNexus.
    Restituisce processi (call chain) ordinati per rilevanza.

    - query: ricerca testuale (natural language o keywords)
    - repo: repo target (omit se c'è una sola repo indicizzata)
    - limit: numero massimo di processi (default: 5)
    - context: contesto del task per migliorare il ranking
    - goal: cosa vuoi trovare (es. "existing auth validation logic")
    - include_content: includi codice sorgente dei simboli (default: False)

    Timeout: 30s.
    """
    args = ["query", query]
    if repo:
        args.extend(["-r", repo])
    args.extend(["-l", str(limit)])
    if context:
        args.extend(["-c", context])
    if goal:
        args.extend(["-g", goal])
    if include_content:
        args.append("--content")

    stdout, stderr = _run_gitnexus(args, timeout=30)
    return _parse_gitnexus_json(stdout, stderr)


@mcp.tool
def gitnexus_context(
    name: str,
    repo: Optional[str] = None,
    file_path: Optional[str] = None,
    kind: Optional[str] = None,
    include_content: bool = False,
) -> str:
    """
    Vista 360° di un simbolo del codice: callers, callees, processi
    partecipation, file location.

    - name: nome del simbolo (function, class, method, ecc.)
    - repo: repo target
    - file_path: file per disambiguare nomi comuni
    - kind: tipo del simbolo ("Function", "Class", "Method", ecc.)
    - include_content: includi codice sorgente completo (default: False)

    Timeout: 30s.
    """
    args = ["context", name]
    if repo:
        args.extend(["-r", repo])
    if file_path:
        args.extend(["--file", file_path])
    if kind:
        args.extend(["--kind", kind])
    if include_content:
        args.append("--content")

    stdout, stderr = _run_gitnexus(args, timeout=30)
    return _parse_gitnexus_json(stdout, stderr)


@mcp.tool
def gitnexus_impact(
    target: str,
    repo: Optional[str] = None,
    direction: str = "upstream",
    depth: int = 3,
    include_tests: bool = False,
) -> str:
    """
    Analisi del blast radius: cosa si rompe se modifichi un simbolo.

    - target: simbolo da analizzare (funzione, classe, metodo)
    - repo: repo target
    - direction: "upstream" (dipendenti) o "downstream" (dipendenze)
    - depth: profondità massima della traversal (default: 3)
    - include_tests: include anche i file di test (default: False)

    Timeout: 30s.
    """
    args = ["impact", target]
    if repo:
        args.extend(["-r", repo])
    args.extend(["-d", direction])
    args.extend(["--depth", str(depth)])
    if include_tests:
        args.append("--include-tests")

    stdout, stderr = _run_gitnexus(args, timeout=30)
    return _parse_gitnexus_json(stdout, stderr)


@mcp.tool
def gitnexus_cypher(
    query: str,
    repo: Optional[str] = None,
) -> str:
    """
    Esegue una query Cypher raw sul knowledge graph.
    Utile per esplorazioni strutturali complesse.

    - query: stringa Cypher (es. "MATCH (c:Class)-[:HAS_METHOD]->(m) RETURN c.name, m.name")
    - repo: repo target (opzionale)

    Timeout: 30s.
    """
    args = ["cypher", query]
    if repo:
        args.extend(["-r", repo])

    stdout, stderr = _run_gitnexus(args, timeout=30)
    return _parse_gitnexus_json(stdout, stderr)


@mcp.tool
def gitnexus_detect_changes(
    repo: Optional[str] = None,
    scope: str = "unstaged",
    base_ref: Optional[str] = None,
) -> str:
    """
    Mappa le modifiche git diff sui simboli indicizzati e
    identifica i processi affected.

    - repo: repo target
    - scope: "unstaged", "staged", "all", o "compare"
    - base_ref: branch/commit per "compare" scope (es. "main")

    Timeout: 30s.
    """
    args = ["detect_changes", "-s", scope]
    if repo:
        args.extend(["-r", repo])
    if base_ref:
        args.extend(["-b", base_ref])

    stdout, stderr = _run_gitnexus(args, timeout=30)
    return _parse_gitnexus_json(stdout, stderr)


@mcp.tool
def gitnexus_rename(
    old_name: str,
    new_name: str,
    repo: Optional[str] = None,
    file_path: Optional[str] = None,
    dry_run: bool = True,
) -> str:
    """
    Rinomina un simbolo su tutti i file usando il knowledge graph
    + text search. Sicuro: fa graph rename con doppia verifica.

    - old_name: nome attuale del simbolo
    - new_name: nuovo nome
    - repo: repo target
    - file_path: file per disambiguare nomi comuni
    - dry_run: preview senza modificare file (default: True)

    Timeout: 30s.
    """
    args = ["rename", old_name, new_name]
    if repo:
        args.extend(["-r", repo])
    if file_path:
        args.extend(["--file", file_path])
    if dry_run:
        args.append("--dry-run")

    stdout, stderr = _run_gitnexus(args, timeout=30)
    return _parse_gitnexus_json(stdout, stderr)


@mcp.tool
def gitnexus_group_list(name: Optional[str] = None) -> str:
    """
    Lista tutti i gruppi di repository configurati, oppure mostra
    i dettagli di un gruppo specifico.

    - name: nome del gruppo (omit per listare tutti)

    Timeout: 15s.
    """
    args = ["group", "list"]
    if name:
        args.append(name)

    stdout, stderr = _run_gitnexus(args, timeout=15)
    return _parse_gitnexus_json(stdout, stderr)


@mcp.tool
def gitnexus_group_sync(
    name: str,
    skip_embeddings: bool = False,
    exact_only: bool = False,
) -> str:
    """
    Ricrea il Contract Registry (contracts.json) per un gruppo:
    estrae HTTP contracts, applica manifest links, exact-match cross-links.

    - name: nome del gruppo da sincronizzare
    - skip_embeddings: solo BM25 (no vector embeddings)
    - exact_only: solo match esatti (cascade)

    Timeout: 60s.
    """
    args = ["group", "sync", name]
    if skip_embeddings:
        args.append("--skip-embeddings")
    if exact_only:
        args.append("--exact-only")

    stdout, stderr = _run_gitnexus(args, timeout=60)
    return _parse_gitnexus_json(stdout, stderr)


@mcp.tool
def gitnexus_group_query(
    query: str,
    group: str,
    limit: int = 5,
    context: Optional[str] = None,
    goal: Optional[str] = None,
) -> str:
    """
    Cerca flussi di esecuzione in tutti i repo di un gruppo.
    Combina i risultati via RRF (Reciprocal Rank Fusion).

    - query: ricerca testuale
    - group: nome del gruppo di repo
    - limit: max processi (default: 5)
    - context: contesto per migliorare ranking
    - goal: cosa vuoi trovare

    Timeout: 30s.
    """
    args = ["group", "query", group, query, "-l", str(limit)]
    if context:
        args.extend(["-c", context])
    if goal:
        args.extend(["-g", goal])

    stdout, stderr = _run_gitnexus(args, timeout=30)
    return _parse_gitnexus_json(stdout, stderr)


@mcp.tool
def gitnexus_status(repo: Optional[str] = None) -> str:
    """
    Mostra lo stato dell'indice GitNexus per la repo corrente
    o per una repo specifica.

    - repo: repo target (omit per la cwd)

    Timeout: 15s.
    """
    args = ["status"]
    if repo:
        args.append(repo)

    stdout, stderr = _run_gitnexus(args, timeout=15)
    return _parse_gitnexus_json(stdout, stderr)


@mcp.tool
def gitnexus_clean(repo: Optional[str] = None) -> str:
    """
    Cancella l'indice GitNexus per la repo corrente o specificata.
    Utile prima di fare un analyze fresh.

    - repo: repo target (omit per la cwd)

    Timeout: 30s.
    """
    args = ["clean"]
    if repo:
        args.append(repo)

    stdout, stderr = _run_gitnexus(args, timeout=30)
    return _parse_gitnexus_json(stdout, stderr)


# ==========================================
# RESOURCES & RESOURCE TEMPLATES
# ==========================================

@mcp.resource("config://workspace")
def get_workspace_info() -> str:
    """Info sulla sandbox workspace corrente."""
    return (
        f"WORKSPACE_DIR={WORKSPACE_DIR}\n"
        f"SHELL_TIMEOUT={SHELL_TIMEOUT}s\n"
        f"FETCH_TIMEOUT={FETCH_TIMEOUT}s\n"
        f"FETCH_USER_AGENT={FETCH_USER_AGENT}\n"
    )


@mcp.resource(
    "file://workspace/{path*}",
    description="Legge un file dal workspace. path supporta sottodirectory (es. server.py, docs/README.md).",
)
def read_workspace_file(path: str) -> str:
    """
    Legge il contenuto di un file del workspace via resources.
    path supporta path con sottodirectory (es. src/main.py).
    """
    try:
        resolved = _resolve_in_sandbox(path)
    except ValueError as e:
        return f"[ERROR] {e}"

    if not resolved.exists():
        return f"[ERROR] File non trovato: {path}"
    if resolved.is_dir():
        return f"[ERROR] '{path}' è una directory, non un file."

    try:
        content = resolved.read_text("utf-8")
        lines = content.split("\n")
        if len(lines) > 5000:
            return (
                f"[TRUNCATED] File con {len(lines)} righe. "
                f"Mostrando prime 5000.\n" + "\n".join(lines[:5000])
            )
        return content
    except UnicodeDecodeError:
        return f"[ERROR] File non leggibile come testo: {path}"
    except PermissionError:
        return f"[ERROR] Permesso negato: {path}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


# ==========================================
# PROMPTS
# ==========================================

@mcp.prompt(
    name="analyze_code",
    description="Analizza il codice in un file specifico del workspace.",
)
def analyze_code_prompt(filepath: str, focus: str = "struttura generale") -> str:
    """
    Crea un prompt per analizzare codice in un file del workspace.
    - filepath: percorso relativo del file (es. server.py)
    - focus: cosa analizzare (es. 'struttura generale', 'sicurezza', 'performance')
    """
    snippet = ""
    try:
        resolved = _resolve_in_sandbox(filepath)
        if resolved.exists() and not resolved.is_dir():
            content = resolved.read_text("utf-8")
            lines = content.split("\n")
            snippet = "\n".join(lines[:200])
            if len(lines) > 200:
                snippet += f"\n# ... ({len(lines) - 200} righe troncate)"
    except (ValueError, Exception):
        snippet = "[Impossibile leggere il file]"

    return (
        f"Analizza il seguente codice nel file `{filepath}`.\n\n"
        f"```\n{snippet}\n```\n\n"
        f"Focalizzati su: {focus}.\n"
        f"Spiegami cosa fa il codice, i pattern usati, e eventuali problemi."
    )


@mcp.prompt(
    name="review_changes",
    description="Analizza le modifiche git non committate nel workspace.",
)
def review_changes_prompt(scope: str = "unstaged", focus: str = "qualità del codice") -> str:
    """
    Prompt per revisionare modifiche git.
    - scope: 'unstaged', 'staged', 'all'
    - focus: aspetto da analizzare (es. 'qualità del codice', 'sicurezza', 'bug')
    """
    diff = ""
    try:
        cmd = ["git", "diff"]
        if scope == "staged":
            cmd = ["git", "diff", "--cached"]
        elif scope == "all":
            cmd = ["git", "diff", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(WORKSPACE_DIR))
        diff = result.stdout[:5000] if result.stdout else "[Nessuna modifica trovata]"
    except Exception:
        diff = "[Impossibile ottenere il diff]"

    return (
        f"Revisiona le seguenti modifiche (scope: {scope}).\n\n"
        f"```diff\n{diff}\n```\n\n"
        f"Focalizzati su: {focus}.\n"
        f"Evidenzia potenziali problemi, suggerisci miglioramenti, "
        f"e valuta la qualità complessiva."
    )


@mcp.prompt(
    name="search_codebase",
    description="Cerca pattern nel codebase e analizza i risultati.",
)
def search_codebase_prompt(pattern: str, file_glob: str = "", context_lines: int = 3) -> str:
    """
    Cerca pattern nel codebase e prepara un'analisi.
    - pattern: regex o stringa da cercare
    - file_glob: filtro estensione (es. '*.py', '*.{ts,js}')
    - context_lines: linee di contesto per ogni match
    """
    args = ["rg", "-n", "-C", str(context_lines), "--color=never", pattern, str(WORKSPACE_DIR)]
    if file_glob:
        args.extend(["--glob", file_glob])

    results = ""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if not output:
            results = f"[Nessun match trovato per '{pattern}']"
        else:
            results = output[:8000]
            if len(output) > 8000:
                results += "\n...[troncato]"
    except FileNotFoundError:
        # Fallback grep
        try:
            grep_args = ["grep", "-rn", "-B", str(context_lines), "-A", str(context_lines), pattern, str(WORKSPACE_DIR)]
            result = subprocess.run(grep_args, capture_output=True, text=True, timeout=30)
            results = result.stdout.strip()[:8000] or f"[Nessun match trovato per '{pattern}']"
        except Exception as e:
            results = f"[Errore: {e}]"
    except Exception as e:
        results = f"[Errore: {e}]"

    return (
        f"Ho cercato '{pattern}' nel codebase e ho trovato:\n\n"
        f"```\n{results}\n```\n\n"
        f"Analizza i risultati: descrivi cosa fa ogni match, "
        f"come si collega al resto del sistema, e se ci sono pattern interessanti."
    )


@mcp.prompt(
    name="explain_command",
    description="Spiega un comando shell o un concetto di sistema.",
)
def explain_command_prompt(command: str) -> str:
    """
    Spiega un comando shell in dettaglio.
    - command: il comando da spiegare
    """
    return (
        f"Spiega in dettaglio il seguente comando shell:\n\n"
        f"```bash\n{command}\n```\n\n"
        f"Descrivi:\n"
        f"1. Cosa fa ogni parte del comando\n"
        f"2. Quali flag/opzioni vengono usati\n"
        f"3. Eventuali rischi o side-effect\n"
        f"4. Alternative o varianti comuni"
    )


@mcp.prompt(
    name="git_commit_message",
    description="Genera un messaggio di commit basato sulle modifiche staged.",
)
def git_commit_message_prompt(style: str = "convenzionale", language: str = "italiano") -> str:
    """
    Genera un messaggio di commit dalle modifiche staged.
    - style: 'convenzionale' (Conventional Commits), 'breve', 'dettagliato'
    - language: 'italiano' o 'english'
    """
    diff = ""
    log = ""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--stat"],
            capture_output=True, text=True, timeout=10, cwd=str(WORKSPACE_DIR)
        )
        diff = result.stdout.strip() or "[Nessuna modifica staged]"

        result = subprocess.run(
            ["git", "log", "--oneline", "-10"],
            capture_output=True, text=True, timeout=10, cwd=str(WORKSPACE_DIR)
        )
        log = result.stdout.strip()
    except Exception:
        pass

    return (
        f"Genera un messaggio di commit per le seguenti modifiche staged.\n\n"
        f"### File modificati\n```\n{diff}\n```\n\n"
        f"### Commit recenti\n```\n{log}\n```\n\n"
        f"Stile: {style}\nLingua: {language}\n\n"
        f"Restituisci SOLO il messaggio di commit, senza spiegazioni aggiuntive."
    )


# ==========================================
# CONFIGURAZIONE CORS E APP ASGI
# ==========================================

cors_middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["mcp-session-id"]
    )
]

app = mcp.http_app(
    path="/sse",
    transport="streamable-http",
    middleware=cors_middleware
)

if __name__ == "__main__":
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("SERVER_PORT", "8081"))
    print(f"🚀 TurboQuant MCP Agent Coding Server")
    print(f"   Sandbox: {WORKSPACE_DIR}")
    print(f"   Listening on http://{host}:{port}/sse")
    import uvicorn
    uvicorn.run("server:app", host=host, port=port)
