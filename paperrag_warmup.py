#!/usr/bin/env python3
"""Warm-up helper for the Paper RAG MCP server.

The Paper RAG server runs on AWS App Runner, which scales to zero when idle.
The first request after idle triggers a cold start (loading the embedding model
+ vector DB) that can take a few minutes -- long enough to blow past the
Anthropic client's request timeout mid-agent-turn, surfacing as a stalled or
errored first tool call. warm_up_paperrag() wakes the instance with a direct,
tiny search before the agent turn so the connector's first tool call hits a warm
server. Uses only the standard library (no third-party HTTP dependency).
"""
import json
import time
import urllib.error
import urllib.request

# Paper RAG MCP Server URL (AWS App Runner; scales to zero when idle)
PAPERRAG_MCP_SERVER_URL = "https://m76rjhx9i3.us-east-1.awsapprunner.com/mcp"

# The warm-up gets a generous timeout of its own so it can absorb a cold start
# without the caller's normal (shorter) request timeout applying.
WARMUP_TIMEOUT_S = 300.0


def warm_up_paperrag(api_key: str, verbose: bool = True) -> bool:
    """Wake the (scale-to-zero) Paper RAG server before the real agent turn.

    Sends a direct MCP handshake plus a tiny ``search_papers`` call, which forces
    all lazy-loading (embedding model, vector DB) so the subsequent agent turn
    runs against a warm instance within the normal request timeout. Best-effort:
    on failure it warns and lets the caller proceed anyway.

    Returns True if the warm-up search completed, False otherwise.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    def _post(payload: dict, ignore_errors: bool = False):
        req = urllib.request.Request(
            PAPERRAG_MCP_SERVER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=WARMUP_TIMEOUT_S) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError):
            if ignore_errors:
                return None
            raise

    if verbose:
        print("Warming up Paper RAG server (cold starts can take a minute)...")
    start = time.monotonic()
    try:
        # Handshake is best-effort; the stateless server accepts tool calls
        # without a session, so only the search itself must succeed.
        _post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "paperrag-warmup", "version": "1.0"},
                },
            },
            ignore_errors=True,
        )
        _post({"jsonrpc": "2.0", "method": "notifications/initialized"}, ignore_errors=True)
        # This search forces the embedding model + vector DB to load.
        _post(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "search_papers", "arguments": {"query": "warmup", "n_results": 1}},
            }
        )
        elapsed = time.monotonic() - start
        if verbose:
            print(f"Paper RAG server ready ({elapsed:.1f}s).\n")
        return True
    except (urllib.error.URLError, TimeoutError) as exc:
        elapsed = time.monotonic() - start
        if verbose:
            print(
                f"Warning: warm-up did not complete after {elapsed:.1f}s ({exc}). "
                "Proceeding anyway -- the first search may be slow.\n"
            )
        return False
