#!/usr/bin/env python3
"""
Scholar Gateway client helper.

Wiley's v2 custom connector authenticates with a static API key sent in an
X-API-Key header on every request (no OAuth token exchange). The key is read
from the environment (never hardcoded, never logged).

The Anthropic MCP connector can only send an `Authorization: Bearer` header,
which the v2 server rejects, so Scholar Gateway is called client-side: this
module speaks MCP (JSON-RPC over Streamable HTTP) directly with httpx, exposes
the server's tools as regular Anthropic tool definitions, and executes the
tool calls the model makes.

Run this module directly to smoke-test connectivity:

    python scholar_auth.py
"""

import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

# Non-secret endpoint URL (safe to keep in source). Streamable HTTP only; no SSE.
SCHOLAR_GATEWAY_MCP_URL = "https://custom-connector-v2.scholargateway.ai/mcp"

# Environment variable name holding the secret API key (value lives in .env).
API_KEY_ENV = "SCHOLAR_GATEWAY_API_KEY"

MCP_PROTOCOL_VERSION = "2025-06-18"


def has_api_key() -> bool:
    """Return True if the Scholar Gateway API key env var is set."""
    return bool(os.environ.get(API_KEY_ENV))


def get_scholar_gateway_api_key() -> str:
    """
    Return the Scholar Gateway API key from the environment.

    Raises:
        RuntimeError: if the key is not set. The message names the variable,
                      never its value.
    """
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"Missing Scholar Gateway API key. Set {API_KEY_ENV} in your .env file."
        )
    return key


class ScholarGatewayClient:
    """
    Minimal MCP client for the Scholar Gateway v2 endpoint.

    Errors raised from here include the HTTP status and response body, never
    the request headers (which carry the API key).
    """

    def __init__(self, api_key: str, timeout: float = 90.0):
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-API-Key": api_key,
        }
        self._http = httpx.Client(timeout=timeout)
        self._next_id = 1
        self._initialize()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ScholarGatewayClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _post(self, payload: dict) -> httpx.Response:
        resp = self._http.post(
            SCHOLAR_GATEWAY_MCP_URL, headers=self._headers, json=payload
        )
        if resp.status_code not in (200, 202):
            raise RuntimeError(
                f"Scholar Gateway request '{payload.get('method')}' failed "
                f"(HTTP {resp.status_code}): {resp.text[:500]}"
            )
        return resp

    def _request(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request and return its `result`."""
        payload = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        self._next_id += 1
        if params is not None:
            payload["params"] = params
        resp = self._post(payload)

        # Responses arrive as plain JSON or as a single SSE "data:" event.
        body = resp.text
        for line in body.splitlines():
            if line.startswith("data:"):
                body = line[len("data:"):].strip()
                break
        try:
            message = json.loads(body)
        except ValueError as exc:
            raise RuntimeError(
                f"Scholar Gateway returned a non-JSON response to '{method}'."
            ) from exc
        if "error" in message:
            raise RuntimeError(
                f"Scholar Gateway error on '{method}': {message['error']}"
            )
        return message.get("result", {})

    def _initialize(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "pubmed-mcp", "version": "1.0"},
            },
        )
        # Session-bound servers return an Mcp-Session-Id; the v2 server is
        # stateless, so there is nothing to carry forward.
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def list_tools(self) -> list[dict]:
        """Return the server's tools as Anthropic tool definitions."""
        tools = self._request("tools/list").get("tools", [])
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t["inputSchema"],
            }
            for t in tools
        ]

    def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Call a tool. Returns (text content, is_error)."""
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        text = "\n".join(
            block.get("text", "")
            for block in result.get("content", [])
            if block.get("type") == "text"
        )
        return text, bool(result.get("isError"))


def _mask(token: str) -> str:
    """Mask a secret for safe display (first/last 4 chars)."""
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}...{token[-4:]}"


def main() -> None:
    """Smoke-test the connection: handshake and list tools."""
    if not has_api_key():
        print(f"Error: {API_KEY_ENV} must be set in .env.")
        raise SystemExit(1)

    key = get_scholar_gateway_api_key()
    print(f"Endpoint: {SCHOLAR_GATEWAY_MCP_URL}")
    print(f"API key:  {_mask(key)}\n")
    try:
        with ScholarGatewayClient(key) as sg:
            tools = sg.list_tools()
    except (RuntimeError, httpx.HTTPError) as exc:
        print(f"FAILED: {exc}")
        raise SystemExit(1)

    print(f"Connected. Tools: {[t['name'] for t in tools]}")


if __name__ == "__main__":
    main()
