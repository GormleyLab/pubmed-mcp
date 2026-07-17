#!/usr/bin/env python3
"""
Scholar Gateway authentication helper.

Fetches a Scholar Gateway access token using the OAuth2 client_credentials grant
against Wiley's custom connector endpoint. This is a non-interactive,
machine-to-machine flow: the Client ID and Client Secret are read from the
environment (never hardcoded, never logged), exchanged for a short-lived access
token, and that token is used as the MCP Connector authorization token.

Run this module directly to smoke-test just the OAuth step:

    python scholar_auth.py
"""

import base64
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

# Non-secret endpoint URLs (safe to keep in source).
SCHOLAR_GATEWAY_MCP_URL = "https://custom-connector-v1.scholargateway.ai/mcp"
SCHOLAR_GATEWAY_TOKEN_URL = (
    "https://custom-connector-v1.scholargateway.ai/oauth2/token"
    "?grant_type=client_credentials"
)

# Environment variable names holding the secret credentials (values live in .env).
CLIENT_ID_ENV = "SCHOLAR_GATEWAY_CLIENT_ID"
CLIENT_SECRET_ENV = "SCHOLAR_GATEWAY_CLIENT_SECRET"


def has_client_credentials() -> bool:
    """Return True if both client-credential env vars are set."""
    return bool(os.environ.get(CLIENT_ID_ENV)) and bool(
        os.environ.get(CLIENT_SECRET_ENV)
    )


def get_scholar_gateway_token(timeout: float = 30.0) -> str:
    """
    Fetch a Scholar Gateway access token via the client_credentials grant.

    Reads SCHOLAR_GATEWAY_CLIENT_ID and SCHOLAR_GATEWAY_CLIENT_SECRET from the
    environment and exchanges them at the token endpoint.

    Wiley has not published whether the credentials belong in the HTTP Basic
    auth header or the form body, so we try Basic auth first and fall back to
    sending the credentials in the body if the server rejects it.

    Returns:
        The access token string.

    Raises:
        RuntimeError: if the credentials are missing or the token request fails.
                      The error includes the HTTP status and response body, but
                      never the client secret or the token itself.
    """
    client_id = os.environ.get(CLIENT_ID_ENV)
    client_secret = os.environ.get(CLIENT_SECRET_ENV)

    if not client_id or not client_secret:
        raise RuntimeError(
            f"Missing Scholar Gateway credentials. Set {CLIENT_ID_ENV} and "
            f"{CLIENT_SECRET_ENV} in your .env file."
        )

    with httpx.Client(timeout=timeout) as client:
        # Attempt 1: HTTP Basic auth header (RFC 6749 preferred method).
        basic = base64.b64encode(
            f"{client_id}:{client_secret}".encode("utf-8")
        ).decode("ascii")
        resp = client.post(
            SCHOLAR_GATEWAY_TOKEN_URL,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={"grant_type": "client_credentials"},
        )

        # Attempt 2: some servers expect the credentials in the body instead.
        if resp.status_code in (400, 401):
            resp = client.post(
                SCHOLAR_GATEWAY_TOKEN_URL,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
            )

    if resp.status_code != 200:
        # Body may reveal what the server expected (e.g. a missing scope);
        # it does not contain our secret.
        raise RuntimeError(
            f"Scholar Gateway token request failed (HTTP {resp.status_code}): "
            f"{resp.text}"
        )

    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Scholar Gateway token endpoint returned non-JSON response: {resp.text}"
        ) from exc

    token = payload.get("access_token")
    if not token:
        raise RuntimeError(
            f"Scholar Gateway token response missing 'access_token'. "
            f"Keys returned: {sorted(payload.keys())}"
        )

    return token


def _mask(token: str) -> str:
    """Mask a token for safe display (first/last 4 chars)."""
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}...{token[-4:]}"


def main() -> None:
    """Smoke-test the OAuth step in isolation."""
    if not has_client_credentials():
        print(
            f"Error: {CLIENT_ID_ENV} and {CLIENT_SECRET_ENV} must be set in .env."
        )
        raise SystemExit(1)

    print("Requesting Scholar Gateway token via client_credentials...")
    print(f"Token endpoint: {SCHOLAR_GATEWAY_TOKEN_URL}")
    try:
        token = get_scholar_gateway_token()
    except RuntimeError as exc:
        print(f"\nFAILED: {exc}")
        raise SystemExit(1)

    print(f"\nToken acquired: {_mask(token)}")
    print("OAuth client_credentials flow works.")


if __name__ == "__main__":
    main()
