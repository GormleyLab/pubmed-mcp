#!/usr/bin/env python3
"""
Scholar Gateway Research Agent

An autonomous agent that searches Wiley's academic articles via Scholar Gateway.
Connects to the Scholar Gateway MCP server client-side (see scholar_auth.py).

The agent ONLY uses Scholar Gateway search results to answer questions - it does not
rely on its internal knowledge for research facts.
"""

import json
import os
import sys

import httpx

from dotenv import load_dotenv
from anthropic import Anthropic

from scholar_auth import (
    API_KEY_ENV,
    ScholarGatewayClient,
    get_scholar_gateway_api_key,
    has_api_key,
)

# Load environment variables from .env file
load_dotenv()

# Ensure Unicode answers (Greek letters, em dashes, etc.) print on Windows
# consoles, whose default cp1252 encoding raises UnicodeEncodeError otherwise.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

SYSTEM_PROMPT = """You are an academic research assistant that answers questions using ONLY information from Scholar Gateway.

Scholar Gateway provides access to over 3 million articles from more than 1,300 Wiley journals across multiple disciplines including science, technology, medicine, social sciences, and humanities.

CRITICAL RULES:
1. You must ONLY use information retrieved from Scholar Gateway searches to answer questions.
2. NEVER use your internal knowledge to provide facts, statistics, or claims.
3. If you cannot find relevant information, say so clearly.
4. Always cite your sources with proper academic references.

WORKFLOW:
1. Analyze the user's research question
2. Use search_wiley_fulltext to find relevant articles (try multiple search strategies if needed)
3. Analyze the returned article content and metadata
4. Synthesize findings ONLY from the retrieved articles
5. Provide proper citations in your response

CITATION FORMAT:
For each claim or finding, cite the source using:
- Author names, title, journal, year
- DOI when available
- Example: "Smith et al. found that... (DOI: 10.1002/example)"

SEARCH TIPS:
- Use natural language queries - Scholar Gateway uses semantic search
- Be specific about the topic or concept you're researching
- Try different phrasings if initial searches yield insufficient results

If searches return no results or insufficient information:
- Try alternative search terms or broader queries
- If still unsuccessful, clearly state what you searched and that you could not find relevant research
- Do NOT fall back on internal knowledge to answer the question

Remember: Your value is in providing evidence-based answers from peer-reviewed academic literature, not general knowledge."""


def run_scholar_agent(research_question: str, verbose: bool = True) -> str:
    """
    Run the Scholar Gateway research agent to answer a question.

    Args:
        research_question: The research question to answer
        verbose: Whether to print progress information

    Returns:
        The agent's response with citations
    """
    # Explicit timeout + limited retries so a transient API stall
    # fails fast and visibly instead of hanging on the SDK defaults
    # (~10 min timeout with silent retries).
    client = Anthropic(timeout=120.0, max_retries=1)

    messages = [{"role": "user", "content": research_question}]

    if verbose:
        print(f"\n{'='*60}")
        print("Scholar Gateway Research Agent")
        print(f"{'='*60}")
        print(f"\nResearch Question: {research_question}\n")
        print("Searching Scholar Gateway...\n")

    # Connect to Scholar Gateway client-side: the v2 server requires an
    # X-API-Key header, which the server-side MCP connector cannot send.
    try:
        scholar_gateway = ScholarGatewayClient(get_scholar_gateway_api_key())
        tools = scholar_gateway.list_tools()
    except (RuntimeError, httpx.HTTPError) as exc:
        return f"Error: could not connect to Scholar Gateway. {exc}"

    # Agentic loop: the model requests searches, we run them against Scholar
    # Gateway and send back the results until it finishes its answer.
    iteration = 0
    max_iterations = 10  # Safety limit

    with scholar_gateway:
        while iteration < max_iterations:
            iteration += 1

            response = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tools,
            )

            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )

            if response.stop_reason != "tool_use":
                if verbose:
                    print(f"\n{'='*60}")
                    print("Research Complete")
                    print(f"{'='*60}\n")
                return final_text

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if verbose:
                    print(f"[scholar_gateway] {block.name}")
                    print(f"  Query: {block.input.get('query', '')}")
                try:
                    text, is_error = scholar_gateway.call_tool(block.name, block.input)
                except (RuntimeError, httpx.HTTPError) as exc:
                    text, is_error = f"Tool call failed: {exc}", True
                if verbose and not is_error:
                    try:
                        print(f"  Found: {json.loads(text).get('count', '?')} passages\n")
                    except (json.JSONDecodeError, AttributeError):
                        print()
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": text,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

    return "Error: Maximum iterations reached without completing the research."


def main():
    """Main entry point for the Scholar Gateway research agent."""
    print("\n" + "=" * 60)
    print("  Scholar Gateway Research Agent")
    print("  Answers questions using Wiley academic articles")
    print("  Powered by Claude + Scholar Gateway")
    print("=" * 60 + "\n")

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set your Anthropic API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)

    # Check for the Scholar Gateway API key
    if not has_api_key():
        print(f"Error: {API_KEY_ENV} not set in your .env file.")
        print("See README.md for details.")
        sys.exit(1)

    # Get research question from command line or prompt
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("Enter your research question:")
        print("(Examples: 'What are recent advances in organic solar cells?'")
        print("          'What does the research say about remote work productivity?')")
        print()
        question = input("> ").strip()

        if not question:
            print("No question provided. Exiting.")
            sys.exit(0)

    # Run the agent
    response = run_scholar_agent(question)

    print("\n" + "=" * 60)
    print("  Answer")
    print("=" * 60 + "\n")
    print(response)


if __name__ == "__main__":
    main()
