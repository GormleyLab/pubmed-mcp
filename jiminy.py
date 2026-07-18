#!/usr/bin/env python3
"""
Paper RAG Research Agent

An autonomous agent that searches your personal academic paper library via Paper RAG.
Uses the Anthropic MCP Connector to connect directly to the Paper RAG MCP server.

The agent uses your indexed papers to answer questions with proper citations.
"""

import os
import sys

from dotenv import load_dotenv
from anthropic import Anthropic

from paperrag_warmup import PAPERRAG_MCP_SERVER_URL, warm_up_paperrag

# Load environment variables from .env file
load_dotenv()

# Ensure Unicode answers (Greek letters, em dashes, etc.) print on Windows
# consoles, whose default cp1252 encoding raises UnicodeEncodeError otherwise.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

SYSTEM_PROMPT = """You are an academic research assistant that answers questions using ONLY information from the Paper RAG database.

Paper RAG is a personal academic paper library with semantic search capabilities. It contains papers that have been indexed with their full text, metadata, and citations.

CRITICAL RULES:
1. You must ONLY use information retrieved from Paper RAG searches to answer questions.
2. NEVER use your internal knowledge to provide facts, statistics, or claims about specific papers.
3. If you cannot find relevant information, say so clearly.
4. Always cite your sources with proper academic references using the BibTeX keys.

AVAILABLE TOOLS:
- search_papers: Semantic search through indexed papers (use natural language queries)
- get_paper_details: Get full metadata for a specific paper by BibTeX key
- database_stats: Get statistics about the paper database
- list_recent_papers: Show recently added papers
- generate_bibliography: Create a .bib file from paper keys

WORKFLOW:
1. Analyze the user's research question
2. Use search_papers to find relevant papers (try multiple search strategies if needed)
3. Use get_paper_details to get more information about promising papers
4. Synthesize findings ONLY from the retrieved papers
5. Provide proper citations using BibTeX keys

CITATION FORMAT:
For each claim or finding, cite the source using:
- Author names, title, year
- BibTeX key in brackets: [AuthorYear]
- Example: "Smith et al. found that polymer nanoparticles improve drug delivery [Smith2024]"

SEARCH TIPS:
- Use natural language queries - Paper RAG uses semantic search
- Be specific about concepts, methods, or materials
- Try different phrasings if initial searches yield insufficient results
- You can filter by year or section type if helpful

If searches return no results or insufficient information:
- Try alternative search terms or broader queries
- Use database_stats to understand what's in the library
- If still unsuccessful, clearly state what you searched and that you could not find relevant papers
- Do NOT fall back on internal knowledge to answer the question

Remember: Your value is in providing evidence-based answers from the user's personal paper library with proper citations."""


def run_paperrag_agent(research_question: str, verbose: bool = True) -> str:
    """
    Run the Paper RAG research agent to answer a question using the MCP Connector.

    Args:
        research_question: The research question to answer
        verbose: Whether to print progress information

    Returns:
        The agent's response with citations
    """
    # Explicit timeout + limited retries so a transient stall in the MCP
    # connector fails fast and visibly instead of hanging on the SDK defaults
    # (~10 min timeout with silent retries).
    client = Anthropic(timeout=120.0, max_retries=1)

    messages = [{"role": "user", "content": research_question}]

    if verbose:
        print(f"\n{'=' * 60}")
        print("Paper RAG Research Agent (MCP Connector)")
        print(f"{'=' * 60}")
        print(f"\nResearch Question: {research_question}\n")
        print("Searching Paper RAG...\n")

    # Get Paper RAG API key
    paperrag_api_key = os.environ.get("PAPERRAG_API_KEY")
    if not paperrag_api_key:
        return "Error: PAPERRAG_API_KEY not set. Please configure your API key in .env"

    # Wake the scale-to-zero server before the agent turn so the connector's
    # first tool call hits a warm instance (see warm_up_paperrag).
    warm_up_paperrag(paperrag_api_key, verbose=verbose)

    mcp_servers = [
        {
            "type": "url",
            "url": PAPERRAG_MCP_SERVER_URL,
            "name": "paper_rag",
            "authorization_token": paperrag_api_key,
        }
    ]

    tools = [
        {
            "type": "mcp_toolset",
            "mcp_server_name": "paper_rag",
        }
    ]

    # Agentic loop. The server-side MCP connector usually finishes in a single
    # turn (tools run server-side), so this rarely iterates more than once; the
    # cap only bounds pause_turn resumptions.
    iteration = 0
    max_iterations = 5  # Safety limit

    while iteration < max_iterations:
        iteration += 1

        # Call Claude with MCP Connector
        response = client.beta.messages.create(
            model="claude-sonnet-5",
            max_tokens=8096,
            system=SYSTEM_PROMPT,
            messages=messages,
            mcp_servers=mcp_servers,
            tools=tools,
            betas=["mcp-client-2025-11-20"],
        )

        # Process response content
        assistant_content = []
        has_tool_use = False
        final_text = ""

        for block in response.content:
            assistant_content.append(block)

            if block.type == "text":
                final_text += block.text

            elif block.type == "mcp_tool_use":
                has_tool_use = True
                if verbose:
                    print(f"[paper_rag] {block.name}")
                    if block.name == "search_papers":
                        query = block.input.get("query", "")
                        print(f"  Query: {query}")
                    elif block.name == "get_paper_details":
                        key = block.input.get("bibtex_key", "")
                        print(f"  BibTeX Key: {key}")
                    elif block.name == "generate_bibliography":
                        keys = block.input.get("bibtex_keys", [])
                        print(f"  Keys: {keys}")

            elif block.type == "mcp_tool_result":
                if verbose:
                    # Show brief result info
                    if hasattr(block, "content") and block.content:
                        for content_block in block.content:
                            if hasattr(content_block, "text"):
                                try:
                                    import json

                                    result_data = json.loads(content_block.text)
                                    if "results" in result_data:
                                        print(
                                            f"  Found: {len(result_data['results'])} results"
                                        )
                                    elif "papers" in result_data:
                                        print(
                                            f"  Found: {len(result_data['papers'])} papers"
                                        )
                                    elif "total_papers" in result_data:
                                        print(
                                            f"  Database: {result_data['total_papers']} papers"
                                        )
                                except (json.JSONDecodeError, TypeError):
                                    pass
                    print()

        # The MCP connector runs tools server-side within a single turn, so one
        # response can contain the tool calls, their results, AND the final answer.
        # Return as soon as the model finishes its turn.
        if response.stop_reason == "end_turn":
            if verbose:
                print(f"\n{'=' * 60}")
                print("Research Complete")
                print(f"{'=' * 60}\n")
            return final_text

        # Long-running server tool use may pause mid-turn; resume it by resending
        # the assistant's partial content (no canned "continue" prompt needed).
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": assistant_content})
            continue

        # Any other stop reason (e.g. max_tokens): return whatever text we have.
        if verbose:
            print(f"\n{'=' * 60}")
            print("Research Complete")
            print(f"{'=' * 60}\n")
        return final_text

    return "Error: Maximum iterations reached without completing the research."


def main():
    """Main entry point for the Paper RAG research agent."""
    print("\n" + "=" * 60)
    print("  Paper RAG Research Agent")
    print("  Answers questions using your indexed paper library")
    print("  Powered by Anthropic MCP Connector")
    print("=" * 60 + "\n")

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set your Anthropic API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)

    # Check for Paper RAG API key
    if not os.environ.get("PAPERRAG_API_KEY"):
        print("Error: PAPERRAG_API_KEY environment variable not set.")
        print("Please set your Paper RAG API key in .env")
        print("See docs/QUICK_START.md for instructions.")
        sys.exit(1)

    # Get research question from command line or prompt
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("Enter your research question:")
        print(
            "(Examples: 'What papers discuss polymer nanoparticles for drug delivery?'"
        )
        print("          'Find papers about machine learning in materials science')")
        print()
        question = input("> ").strip()

        if not question:
            print("No question provided. Exiting.")
            sys.exit(0)

    # Run the agent
    response = run_paperrag_agent(question)

    print("\n" + "=" * 60)
    print("  Answer")
    print("=" * 60 + "\n")
    print(response)


if __name__ == "__main__":
    main()
