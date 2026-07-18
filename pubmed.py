#!/usr/bin/env python3
"""
PubMed Research Agent

An autonomous agent that searches PubMed to answer biomedical research questions.
Uses the Anthropic MCP Connector to connect directly to the PubMed MCP server.

The agent ONLY uses PubMed search results to answer questions - it does not rely on
its internal knowledge for biomedical facts.
"""

import os
import sys

from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables from .env file
load_dotenv()

# Ensure Unicode answers (Greek letters, em dashes, etc.) print on Windows
# consoles, whose default cp1252 encoding raises UnicodeEncodeError otherwise.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# PubMed MCP Server URL
PUBMED_MCP_SERVER_URL = "https://pubmed.mcp.claude.com/mcp"

SYSTEM_PROMPT = """You are a biomedical research assistant that answers questions using ONLY information from PubMed.

PubMed provides access to over 36 million citations for biomedical literature from MEDLINE, life science journals, and online books.

CRITICAL RULES:
1. You must ONLY use information retrieved from PubMed searches to answer questions.
2. NEVER use your internal knowledge to provide biomedical facts, statistics, or claims.
3. If you cannot find relevant information in PubMed, say so clearly.
4. Always cite your sources with proper PubMed references.

WORKFLOW:
1. Analyze the user's research question
2. Use search_articles to find relevant papers (try multiple search strategies if needed)
3. Use get_article_metadata to get full abstracts for promising articles
4. If needed, use find_related_articles to discover more relevant papers
5. Synthesize findings ONLY from the retrieved articles
6. Provide proper citations in your response

CITATION FORMAT:
For each claim or finding, cite the source using:
- Author names, title, journal, year
- PMID and DOI when available
- Example: "Smith et al. found that... (PMID: 12345678, DOI: 10.1000/example)"

SEARCH TIPS:
- Use MeSH terms for precise biomedical searching
- Combine terms with AND/OR operators
- Use field tags like [Title], [Author], [Journal]
- Try broader terms if specific searches yield no results

If PubMed searches return no results or insufficient information:
- Try alternative search terms or broader queries
- If still unsuccessful, clearly state what you searched and that you could not find relevant research
- Do NOT fall back on internal knowledge to answer the question

Remember: Your value is in providing evidence-based answers from peer-reviewed biomedical literature, not general knowledge."""


def run_pubmed_agent(research_question: str, verbose: bool = True) -> str:
    """
    Run the PubMed research agent to answer a question using the MCP Connector.

    Args:
        research_question: The biomedical research question to answer
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
        print(f"\n{'='*60}")
        print("PubMed Research Agent (MCP Connector)")
        print(f"{'='*60}")
        print(f"\nResearch Question: {research_question}\n")
        print("Searching PubMed...\n")

    mcp_servers = [
        {
            "type": "url",
            "url": PUBMED_MCP_SERVER_URL,
            "name": "pubmed",
        }
    ]

    tools = [
        {
            "type": "mcp_toolset",
            "mcp_server_name": "pubmed",
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
                    print(f"[pubmed] {block.name}")
                    if block.name == "search_articles":
                        query = block.input.get("query", "")
                        print(f"  Query: {query}")
                    elif block.name == "get_article_metadata":
                        pmids = block.input.get("pmids", [])
                        print(f"  PMIDs: {', '.join(pmids[:5])}{'...' if len(pmids) > 5 else ''}")
                    elif block.name == "get_full_text_article":
                        pmc_ids = block.input.get("pmc_ids", [])
                        print(f"  PMC IDs: {', '.join(pmc_ids[:3])}")
                    elif block.name == "find_related_articles":
                        pmids = block.input.get("pmids", [])
                        print(f"  Finding related to: {', '.join(pmids[:3])}")

            elif block.type == "mcp_tool_result":
                if verbose:
                    # Show brief result info
                    if hasattr(block, "content") and block.content:
                        for content_block in block.content:
                            if hasattr(content_block, "text"):
                                try:
                                    import json
                                    result_data = json.loads(content_block.text)
                                    if "articles" in result_data:
                                        print(f"  Found: {len(result_data['articles'])} articles")
                                    elif "total_count" in result_data:
                                        print(f"  Total matches: {result_data.get('total_count', 'N/A')}")
                                except (json.JSONDecodeError, TypeError):
                                    pass
                    print()

        # The MCP connector runs tools server-side within a single turn, so one
        # response can contain the tool calls, their results, AND the final answer.
        # Return as soon as the model finishes its turn.
        if response.stop_reason == "end_turn":
            if verbose:
                print(f"\n{'='*60}")
                print("Research Complete")
                print(f"{'='*60}\n")
            return final_text

        # Long-running server tool use may pause mid-turn; resume it by resending
        # the assistant's partial content (no canned "continue" prompt needed).
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": assistant_content})
            continue

        # Any other stop reason (e.g. max_tokens): return whatever text we have.
        if verbose:
            print(f"\n{'='*60}")
            print("Research Complete")
            print(f"{'='*60}\n")
        return final_text

    return "Error: Maximum iterations reached without completing the research."


def main():
    """Main entry point for the PubMed research agent."""
    print("\n" + "=" * 60)
    print("  PubMed Research Agent")
    print("  Answers biomedical questions using PubMed literature")
    print("  Powered by Anthropic MCP Connector")
    print("=" * 60 + "\n")

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set your Anthropic API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)

    # Get research question from command line or prompt
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("Enter your biomedical research question:")
        print("(Examples: 'What are the latest treatments for Type 2 diabetes?'")
        print("          'What is the evidence for mRNA vaccines in cancer treatment?')")
        print()
        question = input("> ").strip()

        if not question:
            print("No question provided. Exiting.")
            sys.exit(0)

    # Run the agent
    response = run_pubmed_agent(question)

    print("\n" + "=" * 60)
    print("  Answer")
    print("=" * 60 + "\n")
    print(response)


if __name__ == "__main__":
    main()
