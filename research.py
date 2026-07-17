#!/usr/bin/env python3
"""
Unified Research Agent

An autonomous agent that searches across multiple academic databases to answer research questions.
Uses the Anthropic MCP Connector to connect to:
- PubMed: Biomedical literature database
- Paper RAG: Personal academic paper library
- Scholar Gateway: Wiley academic articles

The agent ONLY uses search results from these databases to answer questions - it does not rely on
its internal knowledge for research facts.
"""

import os
import sys

from dotenv import load_dotenv
from anthropic import Anthropic

from scholar_auth import (
    SCHOLAR_GATEWAY_MCP_URL,
    get_scholar_gateway_token,
    has_client_credentials,
)

# Load environment variables from .env file
load_dotenv()

# Ensure Unicode answers (Greek letters, em dashes, etc.) print on Windows
# consoles, whose default cp1252 encoding raises UnicodeEncodeError otherwise.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# MCP Server URLs
PUBMED_MCP_SERVER_URL = "https://pubmed.mcp.claude.com/mcp"
PAPERRAG_MCP_SERVER_URL = "https://m76rjhx9i3.us-east-1.awsapprunner.com/mcp"
SCHOLAR_GATEWAY_MCP_SERVER_URL = SCHOLAR_GATEWAY_MCP_URL  # Wiley custom connector

SYSTEM_PROMPT = """You are an academic research assistant that answers questions using ONLY information from multiple research databases.

You have access to three research databases:
1. PubMed: Over 36 million citations for biomedical literature from MEDLINE, life science journals, and online books
2. Paper RAG: Personal academic paper library with semantic search capabilities
3. Scholar Gateway: Over 3 million articles from more than 1,300 Wiley journals across multiple disciplines

CRITICAL RULES:
1. You must ONLY use information retrieved from database searches to answer questions.
2. NEVER use your internal knowledge to provide facts, statistics, or claims.
3. If you cannot find relevant information, say so clearly.
4. Always cite your sources with proper academic references.
5. Choose the most appropriate database(s) for each query, or search multiple databases for comprehensive coverage.

AVAILABLE DATABASES AND TOOLS:

PubMed (pubmed):
- search_articles: Search PubMed with MeSH terms or keywords
- get_article_metadata: Get full abstracts and metadata for articles
- find_related_articles: Discover related papers
- Use for: Biomedical, life sciences, medical research questions

Paper RAG (paper_rag):
- search_papers: Semantic search through indexed papers
- get_paper_details: Get full metadata for specific papers
- database_stats: Get statistics about the paper database
- list_recent_papers: Show recently added papers
- generate_bibliography: Create a .bib file from paper keys
- Use for: Questions about papers in the user's personal library

Scholar Gateway (scholar_gateway):
- semantic_search: Semantic search through Wiley articles
- Use for: Multi-disciplinary research questions, Wiley journal content

WORKFLOW:
1. Analyze the user's research question
2. Determine which database(s) are most appropriate
3. Search relevant databases (try multiple search strategies if needed)
4. Retrieve detailed information from promising results
5. Synthesize findings ONLY from the retrieved articles/papers
6. Provide proper citations in your response

CITATION FORMAT:
For each claim or finding, cite the source using:
- Author names, title, journal, year
- PMID and DOI when available (PubMed)
- BibTeX keys when available (Paper RAG)
- DOI when available (Scholar Gateway)
- Example: "Smith et al. found that... (PMID: 12345678, DOI: 10.1000/example)"

SEARCH TIPS:
- PubMed: Use MeSH terms, combine with AND/OR operators, use field tags like [Title], [Author]
- Paper RAG: Use natural language queries - semantic search works well
- Scholar Gateway: Use natural language queries - semantic search works well
- Try broader terms if specific searches yield no results
- Search multiple databases for comprehensive coverage when appropriate

If searches return no results or insufficient information:
- Try alternative search terms or broader queries
- Try different databases that might have relevant content
- If still unsuccessful, clearly state what you searched and that you could not find relevant research
- Do NOT fall back on internal knowledge to answer the question

Remember: Your value is in providing evidence-based answers from peer-reviewed academic literature across multiple databases, not general knowledge."""


def run_research_agent(research_question: str, verbose: bool = True) -> str:
    """
    Run the unified research agent to answer a question using all three MCP servers.

    Args:
        research_question: The research question to answer
        verbose: Whether to print progress information

    Returns:
        The agent's response with citations
    """
    client = Anthropic()

    messages = [{"role": "user", "content": research_question}]

    if verbose:
        print(f"\n{'=' * 60}")
        print("Unified Research Agent (MCP Connector)")
        print(f"{'=' * 60}")
        print(f"\nResearch Question: {research_question}\n")
        print("Searching across multiple databases...\n")

    # Get API keys/tokens from environment variables
    paperrag_api_key = os.environ.get("PAPERRAG_API_KEY")

    # Obtain a Scholar Gateway token. Prefer the client_credentials flow (custom
    # connector); fall back to a legacy static token. If neither is available, we
    # simply omit Scholar Gateway so it can't fail the whole request — the MCP
    # connector rejects the entire call if any one server's auth is invalid.
    scholar_gateway_token = None
    if has_client_credentials():
        try:
            scholar_gateway_token = get_scholar_gateway_token()
        except RuntimeError as exc:
            if verbose:
                print(f"Warning: could not obtain Scholar Gateway token: {exc}\n")
    else:
        scholar_gateway_token = os.environ.get("SCHOLAR_GATEWAY_TOKEN")

    # Build MCP servers list. PubMed and Paper RAG are always included; Scholar
    # Gateway is added only when a token was obtained.
    # Authorization tokens are kept secret in .env file.
    mcp_servers = [
        {
            "type": "url",
            "url": PUBMED_MCP_SERVER_URL,
            "name": "pubmed",
        },
        {
            "type": "url",
            "url": PAPERRAG_MCP_SERVER_URL,
            "name": "paper_rag",
            "authorization_token": paperrag_api_key,
        },
    ]
    if scholar_gateway_token:
        mcp_servers.append(
            {
                "type": "url",
                "url": SCHOLAR_GATEWAY_MCP_SERVER_URL,
                "name": "scholar_gateway",
                "authorization_token": scholar_gateway_token,
            }
        )

    # Warn if API keys/tokens are missing
    if not paperrag_api_key and verbose:
        print("Warning: PAPERRAG_API_KEY not set. Paper RAG will not be available.\n")
    if not scholar_gateway_token and verbose:
        print(
            "Warning: no Scholar Gateway credentials "
            "(SCHOLAR_GATEWAY_CLIENT_ID/SECRET or SCHOLAR_GATEWAY_TOKEN). "
            "Scholar Gateway will not be available.\n"
        )

    # Build tools list to match the servers that were actually configured.
    tools = [
        {"type": "mcp_toolset", "mcp_server_name": "pubmed"},
        {"type": "mcp_toolset", "mcp_server_name": "paper_rag"},
    ]
    if scholar_gateway_token:
        tools.append({"type": "mcp_toolset", "mcp_server_name": "scholar_gateway"})

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
                    server_name = getattr(block, "server_name", "unknown")
                    print(f"[{server_name}] {block.name}")
                    if block.name == "search_articles":
                        query = block.input.get("query", "")
                        print(f"  Query: {query}")
                    elif block.name == "get_article_metadata":
                        pmids = block.input.get("pmids", [])
                        print(
                            f"  PMIDs: {', '.join(pmids[:5])}{'...' if len(pmids) > 5 else ''}"
                        )
                    elif block.name == "get_full_text_article":
                        pmc_ids = block.input.get("pmc_ids", [])
                        print(f"  PMC IDs: {', '.join(pmc_ids[:3])}")
                    elif block.name == "find_related_articles":
                        pmids = block.input.get("pmids", [])
                        print(f"  Finding related to: {', '.join(pmids[:3])}")
                    elif block.name == "search_papers":
                        query = block.input.get("query", "")
                        print(f"  Query: {query}")
                    elif block.name == "get_paper_details":
                        key = block.input.get("bibtex_key", "")
                        print(f"  BibTeX Key: {key}")
                    elif (
                        block.name == "semantic_search"
                        or block.name == "semanticSearch"
                    ):
                        query = block.input.get(
                            "query", block.input.get("search_query", "")
                        )
                        print(f"  Query: {query}")

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
                                        print(
                                            f"  Found: {len(result_data['articles'])} articles"
                                        )
                                    elif "results" in result_data:
                                        print(
                                            f"  Found: {len(result_data['results'])} results"
                                        )
                                    elif "papers" in result_data:
                                        print(
                                            f"  Found: {len(result_data['papers'])} papers"
                                        )
                                    elif "total_count" in result_data:
                                        print(
                                            f"  Total matches: {result_data.get('total_count', 'N/A')}"
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
    """Main entry point for the unified research agent."""
    print("\n" + "=" * 60)
    print("  Unified Research Agent")
    print("  Answers questions using PubMed, Paper RAG, and Scholar Gateway")
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
        print("Enter your research question:")
        print("(Examples: 'What are the latest treatments for Type 2 diabetes?'")
        print(
            "          'What papers discuss polymer nanoparticles for drug delivery?'"
        )
        print("          'What are recent advances in organic solar cells?')")
        print()
        question = input("> ").strip()

        if not question:
            print("No question provided. Exiting.")
            sys.exit(0)

    # Run the agent
    response = run_research_agent(question)

    print("\n" + "=" * 60)
    print("  Answer")
    print("=" * 60 + "\n")
    print(response)


if __name__ == "__main__":
    main()
