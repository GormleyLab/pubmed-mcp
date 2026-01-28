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

# Load environment variables from .env file
load_dotenv()

# MCP Server URLs
PUBMED_MCP_SERVER_URL = "https://pubmed.mcp.claude.com/mcp"
PAPERRAG_MCP_SERVER_URL = "https://m76rjhx9i3.us-east-1.awsapprunner.com/mcp"
SCHOLAR_GATEWAY_MCP_SERVER_URL = "https://connector.scholargateway.ai/mcp"

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
    scholar_gateway_token = os.environ.get("SCHOLAR_GATEWAY_TOKEN")

    # Build MCP servers list with all three servers explicitly listed
    # Authorization tokens are kept secret in .env file
    mcp_servers = [
        {
            "type": "url",
            "url": "https://pubmed.mcp.claude.com/mcp",
            "name": "pubmed",
        },
        {
            "type": "url",
            "url": "https://m76rjhx9i3.us-east-1.awsapprunner.com/mcp",
            "name": "paper_rag",
            "authorization_token": paperrag_api_key,
        },
        {
            "type": "url",
            "url": "https://connector.scholargateway.ai/mcp",
            "name": "scholar_gateway",
            "authorization_token": scholar_gateway_token,
        },
    ]

    # Warn if API keys/tokens are missing
    if not paperrag_api_key and verbose:
        print("Warning: PAPERRAG_API_KEY not set. Paper RAG will not be available.\n")
    if not scholar_gateway_token and verbose:
        print(
            "Warning: SCHOLAR_GATEWAY_TOKEN not set. Scholar Gateway will not be available.\n"
        )

    # Build tools list - explicitly include all three tool sets
    tools = [
        {
            "type": "mcp_toolset",
            "mcp_server_name": "pubmed",
        },
        {
            "type": "mcp_toolset",
            "mcp_server_name": "paper_rag",
        },
        {
            "type": "mcp_toolset",
            "mcp_server_name": "scholar_gateway",
        },
    ]

    # Agentic loop - continue until we get a final response
    iteration = 0
    max_iterations = 20  # Safety limit

    while iteration < max_iterations:
        iteration += 1

        # Call Claude with MCP Connector
        response = client.beta.messages.create(
            model="claude-sonnet-4-20250514",
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

        # Check if we're done (no more tool use needed)
        if response.stop_reason == "end_turn" and not has_tool_use:
            if verbose:
                print(f"\n{'=' * 60}")
                print("Research Complete")
                print(f"{'=' * 60}\n")
            return final_text

        # If there were tool uses, continue the conversation
        if has_tool_use:
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append(
                {
                    "role": "user",
                    "content": "Please continue analyzing the results and provide your answer.",
                }
            )
        else:
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
