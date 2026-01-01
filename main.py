#!/usr/bin/env python3
"""
PubMed Research Agent using MCP Connector

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

# PubMed MCP Server URL
PUBMED_MCP_SERVER_URL = "https://pubmed.mcp.claude.com/mcp"

SYSTEM_PROMPT = """You are a biomedical research assistant that answers questions using ONLY information from PubMed.

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

If PubMed searches return no results or insufficient information:
- Try alternative search terms or broader queries
- If still unsuccessful, clearly state that you could not find relevant research in PubMed
- Do NOT fall back on internal knowledge to answer the question

Remember: Your value is in providing evidence-based answers from peer-reviewed literature, not general knowledge."""


def run_pubmed_agent(research_question: str, verbose: bool = True) -> str:
    """
    Run the PubMed research agent to answer a question using the MCP Connector.

    Args:
        research_question: The biomedical research question to answer
        verbose: Whether to print progress information

    Returns:
        The agent's response with citations
    """
    client = Anthropic()

    messages = [{"role": "user", "content": research_question}]

    if verbose:
        print(f"\n{'='*60}")
        print("PubMed Research Agent (MCP Connector)")
        print(f"{'='*60}")
        print(f"\nResearch Question: {research_question}\n")
        print("Searching PubMed...\n")

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
            mcp_servers=[
                {
                    "type": "url",
                    "url": PUBMED_MCP_SERVER_URL,
                    "name": "pubmed",
                }
            ],
            tools=[
                {
                    "type": "mcp_toolset",
                    "mcp_server_name": "pubmed",
                }
            ],
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
                    print(f"[Tool Call] {block.name}")
                    if block.name == "search_articles":
                        query = block.input.get("query", "")
                        print(f"  Query: {query}")
                    elif block.name == "get_article_metadata":
                        pmids = block.input.get("pmids", [])
                        print(f"  PMIDs: {', '.join(pmids[:5])}{'...' if len(pmids) > 5 else ''}")
                    elif block.name == "get_full_text_article":
                        pmc_ids = block.input.get("pmc_ids", [])
                        print(f"  PMC IDs: {', '.join(pmc_ids[:3])}")

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

        # Check if we're done (no more tool use needed)
        if response.stop_reason == "end_turn" and not has_tool_use:
            if verbose:
                print(f"\n{'='*60}")
                print("Research Complete")
                print(f"{'='*60}\n")
            return final_text

        # If there were tool uses, continue the conversation
        # The MCP connector handles tool execution, but we need to continue
        # if Claude wants to do more
        if has_tool_use:
            # Add assistant response to messages
            messages.append({"role": "assistant", "content": assistant_content})
            # Continue the loop - Claude may need to make more calls
            # or synthesize the results
            messages.append({
                "role": "user",
                "content": "Please continue analyzing the results and provide your answer."
            })
        else:
            # Got a response without tool use
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
