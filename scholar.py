#!/usr/bin/env python3
"""
Scholar Gateway Research Agent

An autonomous agent that searches Wiley's academic articles via Scholar Gateway.
Uses the Anthropic MCP Connector to connect directly to the Scholar Gateway MCP server.

The agent ONLY uses Scholar Gateway search results to answer questions - it does not
rely on its internal knowledge for research facts.
"""

import os
import sys

from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables from .env file
load_dotenv()

# Scholar Gateway MCP Server URL
SCHOLAR_GATEWAY_MCP_SERVER_URL = "https://connector.scholargateway.ai/mcp"

SYSTEM_PROMPT = """You are an academic research assistant that answers questions using ONLY information from Scholar Gateway.

Scholar Gateway provides access to over 3 million articles from more than 1,300 Wiley journals across multiple disciplines including science, technology, medicine, social sciences, and humanities.

CRITICAL RULES:
1. You must ONLY use information retrieved from Scholar Gateway searches to answer questions.
2. NEVER use your internal knowledge to provide facts, statistics, or claims.
3. If you cannot find relevant information, say so clearly.
4. Always cite your sources with proper academic references.

WORKFLOW:
1. Analyze the user's research question
2. Use semantic_search to find relevant articles (try multiple search strategies if needed)
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
    Run the Scholar Gateway research agent to answer a question using the MCP Connector.

    Args:
        research_question: The research question to answer
        verbose: Whether to print progress information

    Returns:
        The agent's response with citations
    """
    client = Anthropic()

    messages = [{"role": "user", "content": research_question}]

    if verbose:
        print(f"\n{'='*60}")
        print("Scholar Gateway Research Agent (MCP Connector)")
        print(f"{'='*60}")
        print(f"\nResearch Question: {research_question}\n")
        print("Searching Scholar Gateway...\n")

    # Get Scholar Gateway token
    scholar_gateway_token = os.environ.get("SCHOLAR_GATEWAY_TOKEN")
    if not scholar_gateway_token:
        return "Error: SCHOLAR_GATEWAY_TOKEN not set. Please configure your OAuth token."

    mcp_servers = [
        {
            "type": "url",
            "url": SCHOLAR_GATEWAY_MCP_SERVER_URL,
            "name": "scholar_gateway",
            "authorization_token": scholar_gateway_token,
        }
    ]

    tools = [
        {
            "type": "mcp_toolset",
            "mcp_server_name": "scholar_gateway",
        }
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
                    print(f"[scholar_gateway] {block.name}")
                    if block.name == "semantic_search" or block.name == "semanticSearch":
                        query = block.input.get("query", block.input.get("search_query", ""))
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
                                    if "results" in result_data:
                                        print(f"  Found: {len(result_data['results'])} results")
                                    elif "articles" in result_data:
                                        print(f"  Found: {len(result_data['articles'])} articles")
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
        if has_tool_use:
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({
                "role": "user",
                "content": "Please continue analyzing the results and provide your answer."
            })
        else:
            if verbose:
                print(f"\n{'='*60}")
                print("Research Complete")
                print(f"{'='*60}\n")
            return final_text

    return "Error: Maximum iterations reached without completing the research."


def main():
    """Main entry point for the Scholar Gateway research agent."""
    print("\n" + "=" * 60)
    print("  Scholar Gateway Research Agent")
    print("  Answers questions using Wiley academic articles")
    print("  Powered by Anthropic MCP Connector")
    print("=" * 60 + "\n")

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set your Anthropic API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)

    # Check for Scholar Gateway token
    if not os.environ.get("SCHOLAR_GATEWAY_TOKEN"):
        print("Error: SCHOLAR_GATEWAY_TOKEN environment variable not set.")
        print("Please set your Scholar Gateway OAuth token.")
        print("See README.md for instructions on obtaining a token.")
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
