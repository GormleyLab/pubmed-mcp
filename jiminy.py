#!/usr/bin/env python3
"""
Jiminy Research Agent

An autonomous agent that searches your personal paper database using the Jiminy MCP server.
Uses the Anthropic API with manual tool definitions that route to the RunPod endpoint.

The agent uses your curated paper database to answer research questions based on
papers you've uploaded and indexed.
"""

import os
import sys
import json
import httpx

from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables from .env file
load_dotenv()

# RunPod endpoint configuration
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")


def get_runpod_base_url() -> str:
    """Get the RunPod API base URL."""
    return f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"


# Tool definitions for Jiminy MCP server
JIMINY_TOOLS = [
    {
        "name": "search_papers",
        "description": "Search the paper database for relevant content using semantic search. Use this to find papers related to a research question or topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string (natural language)"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-20)",
                    "default": 5
                },
                "filter_section": {
                    "type": "string",
                    "description": "Filter by section type: Methods, Results, Discussion, or Introduction",
                    "enum": ["Methods", "Results", "Discussion", "Introduction"]
                },
                "min_year": {
                    "type": "integer",
                    "description": "Only papers from this year onwards"
                },
                "output_format": {
                    "type": "string",
                    "description": "'text' for human-readable or 'json' for structured data",
                    "default": "text",
                    "enum": ["text", "json"]
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_paper_details",
        "description": "Retrieve complete information about a specific paper including metadata and BibTeX entry.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bibtex_key": {
                    "type": "string",
                    "description": "The BibTeX key of the paper (e.g., 'Smith2024')"
                }
            },
            "required": ["bibtex_key"]
        }
    },
    {
        "name": "list_recent_papers",
        "description": "Show recently added papers in the database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of papers to show",
                    "default": 10
                }
            }
        }
    },
    {
        "name": "database_stats",
        "description": "Get statistics about the paper database including paper count and year distribution.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "generate_bibliography",
        "description": "Create a BibTeX bibliography for specified papers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bibtex_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "BibTeX keys to include in the bibliography"
                },
                "include_abstracts": {
                    "type": "boolean",
                    "description": "Include abstract field in BibTeX entries",
                    "default": False
                }
            },
            "required": ["bibtex_keys"]
        }
    },
    {
        "name": "get_paper_pdf",
        "description": "Retrieve the PDF file for a specific paper as base64-encoded data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bibtex_key": {
                    "type": "string",
                    "description": "The BibTeX key of the paper"
                }
            },
            "required": ["bibtex_key"]
        }
    },
    {
        "name": "delete_paper",
        "description": "Delete a paper from the database and optionally remove associated files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bibtex_key": {
                    "type": "string",
                    "description": "The BibTeX key of the paper to delete"
                },
                "delete_files": {
                    "type": "boolean",
                    "description": "Whether to delete associated PDF and .bib files",
                    "default": True
                }
            },
            "required": ["bibtex_key"]
        }
    },
    {
        "name": "add_paper_from_upload",
        "description": "Add a PDF paper to the database from base64-encoded data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pdf_data": {
                    "type": "string",
                    "description": "Base64-encoded PDF file content"
                },
                "filename": {
                    "type": "string",
                    "description": "Original filename of the PDF"
                },
                "custom_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags to associate with the paper"
                }
            },
            "required": ["pdf_data", "filename"]
        }
    }
]

SYSTEM_PROMPT = """You are a research assistant that answers questions using ONLY information from the Jiminy paper database.

Jiminy is a personal paper database containing curated research papers that have been uploaded and indexed. It uses semantic search to find relevant content across paper sections.

CRITICAL RULES:
1. You must ONLY use information retrieved from Jiminy searches to answer questions.
2. NEVER use your internal knowledge to provide facts, statistics, or claims about specific research.
3. If you cannot find relevant information in the database, say so clearly.
4. Always cite your sources with proper references including BibTeX keys.

WORKFLOW:
1. Analyze the user's research question
2. Use search_papers to find relevant papers (try multiple search strategies if needed)
3. Use get_paper_details to get full information for promising papers
4. Optionally use database_stats or list_recent_papers to understand what's available
5. Synthesize findings ONLY from the retrieved papers
6. Provide proper citations in your response

CITATION FORMAT:
For each claim or finding, cite the source using:
- Author names, title, year
- BibTeX key for reference
- Example: "Smith et al. found that... (Smith2024)"

SEARCH TIPS:
- Use natural language queries - Jiminy uses semantic search
- Filter by section (Methods, Results, Discussion, Introduction) for targeted searches
- Use min_year to focus on recent papers
- Try different phrasings if initial searches yield insufficient results

If searches return no results or insufficient information:
- Try alternative search terms or broader queries
- Use list_recent_papers to see what's in the database
- If still unsuccessful, clearly state what you searched and that you could not find relevant papers
- Do NOT fall back on internal knowledge to answer the question

Remember: Your value is in providing evidence-based answers from the user's curated paper database, not general knowledge."""


def call_jiminy_tool(tool_name: str, arguments: dict, timeout: float = 120.0) -> str:
    """
    Call a Jiminy MCP tool via the RunPod endpoint.

    Args:
        tool_name: Name of the tool to call
        arguments: Tool arguments
        timeout: Request timeout in seconds

    Returns:
        Tool result as string
    """
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "mode": "tool_call",
            "tool": tool_name,
            "arguments": arguments
        }
    }

    with httpx.Client() as client:
        response = client.post(
            f"{get_runpod_base_url()}/runsync",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()

    status = result.get("status")

    if status == "FAILED":
        error = result.get("error", "Unknown error")
        return f"Error: {error}"

    if status in ("IN_QUEUE", "IN_PROGRESS"):
        return f"Error: Job {status.lower().replace('_', ' ')} - try again later"

    output = result.get("output", {})

    if "result" in output:
        return output["result"]

    if "error" in output:
        return f"Error: {output['error']}"

    return str(output)


def run_jiminy_agent(research_question: str, verbose: bool = True) -> str:
    """
    Run the Jiminy research agent to answer a question.

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
        print("Jiminy Research Agent")
        print(f"{'='*60}")
        print(f"\nResearch Question: {research_question}\n")
        print("Searching paper database...\n")

    # Agentic loop - continue until we get a final response
    iteration = 0
    max_iterations = 20  # Safety limit

    while iteration < max_iterations:
        iteration += 1

        # Call Claude with Jiminy tools
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8096,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=JIMINY_TOOLS,
        )

        # Process response content
        assistant_content = []
        tool_results = []
        has_tool_use = False
        final_text = ""

        for block in response.content:
            assistant_content.append(block)

            if block.type == "text":
                final_text += block.text

            elif block.type == "tool_use":
                has_tool_use = True
                tool_name = block.name
                tool_input = block.input

                if verbose:
                    print(f"[jiminy] {tool_name}")
                    if tool_name == "search_papers":
                        query = tool_input.get("query", "")
                        print(f"  Query: {query}")
                        if tool_input.get("filter_section"):
                            print(f"  Section: {tool_input['filter_section']}")
                        if tool_input.get("min_year"):
                            print(f"  Min year: {tool_input['min_year']}")
                    elif tool_name == "get_paper_details":
                        print(f"  BibTeX key: {tool_input.get('bibtex_key', '')}")
                    elif tool_name == "list_recent_papers":
                        print(f"  Count: {tool_input.get('n', 10)}")
                    elif tool_name == "generate_bibliography":
                        keys = tool_input.get("bibtex_keys", [])
                        print(f"  Keys: {', '.join(keys[:5])}{'...' if len(keys) > 5 else ''}")

                # Call the tool
                try:
                    result = call_jiminy_tool(tool_name, tool_input)

                    if verbose:
                        # Show brief result info
                        if tool_name == "search_papers":
                            try:
                                result_data = json.loads(result)
                                if isinstance(result_data, dict) and "results" in result_data:
                                    print(f"  Found: {len(result_data['results'])} results")
                            except (json.JSONDecodeError, TypeError):
                                if "No results" in result or "0 results" in result:
                                    print("  Found: 0 results")
                                else:
                                    print("  Results retrieved")
                        print()

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

                except Exception as e:
                    error_msg = f"Error calling tool: {str(e)}"
                    if verbose:
                        print(f"  Error: {str(e)}\n")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": error_msg,
                        "is_error": True
                    })

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
            messages.append({"role": "user", "content": tool_results})
        else:
            if verbose:
                print(f"\n{'='*60}")
                print("Research Complete")
                print(f"{'='*60}\n")
            return final_text

    return "Error: Maximum iterations reached without completing the research."


def main():
    """Main entry point for the Jiminy research agent."""
    print("\n" + "=" * 60)
    print("  Jiminy Research Agent")
    print("  Answers questions using your curated paper database")
    print("  Powered by RunPod MCP Server")
    print("=" * 60 + "\n")

    # Check for API keys
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set your Anthropic API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)

    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY environment variable not set.")
        print("Please set your RunPod API key in .env:")
        print("  RUNPOD_API_KEY=your-runpod-api-key")
        sys.exit(1)

    if not RUNPOD_ENDPOINT_ID:
        print("Error: RUNPOD_ENDPOINT_ID environment variable not set.")
        print("Please set your RunPod endpoint ID in .env:")
        print("  RUNPOD_ENDPOINT_ID=bonn0doh0yb272")
        sys.exit(1)

    # Get research question from command line or prompt
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("Enter your research question:")
        print("(This will search your personal paper database)")
        print()
        question = input("> ").strip()

        if not question:
            print("No question provided. Exiting.")
            sys.exit(0)

    # Run the agent
    response = run_jiminy_agent(question)

    print("\n" + "=" * 60)
    print("  Answer")
    print("=" * 60 + "\n")
    print(response)


if __name__ == "__main__":
    main()
