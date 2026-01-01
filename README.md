# PubMed Research Agent

An autonomous agent that searches PubMed to answer biomedical research questions using the Anthropic MCP Connector.

## Overview

This agent uses Claude's [MCP Connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector) to connect directly to Anthropic's PubMed MCP server. The agent:

- **Only uses PubMed results** - never relies on internal knowledge for biomedical facts
- **Autonomously searches** - makes multiple tool calls to find and analyze relevant papers
- **Provides citations** - includes PMIDs, DOIs, and full reference information

## Requirements

- Python 3.10+
- Anthropic API key

## Installation

1. Clone or download this repository

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Copy the example environment file and add your API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your Anthropic API key
   ```

## Usage

### Command Line

```bash
# With a question as argument
python main.py "What are the latest treatments for Type 2 diabetes?"

# Interactive mode
python main.py
```

### As a Module

```python
from main import run_pubmed_agent

response = run_pubmed_agent(
    "What is the evidence for mRNA vaccines in cancer treatment?",
    verbose=True
)
print(response)
```

## Example Questions

- "What are the recent findings on GLP-1 receptor agonists for weight loss?"
- "What is the current evidence for CRISPR gene therapy in sickle cell disease?"
- "What are the risk factors for long COVID?"
- "What treatments are effective for treatment-resistant depression?"

## How It Works

The agent uses the Anthropic Messages API with the MCP Connector beta feature:

1. **MCP Server Connection**: Connects to `https://pubmed.mcp.claude.com/mcp`
2. **Tool Discovery**: The PubMed MCP server provides tools like `search_articles`, `get_article_metadata`, `find_related_articles`, etc.
3. **Agentic Loop**: Claude autonomously calls tools until it has enough information to answer
4. **Citation Synthesis**: The final response includes proper academic citations

## Available PubMed Tools

The MCP server provides these tools (discovered automatically):

| Tool | Description |
|------|-------------|
| `search_articles` | Search PubMed with keywords, authors, MeSH terms |
| `get_article_metadata` | Get full abstracts, authors, MeSH terms for PMIDs |
| `find_related_articles` | Find similar articles or linked resources |
| `get_full_text_article` | Retrieve full text from PubMed Central |
| `convert_article_ids` | Convert between PMID, PMCID, and DOI |
| `lookup_article_by_citation` | Find articles by citation details |

## Configuration

Environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |

## License

MIT
