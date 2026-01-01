# Research Agents

Autonomous agents that search academic databases to answer research questions using the Anthropic MCP Connector.

## Overview

Two standalone research agents, each connecting to a different academic database:

| Program | Database | Coverage | Authentication |
|---------|----------|----------|----------------|
| `pubmed.py` | PubMed | 36M+ biomedical citations | None required |
| `scholar.py` | Scholar Gateway | 3M+ Wiley articles | OAuth required |

Both agents:
- **Only use database results** - never rely on internal knowledge
- **Autonomously research** - make multiple tool calls to find relevant papers
- **Provide citations** - include PMIDs, DOIs, and full references

## Requirements

- Python 3.10+
- Anthropic API key
- Scholar Gateway token (for `scholar.py` only)

## Installation

1. Clone or download this repository

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env and add your keys
   ```

## Configuration

### Required for both programs

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key ([get one here](https://console.anthropic.com/)) |

### Required for `scholar.py` only

| Variable | Description |
|----------|-------------|
| `SCHOLAR_GATEWAY_TOKEN` | OAuth token for Scholar Gateway |

### Getting a Scholar Gateway Token

Scholar Gateway uses OAuth 2.1 via CONNECT SSO. Use the MCP Inspector to obtain an access token:

```bash
npx @modelcontextprotocol/inspector
```

1. For "Transport type", select "SSE" or "Streamable HTTP"
2. Enter URL: `https://connector.scholargateway.ai/mcp`
3. Click "Open Auth Settings" after "Need to configure authentication?"
4. Click "Quick OAuth Flow" and authorize via CONNECT SSO
5. Follow the steps until "Authentication complete"
6. Copy the `access_token` value from the result
7. Paste it as the value of `SCHOLAR_GATEWAY_TOKEN` in your `.env` file

**Note:** OAuth tokens expire. You'll need to repeat this process when your token expires.

## Usage

### PubMed Agent

Search biomedical literature:

```bash
# With a question as argument
python pubmed.py "What are the latest treatments for Type 2 diabetes?"

# Interactive mode
python pubmed.py
```

**Best for:**
- Biomedical and clinical research
- Life sciences literature
- Medical treatment evidence
- Drug mechanisms and pharmacology

### Scholar Gateway Agent

Search Wiley's academic articles:

```bash
# With a question as argument
python scholar.py "What are recent advances in organic solar cells?"

# Interactive mode
python scholar.py
```

**Best for:**
- Broader scientific topics
- Wiley journal content
- Interdisciplinary research
- Social sciences and humanities

## Example Questions

### PubMed (`pubmed.py`)
- "What are the recent findings on GLP-1 receptor agonists for weight loss?"
- "What is the current evidence for CRISPR gene therapy in sickle cell disease?"
- "What are the risk factors for long COVID?"

### Scholar Gateway (`scholar.py`)
- "What are recent advances in organic solar cell efficiency?"
- "What does the research say about remote work productivity?"
- "What are the environmental impacts of microplastics?"

## How It Works

```
┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────────────────────┐
│  Claude Agent   │────▶│  MCP Connector (Anthropic API)   │
└────────┬────────┘     └──────────────────────────────────┘
         │                        │
         │              ┌─────────┴─────────┐
         │              ▼                   ▼
         │      ┌───────────────┐   ┌───────────────────┐
         │      │    PubMed     │   │  Scholar Gateway  │
         │      │  (pubmed.py)  │   │   (scholar.py)    │
         │      └───────────────┘   └───────────────────┘
         │
         ▼
┌─────────────────┐
│ Cited Answer    │
└─────────────────┘
```

1. **MCP Server Connection**: Each agent connects to its respective MCP server
2. **Tool Discovery**: Servers provide search and metadata tools
3. **Agentic Loop**: Claude autonomously calls tools until it has enough information
4. **Citation Synthesis**: Final response includes proper academic citations

## Available Tools

### PubMed Tools (`pubmed.py`)

| Tool | Description |
|------|-------------|
| `search_articles` | Search with keywords, authors, MeSH terms |
| `get_article_metadata` | Get full abstracts, authors, MeSH terms |
| `find_related_articles` | Find similar articles or linked resources |
| `get_full_text_article` | Retrieve full text from PubMed Central |
| `convert_article_ids` | Convert between PMID, PMCID, and DOI |
| `lookup_article_by_citation` | Find articles by citation details |

### Scholar Gateway Tools (`scholar.py`)

| Tool | Description |
|------|-------------|
| `semantic_search` | Semantic search across Wiley's article database |

## As a Module

```python
# PubMed
from pubmed import run_pubmed_agent

response = run_pubmed_agent(
    "What is the evidence for mRNA vaccines in cancer treatment?",
    verbose=True
)

# Scholar Gateway
from scholar import run_scholar_agent

response = run_scholar_agent(
    "What are recent advances in renewable energy storage?",
    verbose=True
)
```

## License

MIT
