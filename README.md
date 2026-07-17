# Research Agents

Autonomous agents that search academic databases to answer research questions using the Anthropic MCP Connector.

## Overview

Four agents, each connecting to different academic databases:

| Program | Database | Coverage | Authentication |
|---------|----------|----------|----------------|
| `pubmed.py` | PubMed | 36M+ biomedical citations | None required |
| `scholar.py` | Scholar Gateway | 3M+ Wiley articles | OAuth required |
| `jiminy.py` | Paper RAG | Personal paper library | API key required |
| `research.py` | All three | Combined coverage | All keys recommended |

All agents:
- **Only use database results** - never rely on internal knowledge
- **Autonomously research** - make multiple tool calls to find relevant papers
- **Provide citations** - include PMIDs, DOIs, BibTeX keys, and full references

## Requirements

- Python 3.10+
- Anthropic API key
- Scholar Gateway token (for `scholar.py` only)
- Paper RAG API key (for `jiminy.py` only)

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

### Required for `jiminy.py` only

| Variable | Description |
|----------|-------------|
| `PAPERRAG_API_KEY` | API key for the Paper RAG MCP server |

### Required for `scholar.py` only

Two ways to authenticate — `scholar.py` prefers the first and falls back to the second:

| Variable | Description |
|----------|-------------|
| `SCHOLAR_GATEWAY_CLIENT_ID` | Client ID for the Wiley custom connector (preferred) |
| `SCHOLAR_GATEWAY_CLIENT_SECRET` | Client Secret for the Wiley custom connector (preferred) |
| `SCHOLAR_GATEWAY_TOKEN` | Legacy static OAuth token (fallback, optional) |

### Scholar Gateway authentication (client_credentials — preferred)

Wiley's custom connector supports the OAuth2 **client_credentials** grant. This is a
non-interactive, machine-to-machine flow: `scholar.py` exchanges your Client ID and
Client Secret for a short-lived access token on every run, so there is no browser
login and no token to manually refresh.

1. Add your credentials to `.env` (these are secrets — `.env` is git-ignored):
   ```
   SCHOLAR_GATEWAY_CLIENT_ID=your-client-id
   SCHOLAR_GATEWAY_CLIENT_SECRET=your-client-secret
   ```
2. Verify the OAuth step in isolation before running the agent:
   ```bash
   python scholar_auth.py
   ```
   A masked token is printed on success. The token exchange runs against
   `https://custom-connector-v1.scholargateway.ai/oauth2/token`, and the agent
   connects to `https://custom-connector-v1.scholargateway.ai/mcp`.

### Legacy: getting a Scholar Gateway token via MCP Inspector (fallback)

If you don't have client credentials, `scholar.py` falls back to a static
`SCHOLAR_GATEWAY_TOKEN`. Scholar Gateway also supports OAuth 2.1 via CONNECT SSO;
use the MCP Inspector to obtain an access token:

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

**Note:** These static tokens expire. You'll need to repeat this process when the
token expires — which is why the client_credentials flow above is preferred.

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

### Paper RAG Agent (Jiminy)

Search your personal academic paper library:

```bash
# With a question as argument
python jiminy.py "What papers discuss polymer nanoparticles for drug delivery?"

# Interactive mode
python jiminy.py
```

**Best for:**
- Questions about papers in your indexed library
- Finding connections between papers you've collected
- Generating bibliographies from your paper collection
- Semantic search across your personal research corpus

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

### Unified Research Agent

Search across all three databases simultaneously:

```bash
# With a question as argument
python research.py "What are the latest treatments for Type 2 diabetes?"

# Interactive mode
python research.py
```

**Best for:**
- Comprehensive literature searches across multiple sources
- Cross-referencing findings between databases
- When you're unsure which database has the most relevant content

**Note:** Requires `ANTHROPIC_API_KEY`. `PAPERRAG_API_KEY` and `SCHOLAR_GATEWAY_TOKEN` are recommended but optional — the agent will use whichever databases are configured.

## Example Questions

### PubMed (`pubmed.py`)
- "What are the recent findings on GLP-1 receptor agonists for weight loss?"
- "What is the current evidence for CRISPR gene therapy in sickle cell disease?"
- "What are the risk factors for long COVID?"

### Paper RAG (`jiminy.py`)
- "What papers discuss polymer nanoparticles for drug delivery?"
- "Find papers about machine learning in materials science"
- "What's in my library about CRISPR applications?"

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
         │              ┌─────────┼─────────┐
         │              ▼         ▼         ▼
         │      ┌──────────┐ ┌──────────┐ ┌───────────────────┐
         │      │  PubMed  │ │ Paper RAG│ │  Scholar Gateway  │
         │      └──────────┘ └──────────┘ └───────────────────┘
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

### Paper RAG Tools (`jiminy.py`)

| Tool | Description |
|------|-------------|
| `search_papers` | Semantic search through indexed papers |
| `get_paper_details` | Get full metadata for a specific paper |
| `database_stats` | Get statistics about the paper database |
| `list_recent_papers` | Show recently added papers |
| `generate_bibliography` | Create a .bib file from paper keys |

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

# Paper RAG (Jiminy)
from jiminy import run_paperrag_agent

response = run_paperrag_agent(
    "What papers discuss polymer nanoparticles for drug delivery?",
    verbose=True
)

# Scholar Gateway
from scholar import run_scholar_agent

response = run_scholar_agent(
    "What are recent advances in renewable energy storage?",
    verbose=True
)

# Unified (all databases)
from research import run_research_agent

response = run_research_agent(
    "What are the latest treatments for Type 2 diabetes?",
    verbose=True
)
```

## License

MIT
