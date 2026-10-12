# Research Agents

Autonomous agents that search academic databases to answer research questions using MCP servers (via the Anthropic MCP Connector, and client-side for Scholar Gateway).

## Overview

Four agents, each connecting to different academic databases:

| Program | Database | Coverage | Authentication |
|---------|----------|----------|----------------|
| `pubmed.py` | PubMed | 36M+ biomedical citations | None required |
| `scholar.py` | Scholar Gateway | 3M+ Wiley articles | API key required |
| `jiminy.py` | Paper RAG | Personal paper library | API key required |
| `research.py` | All three | Combined coverage | All keys recommended |

All agents:
- **Only use database results** - never rely on internal knowledge
- **Autonomously research** - make multiple tool calls to find relevant papers
- **Provide citations** - include PMIDs, DOIs, BibTeX keys, and full references

## Requirements

- Python 3.10+
- Anthropic API key
- Scholar Gateway API key (for `scholar.py` only)
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

| Variable | Description |
|----------|-------------|
| `SCHOLAR_GATEWAY_API_KEY` | API key issued by Wiley for the v2 custom connector |

### Scholar Gateway authentication

Wiley's v2 custom connector (`https://custom-connector-v2.scholargateway.ai/mcp`,
Streamable HTTP only) authenticates with a static API key sent in an `X-API-Key`
header on every request. There is no OAuth token exchange and nothing to refresh.

1. Add the key to `.env` (it is a secret — `.env` is git-ignored; never commit it):
   ```
   SCHOLAR_GATEWAY_API_KEY=your-api-key
   ```
2. Verify connectivity before running the agent:
   ```bash
   python scholar_auth.py
   ```
   This runs an MCP handshake against the endpoint and lists the available
   tools. The key is shown masked.

**Note:** Don't paste the key into hosted, third-party MCP inspector websites. If
you want a GUI, run the official inspector locally with
`npx @modelcontextprotocol/inspector`.

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

**Note:** Requires `ANTHROPIC_API_KEY`. `PAPERRAG_API_KEY` and `SCHOLAR_GATEWAY_API_KEY` are recommended but optional — the agent will use whichever databases are configured.

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
         │              ┌─────────┴─────────┐
         │              ▼                   ▼
         │      ┌──────────┐         ┌──────────┐
         │      │  PubMed  │         │ Paper RAG│
         │      └──────────┘         └──────────┘
         │
         │  client-side tool calls (X-API-Key)
         ├─────────────────────────▶┌───────────────────┐
         │                          │  Scholar Gateway  │
         │                          └───────────────────┘
         ▼
┌─────────────────┐
│ Cited Answer    │
└─────────────────┘
```

1. **MCP Server Connection**: PubMed and Paper RAG are reached through
   Anthropic's server-side MCP connector. Scholar Gateway's v2 endpoint requires
   an `X-API-Key` header, which the connector can't send, so `scholar_auth.py`
   talks to it directly and the agent executes its tool calls locally
2. **Tool Discovery**: Servers provide search and metadata tools
3. **Agentic Loop**: Claude autonomously calls tools until it has enough information
4. **Citation Synthesis**: Final response includes proper academic citations

### Paper RAG cold starts and warm-up

The Paper RAG server (used by `jiminy.py` and `research.py`) runs on **AWS App
Runner, which scales to zero when idle**. The first request after an idle period
— or right after a deploy — triggers a **cold start** while the instance loads
the embedding model and vector database. This can take up to a couple of minutes.

If a cold start happens *during* an agent turn, the connector's first tool call
can stall long enough to exceed the Anthropic client's request timeout, which
previously surfaced as a hang or an errored first search. Two safeguards address
this:

- **Warm-up before the turn.** `jiminy.py` and `research.py` call
  `warm_up_paperrag()` (in `paperrag_warmup.py`) before invoking the agent. It
  sends a direct, tiny `search_papers` request — with its own generous 300s
  timeout — that forces the server to finish cold-starting, so the actual agent
  turn runs against a warm instance. You'll see `Warming up Paper RAG server...`
  followed by `Paper RAG server ready (N.Ns)`; on a warm server this is ~1s, on a
  cold start it reflects the real spin-up time. The warm-up is best-effort — if it
  can't complete it prints a warning and proceeds.
- **Bounded client timeout.** All agents construct the Anthropic client with
  `timeout=120.0, max_retries=1`, so any residual stall fails fast and visibly
  instead of hanging on the SDK defaults (~10 min with silent retries).

To avoid cold starts entirely, configure the App Runner service with a minimum
of **1 instance** (keeps one warm) rather than scaling to zero.

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
| `search_wiley_fulltext` | Semantic search over the full text of Wiley articles; returns passages with citation metadata |

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
