# Project instructions

## Project and architecture

This repository contains Python command-line research agents. `README.md` is the
reference for installation, configuration, usage, and current operational behavior.

- `pubmed.py` searches PubMed; `jiminy.py` searches Paper RAG; `scholar.py`
  searches Scholar Gateway; `research.py` combines the three sources.
- Research answers must use retrieved database results and include academic
  citations (PMIDs, DOIs, or BibTeX keys where available). When evidence is
  insufficient, say so rather than supplying research facts from model knowledge.
- PubMed and Paper RAG use Anthropic's server-side MCP connector. Scholar Gateway
  uses the client-side `ScholarGatewayClient` in `scholar_auth.py`, with a static
  `SCHOLAR_GATEWAY_API_KEY` sent as `X-API-Key` to the v2 Streamable HTTP endpoint.
  Preserve this authentication contract; it does not use OAuth or token refresh.
- Keep the shared Paper RAG URL and best-effort warm-up in `paperrag_warmup.py`.
  Both `jiminy.py` and `research.py` warm the server before the agent turn. The
  warm-up has its own 300-second timeout and proceeds with a warning on failure.
- Preserve bounded Anthropic client settings (`timeout=120.0, max_retries=1`)
  and Unicode console output support.
- The unified agent requires `ANTHROPIC_API_KEY` and uses the optional databases
  when their keys are configured.

## Development and validation

The application requires Python 3.10+ and dependencies in `requirements.txt`.
Use the project virtual environment; the system Python may be too old:

```bash
python3 -m venv .venv  # Use a Python 3.10+ interpreter.
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s scripts -p test_sync_agent_config.py
.venv/bin/python scripts/sync_agent_config.py --check
git diff --check
```

There is no configured application test suite, formatter, or linter. For Python
changes, check syntax and run focused tests appropriate to the behavior changed.
The research entrypoints and `scholar_auth.py` connectivity check call external
services and may incur API costs; they are not offline validation commands.

Keep credentials in the ignored `.env` file. Never commit keys, print their full
values, or paste them into hosted third-party inspectors. Document environment
variables using placeholders in `.env.example` and update the README when usage
or operational behavior changes.

## Agent configuration

`AGENTS.md` is the authoritative project-instruction file. Keep `CLAUDE.md` as
the exact `@AGENTS.md` compatibility import with a trailing newline. Author future
repository development skills in `.agents/skills/`, then run
`.venv/bin/python scripts/sync_agent_config.py --write` to refresh
`.claude/skills/`. Run `--check` before committing. Do not edit generated mirrors
directly. With no repository skills, skill directories and the manifest are
omitted.

The sync script and its tests are unchanged copies from `geppetto-ecosystem`.
Make tooling fixes in that master with regression coverage, then copy both files
here; do not modify the local copies independently.

## Git and deployment

Preserve unrelated local changes and ignored local configuration. Leave changes
uncommitted and unpushed unless the user explicitly requests otherwise.

The default branch is `main`. The agent configuration workflow validates pull
requests and pushes to `main`; it does not deploy. There are no repository-defined
deployment or publishing workflows. Inspect current workflows and any known
external integrations before reporting the deployment consequences of a push.
