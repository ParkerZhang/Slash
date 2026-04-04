# TUI Slash Commands V2.1

V2.1 is a terminal-first data workspace for loading ElasticSearch JSON and CSV data, running slash commands, generating schemas with AI, **clustering analysis with embeddings**, and refreshing command logic without restarting the TUI.

## V2.1 Features

- **AI clustering analysis**: `/cluster <fileId> [k] [--Dominance|--Spreading]` uses local embeddings and K-means to discover dominant groups
- **AI schema generation**: `/ai <id>` uses the current Ollama model to infer JSON Schema from loaded data
- **AI pattern analysis**: `/ai analyze <fileId> "<question>"` clusters data and provides AI interpretation
- **Refresh on the fly**: `/refresh` rebuilds the command registry so command/core changes can be picked up without restarting the app
- **Schema workflow**: `/schema <id>` opens an interactive schema view, supports multi-select sub-schemas, and `/saveSchema` exports full or selected schemas
- **AI compare**: `/ai compare <f1> <f2> <question>` can explain what is missing, detect vendor coverage gaps, and summarize structural patterns
- **Terminal-native data operations**: preview, compare, match, diff, minus, sort, SQL, and save commands all stay available in the same workflow

## Screenshots

![V2.1 Clustering](docs/V2.1%20Clustering.png)

![V2.1 AI Analyze](docs/V2.1%20AI%20Analyze.png)

![Main View](docs/Screenshot%20from%202026-04-03%2014-37-37.png)

![Latest V2 View](docs/Screenshot%20from%202026-04-03%2021-00-53.png)

![Latest V2 View 2](docs/Screenshot%20from%202026-04-03%2021-00-22.png)

## Project Structure

```
tui-slash/
├── data/          # Data files (CSV, JSON) - default load directory
├── docs/          # Documentation, screenshots, assets
├── modelFiles/    # Local embedding models (all-MiniLM-L6-v2)
├── scripts/       # Helper scripts (download-model.js)
├── src/           # Source code
│   ├── ai/        # Ollama integration & embedding helpers
│   ├── commands/  # Slash commands & command registry
│   ├── core/      # Shared types, schema helpers, clustering, diff logic
│   ├── test/      # Vitest coverage for commands and core logic
│   ├── train-ai/  # AI training workflow (providers, prompts, persistence)
│   └── tui/       # Ink UI and interactive view state
├── bin/           # CLI entry point
├── .cli_history.json  # Persistent command history & loaded files
├── .ai-provider.json  # AI provider configuration (ollama, llama-cpp)
├── .train-ai/     # Workflow state & cache
└── package.json
```

## V2.1 Architecture Summary

- **`src/tui/`** - Ink UI with banner cycling, output scrolling, haiku banners
- **`src/commands/`** - Slash commands including `/cluster`, `/ai analyze`, `/ai train`, `/aiProvider`
- **`src/core/`** - Clustering (K-means), local embeddings (all-MiniLM-L6-v2), diff logic, types
- **`src/ai/`** - Ollama integration and streaming schema inference helpers
- **`src/train-ai/`** - AI training workflow with provider management, guided prompts, persistence
- **`modelFiles/`** - Local embedding model cache (~23MB, auto-downloaded)

## Tech Stack

- TypeScript
- Ink (TUI framework)
- React

## Installation

```bash
npm install
```

## Prerequisites

- Node.js 18+
- npm 9+
- Ollama installed and available on `PATH` for AI features
- At least one local model pulled if you want to use `/model` and `/ai`

## Usage

```bash
npm start
# Or via CLI command:
tui-slash
```

Basic V2 flow:

```bash
/model gemma4:latest
/load f1 data/elasticSearchResult.json
/ai f1
/schema f1
/ai f1 --selected-only
/saveSchema f1 /tmp/f1-schema.json
/refresh
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/refresh` | Rebuild command modules inside the running TUI |
| `/model [name]` | Show or set the current Ollama model, with saved selection |
| `/ai <id> [--selected-only] [instructions]` | Infer a schema from the model, or build a schema file from the current sub-schema selections |
| `/ai analyze <fileId> "<question>"` | Run AI clustering analysis to find dominant groups in your data |
| `/ai compare <f1> <f2> <question>` | Compare two files with AI-powered pattern analysis |
| `/schema <id>` | Show the stored schema for a file and interactively multi-select sub-schemas |
| `/saveSchema <id> <filepath> [--selected-only]` | Save the full schema or only the selected sub-schemas to disk |
| `/load <id> <file>` | Load JSON file (default: `./data/`) |
| `/loadCsv <id> <file> [sep]` | Load CSV file (default separator: `\|`, default dir: `./data/`) |
| `/preview <id>` | Preview loaded file |
| `/files` | List loaded files |
| `/compare <id1> <id2>` | Compare two files side by side |
| `/match <reqId> <respId>` | Match request vs response by key fields |
| `/keyfield <id>` | Set key field for matching |
| `/sort <id> <field> [asc\|desc]` | Sort file by field (default: asc) |
| `/minus <id1> <id2>` | List records in id1 but not in id2 (uses keyfields) |
| `/diff <id1> <id2>` | Diff files - records in f1 missing from f2 |
| `/sql "select f1.* from f1 where f1.isin not in (select isin from f2)"` | SQL-like query |
| `/save <id> <filepath>` | Save file to disk |
| `/clear` | Clear command history |
| `/exit` | Exit application |

## Navigation

- `↑` / `↓` - Navigate history or records
- `Enter` - Select, expand, or toggle in interactive views
- `Space` - Toggle sub-schema selection in schema view
- `Esc` - Go back to previous view
- `Tab` - Cycle through command/file suggestions

## AI And Schema Workflow

```bash
# Set model and infer schema
/model gemma4:latest
/load f1 data/elasticSearchResult.json
/ai f1

# Inspect schema and choose only the parts you want
/schema f1

# Build a previewable selected-only schema file
/ai f1 --selected-only

# Export schema artifacts
/saveSchema f1 /tmp/f1-full-schema.json
/saveSchema f1 /tmp/f1-selected-schema.json --selected-only

# Reload command/core changes without restarting the TUI
/refresh
```

## AI Clustering Analysis

Discover dominant groups and patterns in your data using **local embeddings** and K-means:

### Quick Start
```bash
# Load your data
/load f1 data/clustering.csv

# Cluster with default 4 groups (Dominance mode)
/cluster f1

# Specify cluster count
/cluster f1 6

# Force specific field
/cluster f1 4 --key country

# Switch to Spreading mode
/cluster f1 4 --Spreading
```

### Output Example
```
📊 Clustering Results (dominance)
Field: country | Records: 58 | Clusters: 4
──────────────────────────────────────────────────
Banking ■
  BR ●●●●
  MX ●●
  CL ●
  AR ●

Tech ▲
  US ●●●●●
  HK ●●●●

──────────────────────────────────────────────────
Saved as: f1-clustering-4
```

**Options:**
- `--Dominance` (default): Find concentrated fields where few values dominate
- `--Spreading`: Find evenly distributed fields (high entropy)
- `--key <field>`: Force clustering on specific field
- `-k <field>`: Shorthand for --key

## /ai analyze Workflow

Full AI-powered analysis with clustering:

```bash
# 1. Load your data
/load missing data/SecurityPricingRequest.csv

# 2. Run AI analysis with natural language question
/ai analyze missing "what are the dominant groups, top 4 clusters"
```

**How it works:**
1. **Local Embeddings**: Records embedded using all-MiniLM-L6-v2 (no Ollama needed)
2. **Multi-Dimensional Features**: Combines semantic + geographic features
3. **K-Means Clustering**: Groups similar records automatically
4. **AI Interpretation**: Your selected model analyzes patterns and provides insights

**Mental Model Visualization:**
```
Sector Symbol
  COUNTRY ●●●●    COUNTRY ●●    COUNTRY ●
```

- Each cluster shows dominant sector with symbol
- Country distribution shown with dots (each ● = 1 record)
- Easy to spot geographic concentration or spread

## AI Compare Example

Prompt:

```bash
/ai compare f1 f2 "what <f1> is missing in <f2>, and what the pattern is"
```

Example V2 findings the AI can surface:

- Suggests pricing vendor coverage is limited to Tier 1 liquidity pools.
- LATAM and certain regional exchanges may be missing from the response.
- Response adds fields like `price` and `business_date`.
- Response transforms `status` from `pending` into `active` or `inactive`.
- Response date can lag the request date, for example `2024-12-15` vs `2024-12-16`.

## View Modes

- **main** - Main input panel with command history
- **preview** - Preview records in a file
- **detail** - View full record details
- **nested** - Drill down into nested objects
- **compare** - Side-by-side comparison of two files
- **match** - Match results between request and response
- **keyfield** - Select key field for matching
- **schema** - Inspect the root schema and multi-select sub-schemas

## File Path Resolution

Files are resolved in this order:
1. Absolute paths (e.g., `/home/user/data/file.csv`)
2. Paths starting with `data/` or `./data/`
3. Relative to `./data/` directory (default)
