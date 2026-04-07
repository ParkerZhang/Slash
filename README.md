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
# Start full TUI (Ink/React)
npm start

# Start Minimal TUI (Readline mode - no dependencies)
npm run start:minimal

# Run non-interactive command collection
npm run cli -- "/load f1 data.csv /diff f1 f2"
```

## Non-Interactive CLI (Command Collection)

You can run a sequence of commands without entering the TUI by using the `cli` script. This is ideal for CI/CD pipelines or batch processing.

### Example: Batch Load, Diff, and Export
```bash
npm run cli -- "/load req SecurityPricingRequest.csv /load resp SecurityPricingResponse.csv /diff req resp /cat req-diff-resp /save req-diff-resp fromNewCli.csv"
```

**Sample Output:**
```text
> Executing: /load req SecurityPricingRequest.csv
Loaded SecurityPricingRequest.csv as 'req' (59 rows) | Key field: isin

> Executing: /load resp SecurityPricingResponse.csv
Loaded SecurityPricingResponse.csv as 'resp' (52 rows) | Key field: isin

> Executing: /diff req resp
Created 'req-diff-resp' with 8 records

> Executing: /cat req-diff-resp
--- req-diff-resp (8/8 records) ---
isin|currency|exchange_code|country|name|sector|requested_date|status
BRPETRACNPR6|BRL|LATAM|BR|Petrobras PN|Energy|2024-12-16|pending
BRVALEACNOR0|BRL|LATAM|BR|VALE SA|Energy|2024-12-16|pending
BRITUBACNOR4|BRL|LATAM|BR|Itau Unibanco|Banking|2024-12-16|pending
MXP370711014|MXN|LATAM|MX|America Movil|Automotive|2024-12-16|pending
MXP001661018|MXN|LATAM|MX|Walmart de Mexico|Retail|2024-12-16|pending
CLP7847L1080|CLP|LATAM|CL|Enel Chile|Energy|2024-12-16|pending
CLP249051066|CLP|LATAM|CL|Banco de Chile|Other|2024-12-16|pending
ARBCOM460L12|ARS|LATAM|AR|Grupo Financiero Galicia|Banking|2024-12-16|pending

> Executing: /save req-diff-resp fromNewCli.csv
Saved req-diff-resp (8 records) to /home/u/tui-slash/data/fromNewCli.csv
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
| `/cat <id> [limit]` | Print file contents to terminal |
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

## View Modes

- **main** - Main input panel with command history
- **preview** - Preview records in a file
- **detail** - View full record details
- **nested** - Drill down into nested objects
- **compare** - Side-by-side comparison of two files
- **match** - Match results between request and response
- **keyfield** - Select key field for matching
- **schema** - Inspect the root schema and multi-select sub-schemas
