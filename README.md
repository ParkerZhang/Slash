# TUI Slash Commands

A TUI (Text User Interface) application for viewing and comparing ElasticSearch JSON and CSV files with slash commands.

## Screenshots

![Main View](docs/Screenshot%20from%202026-04-03%2014-37-37.png)

![Preview View](docs/Screenshot%20from%202026-04-03%2014-37-53.png)

![Compare View](docs/Screenshot%20from%202026-04-03%2014-40-24.png)

## Project Structure

```
tui-slash/
├── data/          # Data files (CSV, JSON) - default load directory
├── docs/          # Documentation, screenshots, assets
├── src/           # Source code
├── bin/           # CLI entry point
├── .cli_history.json  # Persistent command history & loaded files
└── package.json
```

## Tech Stack

- TypeScript
- Ink (TUI framework)
- React

## Installation

```bash
npm install
```

## Usage

```bash
npm start
# Or via CLI command:
tui-slash
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
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
- `Enter` - Select/expand details
- `Esc` - Go back to previous view
- `Tab` - Cycle through command/file suggestions

## Example Usage

```bash
# Load CSV files (from ./data/ by default)
/loadCsv f1 SecurityPricingRequest.csv
/loadCsv f2 SecurityPricingResponse.csv
/keyfield f1
/keyfield f2

# Preview loaded file
/preview f1

# Match request with response
/match f1 f2

# Find differences
/diff f1 f2

# SQL-like query
/sql "select f1.* from f1 where f1.isin not in (select isin from f2)"
```

## View Modes

- **main** - Main input panel with command history
- **preview** - Preview records in a file
- **detail** - View full record details
- **nested** - Drill down into nested objects
- **compare** - Side-by-side comparison of two files
- **match** - Match results between request and response
- **keyfield** - Select key field for matching

## File Path Resolution

Files are resolved in this order:
1. Absolute paths (e.g., `/home/user/data/file.csv`)
2. Paths starting with `data/` or `./data/`
3. Relative to `./data/` directory (default)
