# TUI-Slash V2.1 Architecture

V2.1 extends V2 with AI-powered clustering analysis. It uses embeddings and K-means clustering to discover dominant groups in data, while maintaining all V2 capabilities.

## Core Components

### 9. Clustering Module (`src/core/clustering.ts`)
Provides K-means clustering implementation with:
- Standard Euclidean distance clustering (`kMeans`)
- Cosine distance variant (`kMeansCosine`)
- Dominant term extraction for cluster interpretation
- Cluster summary formatting

### 10. Embedding Module (`src/ai/embedding.ts`)
Handles Ollama embedding generation:
- Text serialization of records
- Streaming embedding generation with progress callbacks
- Response parsing and validation

### 1. Core Types (`src/core/types.ts`)
Centralizes shared data structures for loaded files, schemas, view modes, command results, and the `Workspace` interface.

In V2, `LoadedFile` is not just raw data. It can also carry:
- inferred root schema
- extracted sub-schemas
- selected sub-schema paths

### 2. Command Registry (`src/commands/commandRegistry.ts`)
Provides the `SlashCommand` interface and the `CommandRegistry` class.

V2 change:
- registries are instance-based
- the TUI owns a registry instance
- `/refresh` rebuilds that registry at runtime

### 3. Command Implementations (`src/commands/index.ts`)
Contains all slash command implementations.

Key V2 commands:
- `/ai` for streaming schema inference and selected-only schema generation
- `/schema` for schema inspection and multi-select sub-schema workflow
- `/saveSchema` for schema export
- `/refresh` for on-the-fly command reload

### 4. Workspace Abstraction
Commands interact with the system via a `Workspace` interface, allowing for different implementations:
- **TuiWorkspace (in `src/tui/index.tsx`)**: Bridges commands to React state and TUI components.
- **MemoryWorkspace (in `src/core/runner.ts`)**: An in-memory implementation for CLI and test usage.

V2 adds a lightweight command output hook so long-running commands such as `/ai` can stream progress back into the TUI.

### 5. CLI Runner (`src/core/runner.ts`)
A non-interactive entry point that creates its own command registry and executes commands using a `MemoryWorkspace`.

### 6. TUI Implementation (`src/tui/index.tsx`)
A React-based Ink application that:
- stores loaded files and view state
- owns the active command registry
- can rebuild the registry with `/refresh`
- renders preview, compare, detail, nested, keyfield, and schema views
- displays streaming AI output and status while schema inference is running

### 7. Schema Helpers (`src/core/schema.ts`)
Provides schema extraction utilities:
- extract sub-schemas from a root JSON Schema
- build a selected-only schema object from the current selection set

### 8. AI Integration (`src/ai/modelManager.ts`)
Provides Ollama integration for:
- model discovery
- streaming model execution
- schema response parsing and validation

## Source Layout

- `src/tui/`: Ink application and interactive terminal UI
- `src/commands/`: Slash command registry and command implementations
- `src/core/`: Shared domain types, schema helpers, persistence, diff logic, clustering, and in-memory runner
- `src/ai/`: Ollama model discovery, AI-related helpers, and embedding generation
- `src/test/`: Vitest coverage for commands, clustering, and core logic

## Command Lifecycle

1. **Input**: User enters a slash command in the TUI or via the CLI runner.
2. **Dispatch**: The active `CommandRegistry` parses the command and resolves the handler.
3. **Execution**: The command executes against the `Workspace`.
4. **Streaming output**: Long-running commands can push intermediate output through the workspace output hook.
5. **Result**: The command returns a `CommandResult` with output and optional actions such as `VIEW_CHANGE`, `CLEAR`, `EXIT`, or `REFRESH`.
6. **UI reaction**: The TUI updates state, changes view, or rebuilds the command registry when `/refresh` is invoked.

## V2 Feature Flows

### AI Schema Flow
1. Load a file with `/load` or `/loadCsv`
2. Select a model with `/model`
3. Run `/ai <id>`
4. Stream model output in the TUI
5. Store root schema and extracted sub-schemas on the source file
6. Create a previewable `*-schema` file

### AI Compare Flow
1. Load request and response files
2. Set a model with `/model`
3. Run `/ai compare <f1> <f2> <question>`
4. Build prompt context from raw file payloads, key fields, and missing-record analysis
5. Stream the analysis in the TUI
6. Surface findings such as:
   pricing vendor coverage bias toward Tier 1 pools
   missing LATAM or regional exchange coverage
   response-only fields like `price` and `business_date`
   status transformation from `pending` to `active` or `inactive`
   request/response business date lag

### AI Clustering Analysis Flow
1. Load a file with `/load` or `/loadCsv`
2. Set a model with `/model` (must support embeddings)
3. Run `/ai analyze <fileId> "<question>"`
4. Extract number of clusters from question (defaults to 4)
5. **Generate name embeddings** via Ollama for each record
6. **Build multi-dimensional features**:
   - Name embeddings (semantic similarity, sector, brand)
   - Geographic features (one-hot encoded: country, exchange, currency)
   - Apply weights: 0.7 (name) + 0.3 (geo) by default
   - Normalize combined feature vectors
7. Run K-means clustering on multi-dimensional feature vectors
8. **Evaluate cluster quality**:
   - Geography purity (lower = better name-based clustering)
   - Name/sector purity
   - Balance score (cluster size equality)
   - Cross-region detection
9. **Build multi-dimensional profiles** for each cluster:
   - Sector breakdown (Tech, Banking, Energy, Automotive, Telecom)
   - Geographic breakdown
   - Exchange coverage
   - Cross-region presence
10. **Format mental model visualization**:
    - Shows sector symbols (▲ Tech, ■ Banking, ◆ Energy, ● Automotive)
    - Country distribution with dot notation (e.g., `US ●●●  JP ●●`)
11. Build AI analysis prompt with enhanced clustering context
12. Stream AI interpretation in the TUI
13. Store clustering result as new file with `_cluster` field

The clustering workflow automatically:
- Parses cluster count from the question (e.g., "top 4 clusters")
- Shows progress during embedding generation
- Provides multi-dimensional cluster summary with quality metrics
- Creates a new file `<fileId>-clustering-<k>` for further analysis
- Includes mental model visualization in output

**Key V2.1 Improvements:**
- **Multi-dimensional features**: Not just embeddings, but structured categorical encoding
- **Weight control**: Balances semantic similarity vs geographic grouping
- **Normalization**: Prevents any single feature group from dominating
- **Quality metrics**: Evaluates how well clusters separate by different dimensions
- **Mental model visualization**: Intuitive sector/country distribution display
- **Sector inference**: Automatically classifies companies into sectors from names
- **Cross-region detection**: Identifies clusters that span multiple geographies

### Schema Selection Flow
1. Open `/schema <id>`
2. Navigate sub-schemas with arrow keys
3. Toggle multiple sub-schemas with `Space` or `Enter`
4. Build a selected-only schema with `/ai <id> --selected-only`
5. Export with `/saveSchema <id> <path> [--selected-only]`

### Refresh Flow
1. Update command/core implementation on disk
2. Run `/refresh`
3. Re-import command module and rebuild registry
4. Continue using the running TUI without restarting

## Refresh Boundary

`/refresh` is designed for command-layer changes:
- command implementations
- command registration
- core helpers consumed by commands

It is not a full hot-reload system for the Ink component tree. Changes to `src/tui/index.tsx` may still require restarting the process.

## Testing
Commands are tested in isolation using `MemoryWorkspace` and `CommandRegistry`.

Current test coverage focuses on:
- command registration and dispatch
- AI schema command behavior
- selected-only schema generation
- schema export
- diff and SQL command behavior
