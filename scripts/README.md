# Chat Script — Financial NLP with Local Embeddings

Interactive REPL for tokenization exploration, embedding similarity, and financial data enrichment using a local ONNX model.

## Setup

```bash
npm install
npm run chat
```

The MIC model (`all-MiniLM-L6-v2-mic`) with custom ticker tokens auto-loads if present under `scripts/modelFiles/`; otherwise falls back to the base model.

## Key Commands

### Tokenization

```
tokens ^GSPC       Shows WP tokenization (single token if in MIC vocab)
trace AAPL         Full pipeline: normalize → pre-tokenize → WordPiece
```

### Embedding Similarity

```
dist AAPL TSLA     Cosine distance between two tokens
nn NVDA 5          Nearest neighbors from baseline set
show META          Dim activations + nearest neighbors
diff AAPL MSFT     Compare embedding dimensions side-by-side
```

### Identifier Database

```
ids                List all saved identifiers (tickers, names, exchanges)
ids save AAPL      Add a ticker to the database
ids tokens         Show tokens eligible for model addition
ids add-model      Add all DB tickers to the MIC tokenizer vocab (detection only, maps to [UNK])
ids embed-model    Add all DB tickers with ONNX embedding resize (full token support)
```

Stored in `scripts/identifiers.json`.

### Financial Data

```
quote ^GSPC        Live price from Yahoo Finance
chart TSLA 1mo     Unicode sparkline with H/L labels
news               Recent financial headlines from Yahoo RSS
```

### Content Scanning

```
scan <url>         Extract tickers from an article (data-ylk, /quote/ links, vocab match)
                   Auto-saves unknown tickers enriched via Yahoo search API
page <url>         Fetch article, detect tickers, colorize with live quotes
page <file>        Same analysis on a local file
page sample.csv    Resolve sample listings through MIC/root/ISIN links and quote matches
page samples.csv   Alias for sample.csv
page all           Analyze all cached news articles
```

### Model Selection (overrides auto-detect)

```
--mic <cmd>        Expand MIC codes then run cmd
--mic-model <cmd>  Force MIC model for command
--orig-model <cmd> Force original model for command
```

## Full Pipeline

Run these steps in order to rebuild the model, embed identifiers, build the index, verify, and start the chat:

```bash
# 1. Rebuild MIC model from scratch
python3 rebuild_mic_model.py

# 2. Sync registry (merges sample.csv + identifiers.json → registry.json)
python3 sync_registry.py

# 3. Embed all registry identifiers (ISINs, SEDOLs, tickers) into the model
python3 embed_registry.py

# 4. Pre-compute full-transformer embeddings for all tokens
python3 build_embed_index.py

# 5. Post-build checks
python3 connect_identifiers_sample.py
python3 test_mic_model.py
python3 verify_accuracy.py

# 6. Launch interactive chat
npm run chat
```

Useful smoke tests in the REPL:

```text
page identifiers.json
page sample.csv
quote TSLA
family classify TSLA
family classify TLO.GY
```

Expected connector behavior includes:

```text
TSLA.AV    -> TSLA (root_ticker)
TLO.GY     -> TSLA (isin)
TSLA.UW    -> TSLA (exact_yahoo_ticker)
```

## Architecture

- **Local model**: ONNX quantized, 384-dim, base at `modelFiles/Xenova/all-MiniLM-L6-v2/`, MIC at `scripts/modelFiles/Xenova/all-MiniLM-L6-v2-mic/`
- **Custom tokens**: ~191 ticker symbols added to tokenizer vocab via `scripts/add_mic_tokens.py`
- **Index tickers**: `^GSPC`, `^DJI`, `^IXIC`, `^TNX` — added as vocab tokens (plus bare ticker variants for detection)
- **HTTP**: Pure Node `http`/`https` (no `fetch`/`undici` — avoids Yahoo's oversized header bug)
- **Proxy**: Respects `HTTP_PROXY`/`HTTPS_PROXY` env vars
