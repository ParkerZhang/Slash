#!/usr/bin/env node

import * as path from 'path';
import { fileURLToPath } from 'url';
import { env, pipeline, AutoTokenizer } from '@huggingface/transformers';
import * as readline from 'readline';
import * as fs from 'fs';

// HTTP helper using built-in modules (bypasses undici header limits)
import * as http from 'http';
import * as https from 'https';
import * as cp from 'child_process';

function httpGet(url, timeout = 15000, maxRedirects = 3) {
  return new Promise((resolve, reject) => {
    const mod = url.startsWith('https') ? https : http;
    const proxy = process.env.HTTPS_PROXY || process.env.HTTP_PROXY || process.env.https_proxy || process.env.http_proxy;
    const opts = {
      method: 'GET',
      headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' },
      timeout,
    };
    if (proxy) {
      const p = new URL(proxy);
      opts.hostname = p.hostname;
      opts.port = p.port;
      opts.path = url;
      opts.headers['Host'] = new URL(url).hostname;
    } else {
      const u = new URL(url);
      opts.hostname = u.hostname;
      opts.port = u.port;
      opts.path = u.pathname + u.search;
    }
    const req = mod.request(opts, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location && maxRedirects > 0) {
        const redirect = new URL(res.headers.location, url).href;
        resolve(httpGet(redirect, timeout, maxRedirects - 1));
        return;
      }
      const chunks = [];
      res.on('data', d => chunks.push(d));
      res.on('end', () => resolve(Buffer.concat(chunks).toString()));
    });
    req.on('error', reject);
    req.end();
  });
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, 'modelFiles');
const lastNews = [];

const GRN = '\x1b[32m', YLW = '\x1b[33m', RED = '\x1b[31m', RST = '\x1b[0m';

let micVocab = null;
function getMicVocab() {
  if (micVocab) return micVocab;
  try {
    const tokPath = path.join(MIC_MODEL, 'tokenizer.json');
    const tok = JSON.parse(fs.readFileSync(tokPath, 'utf-8'));
    micVocab = new Set(Object.keys(tok.model?.vocab || {}));
    return micVocab;
  } catch { return null; }
}

async function scanUrl(url) {
  const html = await httpGet(url);
  const text = html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
  const found = new Set();
  const knownTickers = Object.keys(loadIdentifiers());
  for (const m of html.matchAll(/data-ylk="[^"]*slk:([A-Za-z0-9^]+);/g)) {
    found.add(m[1].toUpperCase());
  }
  for (const m of html.matchAll(/\/quote\/([A-Za-z0-9^]+)\//g)) {
    found.add(m[1].toUpperCase());
  }
  const words = new Set(text.toUpperCase().split(/\s+/).filter(w => w.length >= 1));
  for (const t of knownTickers) {
    if (words.has(t)) found.add(t);
  }
  // Also check MIC model's vocab: any word that's a single token is a known entity
  const vocab = getMicVocab();
  if (vocab) {
    for (const w of words) {
      if (/^[A-Z]{1,5}$/.test(w) && vocab.has(w) && !knownTickers.includes(w)) {
        found.add(w);
      }
    }
  }
  return [...found].sort();
}

async function enrichTickers(tickers) {
  const enriched = {};
  for (const t of tickers) {
    try {
      const body = await httpGet(`https://query1.finance.yahoo.com/v1/finance/search?q=${encodeURIComponent(t)}`);
      const data = JSON.parse(body);
      const match = data.quotes?.find(q => q.symbol?.toUpperCase() === t);
      enriched[t] = match ? {
        valid: true,
        name: match.shortname || match.longname || t,
        type: (match.quoteType || '?').toLowerCase(),
        exchange: match.exchange || '?',
        longname: match.longname
      } : { valid: false };
    } catch {
      enriched[t] = { valid: false };
    }
  }
  return enriched;
}

async function processScanResults(tickers) {
  console.log(`Potential identifiers: ${tickers.length}`);
  console.log('Verifying via Yahoo Finance...');
  const enriched = await enrichTickers(tickers);
  const db = loadIdentifiers();
  let saved = 0;
  for (const t of tickers) {
    const e = enriched[t];
    if (e?.valid) {
      const prev = db[t] || {};
      db[t] = {
        name: e.name,
        type: e.type,
        exchange: e.exchange,
        longname: e.longname || prev.longname,
        source: prev.source || 'scan-auto'
      };
      saved++;
    }
  }
  saveIdentifiers(db);
  console.log(`\nResults (${saved} verified, ${tickers.length - saved} unknown):`);
  for (const t of tickers) {
    const e = enriched[t];
    const color = e?.valid ? GRN : YLW;
    const info = e?.valid ? `${e.name.padEnd(35)} [${e.type}] @${e.exchange}` : '(not found on Yahoo)';
    console.log(`  ${color}${t.padEnd(8)} ${info}${RST}`);
  }
}
env.cacheDir = MODEL_DIR;

const ORIG_MODEL = 'Xenova/all-MiniLM-L6-v2';
const MIC_MODEL = path.resolve(MODEL_DIR, 'Xenova/all-MiniLM-L6-v2-mic');

const MIC_MAP = {
  XNYS: 'New York Stock Exchange',
  XNAS: 'NASDAQ Stock Exchange',
  XLON: 'London Stock Exchange',
  XTKS: 'Tokyo Stock Exchange',
  XHKG: 'Hong Kong Stock Exchange',
  XPAR: 'Euronext Paris',
  XSHG: 'Shanghai Stock Exchange',
  XTSE: 'Toronto Stock Exchange',
  XASX: 'Australian Securities Exchange',
  XBSP: 'Brazil Stock Exchange',
  XFRA: 'Frankfurt Stock Exchange',
  XSWX: 'Swiss Exchange',
  XNZE: 'New Zealand Exchange',
  XSGO: 'Santiago Stock Exchange',
  XBOM: 'BSE India',
  XNSE: 'National Stock Exchange India',
  XKOS: 'Korea Exchange',
  XSES: 'Singapore Exchange',
  XTAI: 'Taiwan Stock Exchange',
  XWBO: 'Vienna Stock Exchange',
};

function resolveMIC(text) {
  return text.replace(/\b[A-Z]{4}\b/g, m => MIC_MAP[m] || m);
}

const IDENT_PATH = path.join(__dirname, 'identifiers.json');
const IDENT_ISIN_PATH = path.join(__dirname, 'identifiers_isin.json');
const SAMPLE_PATH = path.join(__dirname, 'sample.csv');

const YAHOO_SUFFIX = {
  XNAS: '',
  XNGS: '',
  XNYS: '',
  XASE: '',
  XETR: '.DE',
  XFRA: '.F',
  XLON: '.L',
  XTSE: '.TO',
  XWBO: '.VI',
  BVMF: '.SA',
  XHKG: '.HK',
  XTKS: '.T',
};

function loadIdentifiers() {
  try {
    return JSON.parse(fs.readFileSync(IDENT_PATH, 'utf-8'));
  } catch {
    return {};
  }
}

function saveIdentifiers(data) {
  fs.writeFileSync(IDENT_PATH, JSON.stringify(data, null, 2));
}

function loadJsonFile(filePath, fallback = {}) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch {
    return fallback;
  }
}

function splitCsvLine(line) {
  const cols = [];
  let cur = '';
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        quoted = !quoted;
      }
      continue;
    }
    if (ch === ',' && !quoted) {
      cols.push(cur);
      cur = '';
      continue;
    }
    cur += ch;
  }
  cols.push(cur);
  return cols;
}

function tickerRoot(ticker) {
  return (ticker || '').trim().toUpperCase().split(/[._/]/)[0];
}

function resolveYahooTicker(ticker, mic) {
  return `${tickerRoot(ticker)}${YAHOO_SUFFIX[(mic || '').trim().toUpperCase()] ?? ''}`;
}

function loadSampleIdentifierAliases() {
  if (!fs.existsSync(SAMPLE_PATH)) return new Map();

  const identifiers = loadIdentifiers();
  const identifierIsins = loadJsonFile(IDENT_ISIN_PATH, {});
  const exact = new Set(Object.keys(identifiers).map(t => t.toUpperCase()));
  const roots = new Map();
  for (const ticker of Object.keys(identifiers)) {
    const root = tickerRoot(ticker);
    if (!roots.has(root)) roots.set(root, ticker.toUpperCase());
  }
  const byIsin = new Map();
  for (const [ticker, isin] of Object.entries(identifierIsins)) {
    if (isin) byIsin.set(String(isin).trim().toUpperCase(), ticker.toUpperCase());
  }

  const lines = fs.readFileSync(SAMPLE_PATH, 'utf-8').trim().split(/\r?\n/);
  const header = splitCsvLine(lines.shift() || '').map(h => h.trim().toUpperCase());
  const idx = name => header.indexOf(name);
  const tickerIdx = idx('TICKER');
  const micIdx = idx('MIC');
  const isinIdx = idx('ISIN');
  if (tickerIdx < 0 || micIdx < 0 || isinIdx < 0) return new Map();

  const aliases = new Map();
  for (const line of lines) {
    const cols = splitCsvLine(line);
    const sampleTicker = (cols[tickerIdx] || '').trim().toUpperCase();
    if (!sampleTicker) continue;
    const mic = (cols[micIdx] || '').trim().toUpperCase();
    const isin = (cols[isinIdx] || '').trim().toUpperCase();
    const yahooTicker = resolveYahooTicker(sampleTicker, mic).toUpperCase();
    const root = tickerRoot(sampleTicker);
    const quoteTicker = exact.has(yahooTicker) ? yahooTicker
      : roots.get(root)
      || byIsin.get(isin)
      || yahooTicker;
    aliases.set(sampleTicker, quoteTicker);
  }
  return aliases;
}

const BASELINE_WORDS = [
  'apple', 'banana', 'car', 'dog', 'elephant', 'forest', 'garden', 'house',
  'ice', 'jungle', 'kite', 'lion', 'mountain', 'night', 'ocean', 'piano',
  'queen', 'river', 'sun', 'tree', 'umbrella', 'valley', 'water', 'yellow',
  'zero', 'angel', 'bridge', 'cloud', 'dream', 'earth', 'flame', 'glass',
  'heart', 'island', 'jewel', 'king', 'lake', 'music', 'north', 'orange',
  'peace', 'rain', 'stone', 'tower', 'unity', 'voice', 'wind', 'wave',
  'book', 'code', 'data', 'energy', 'fire', 'gold', 'hope', 'image',
  'light', 'metal', 'number', 'oak', 'plant', 'quest', 'road', 'salt',
  'time', 'use', 'view', 'wall', 'year', 'zen', 'art', 'bird',
  'color', 'door', 'east', 'faith', 'grace', 'hill', 'idea', 'joy',
  'key', 'life', 'mind', 'name', 'open', 'path', 'rest', 'sand',
];

let extractor = null;
let tokenizer = null;
let loadedModelId = null;
let loadingPromise = null;
const embedCache = {};

// Full-vocab embedding index (loaded for MIC model)
let vocabIndex = null;        // Float32Array [N, 384]
let vocabIndexTokens = null;  // string[]
let vocabIndexN = 0;

// Raw ONNX embedding weight (for family commands — 4 vacuum dims)
let embedWeight = null;        // Float32Array [vocab_size, 384]
let embedWeightTokens = null;  // string[]
let embedWeightN = 0;
let embedWeightDim = 384;

// Family 4-d vacuum vectors (loaded from family_vac.json)
let familyVac = null;   // {token: [d18, d62, d28, d245]}
let familyVacDims = [18, 62, 28, 245];

async function loadVocabIndex(modelDir) {
  const binPath = path.join(modelDir, 'vocab_embed.bin');
  const tokPath = path.join(modelDir, 'vocab_embed_tokens.json');
  if (!fs.existsSync(binPath) || !fs.existsSync(tokPath)) {
    console.log('  (no vocab index — run scripts/build_embed_index.py)');
    return;
  }
  const buf = fs.readFileSync(binPath);
  const floatArr = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const dim = 384;
  vocabIndex = floatArr;
  vocabIndexN = floatArr.length / dim;
  vocabIndexTokens = JSON.parse(fs.readFileSync(tokPath, 'utf-8'));
  console.log(`  vocab index: ${vocabIndexN} embeddings`);

  // Load raw embedding weight for family commands
  const ewPath = path.join(modelDir, 'embed_weight.bin');
  const ewTokPath = path.join(modelDir, 'embed_weight_tokens.json');
  const fvPath = path.join(modelDir, 'family_vac.json');
  if (fs.existsSync(ewPath) && fs.existsSync(fvPath)) {
    const ewBuf = fs.readFileSync(ewPath);
    embedWeight = new Float32Array(ewBuf.buffer, ewBuf.byteOffset, ewBuf.byteLength / 4);
    embedWeightTokens = JSON.parse(fs.readFileSync(ewTokPath, 'utf-8'));
    embedWeightN = embedWeight.length / 384;
    familyVac = JSON.parse(fs.readFileSync(fvPath, 'utf-8'));
    console.log(`  embed weight: ${embedWeightN}×384 (raw ONNX)`);
    console.log(`  family tokens: ${Object.keys(familyVac).length}`);
  }
}

/** Look up a token's 4-d vacuum vector from the raw embedding weight. */
function getVacVector(token) {
  // First check family_vac.json (pre-computed for family tokens)
  if (familyVac) {
    if (familyVac[token]) return familyVac[token];
    if (familyVac[token.toLowerCase()]) return familyVac[token.toLowerCase()];
    if (familyVac[token.toUpperCase()]) return familyVac[token.toUpperCase()];
  }
  // Fallback: look up in embedWeight
  if (!embedWeight || !embedWeightTokens) return null;
  let idx = embedWeightTokens.indexOf(token);
  if (idx < 0) idx = embedWeightTokens.indexOf(token.toLowerCase());
  if (idx < 0) idx = embedWeightTokens.indexOf(token.toUpperCase());
  
  if (idx < 0) return null;
  const v = [];
  for (const d of [18, 62, 28, 245]) {
    v.push(embedWeight[idx * 384 + d]);
  }
  return v;
}

/** Cosine similarity between two 4-d vacuum vectors. */
function vacCos(a, b) {
  let dot = 0, na2 = 0, nb2 = 0;
  for (let i = 0; i < 4; i++) {
    dot += a[i] * b[i];
    na2 += a[i] * a[i];
    nb2 += b[i] * b[i];
  }
  const na = Math.sqrt(na2), nb = Math.sqrt(nb2);
  if (na < 1e-8 || nb < 1e-8) return 0;
  return dot / (na * nb);
}

/** Search full vocab index for nearest neighbors. Returns [{token, sim}]. */
function searchIndex(queryEmb, topK = 20) {
  if (!vocabIndex) return [];
  const dim = 384;
  const scores = [];
  for (let i = 0; i < vocabIndexN; i++) {
    let dot = 0;
    const offset = i * dim;
    for (let j = 0; j < dim; j++) {
      dot += queryEmb[j] * vocabIndex[offset + j];
    }
    scores.push({ idx: i, sim: dot, token: vocabIndexTokens[i].toUpperCase() });
  }
  scores.sort((a, b) => b.sim - a.sim);
  return scores.slice(0, topK);
}

async function loadModel(modelId) {
  if (loadedModelId === modelId && extractor) return;
  if (loadingPromise) return loadingPromise;
  loadingPromise = (async () => {
    process.stdout.write(`Loading ${path.basename(modelId)}... `);
    extractor = null;
    tokenizer = null;
    loadedModelId = null;
    tokenizer = await AutoTokenizer.from_pretrained(modelId);
    extractor = await pipeline('feature-extraction', modelId, {
      quantized: true,
    });
    loadedModelId = modelId;
    process.stdout.write('ready.\n');
    // Load full-vocab index if available
    if (modelId === MIC_MODEL) {
      await loadVocabIndex(modelId);
    }
  })();
  try {
    await loadingPromise;
  } finally {
    loadingPromise = null;
  }
}

async function embed(modelId, text) {
  await loadModel(modelId);
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

async function getTokenizer(modelId) {
  await loadModel(modelId);
  return tokenizer;
}

function cosineSimilarity(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

function cosineDistance(a, b) {
  return 1 - cosineSimilarity(a, b);
}

function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}

const baselineCache = {};

async function buildBaseline(modelId) {
  if (baselineCache[modelId]) return baselineCache[modelId];
  process.stdout.write('Computing baseline statistics... ');
  const embs = await Promise.all(BASELINE_WORDS.map(w => embed(modelId, w)));
  const dists = [];
  for (let i = 0; i < embs.length; i++) {
    for (let j = i + 1; j < embs.length; j++) {
      dists.push(cosineDistance(embs[i], embs[j]));
    }
  }
  const mean = dists.reduce((a, b) => a + b, 0) / dists.length;
  const variance = dists.reduce((a, b) => a + (b - mean) ** 2, 0) / dists.length;
  const std = Math.sqrt(variance);
  baselineCache[modelId] = { mean, std };
  process.stdout.write(`done.\n`);
  process.stdout.write(`Baseline: mean=${mean.toFixed(4)}, std=${std.toFixed(4)} (${BASELINE_WORDS.length} words, ${dists.length} pairs)\n\n`);
  return baselineCache[modelId];
}

function fmtDist(raw, mean, std) {
  const z = std > 0 ? ((raw - mean) / std).toFixed(2) : 'N/A';
  return `${raw.toFixed(4)} (z=${z})`;
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: 'embed> ',
});

async function main() {
  let modelId = fs.existsSync(path.join(MIC_MODEL, 'tokenizer.json')) ? MIC_MODEL : ORIG_MODEL;

  async function getEmbedding(text) {
    if (embedCache[modelId]?.[text]) return embedCache[modelId][text];
    const emb = await embed(modelId, text);
    if (!embedCache[modelId]) embedCache[modelId] = {};
    embedCache[modelId][text] = emb;
    return emb;
  }

  console.log('Commands:');
  console.log('  <word|phrase>              Show embedding info');
  console.log('  dist <a> <b>               Cosine distance between two words');
  console.log('  sim <a> <b>                Cosine similarity between two words');
  console.log('  euc <a> <b>                Euclidean distance');
  console.log('  nn <word> [n]              N nearest neighbors from baseline words');
  console.log('  family nn <word> [n]       N nearest neighbors among family tokens');
  console.log('  family dist <a> <b>        Distance between two family tokens');
  console.log('  family classify <word>     Closest marker type among {#MIC,ISIN,SEDOL,TICKER}');
  console.log('  stats <word>               Show distances to all baseline words');
  console.log('  show <word>                Explore embedding shape (dims, neighbors)');
  console.log('  diff <a> <b>               Show how two embeddings differ dimension-wise');
  console.log('  tokens <text>              Show tokenization result');
  console.log('  trace <text>               Trace full tokenization pipeline');
  console.log('  clear                      Clear embedding cache');
  console.log('  --mic <cmd> ...            Expand MIC codes before processing');
  console.log('  --mic-model <cmd> ...      Use MIC model (with new tokens)');
  console.log('  --orig-model <cmd> ...     Switch back to original model');
  console.log('  ids [filter]               List/search saved identifiers');
  console.log('  ids save <TICKER> <name>   Save a new identifier');
  console.log('  ids del <TICKER>           Delete an identifier');
  console.log('  quote <ticker>             Fetch current price from Yahoo');
  console.log('  listings <ticker>          Show all exchange listings for a company');
  console.log('  chart <ticker> [range] [interval]  Text chart (range: 1d/5d/1mo/1y, interval: 1m/5m/1h/1d)');
  console.log('  news [topic]               Find recent financial news articles');
  console.log('  scan <url>                 Extract identifiers from an article');
  console.log('  page <url>                 Show article with tickers colorized + quotes');
  console.log('  exit / quit                Exit\n');
  console.log('(Model loads on first command)\n');

  rl.prompt();

  rl.on('line', async (line) => {
    const trimmed = line.trim();
    if (!trimmed) { rl.prompt(); return; }

    let parts = trimmed.split(/\s+/);
    let useMIC = false;
    while (parts[0] === '--mic' || parts[0] === '--mic-model' || parts[0] === '--orig-model') {
      if (parts[0] === '--mic') { useMIC = true; modelId = MIC_MODEL; }
      if (parts[0] === '--mic-model') modelId = MIC_MODEL;
      if (parts[0] === '--orig-model') modelId = ORIG_MODEL;
      parts = parts.slice(1);
    }
    const cmd = parts[0].toLowerCase();
    const expand = s => useMIC ? resolveMIC(s) : s;

    try {
      if (cmd === 'exit' || cmd === 'quit') {
        rl.close();
        return;
      }

      if (cmd === 'tokens' || cmd === 'tokenize') {
        const text = expand(parts.slice(1).join(' '));
        if (!text) { console.log('Usage: tokens <text>'); rl.prompt(); return; }
        if (useMIC) console.log(`(expanded: "${text}")`);
        const tok = await getTokenizer(modelId);
        const encoded = await tok(text);
        const ids = Array.from(encoded.input_ids.data);
        const tokens = tok.tokenize(text);
        const allTokens = ['[CLS]', ...tokens, '[SEP]'];
        const special = new Set(tok.all_special_tokens);
        console.log(`"${text}" → ${ids.length} tokens\n`);
        console.log('  #   │    id │ token');
        console.log('  ───┼───────┼────────────────');
        for (let i = 0; i < ids.length; i++) {
          const flag = special.has(allTokens[i]) ? ' *' : '  ';
          console.log(`  ${String(i).padStart(2)} │ ${String(ids[i]).padStart(5)} │ ${allTokens[i]}${flag}`);
        }
          console.log('\n  * = special token');
        rl.prompt();
        return;
      }

      if (cmd === 'tokenize-url' || cmd === 'page' || cmd === 'tu') {
        const arg = parts.slice(1).join(' ');
        const getPage = async (url) => {
          console.log(`Fetching ${url}...`);
          const html = await httpGet(url);
          const text = html.replace(/<script[\s\S]*?<\/script>/gi, ' ').replace(/<style[\s\S]*?<\/style>/gi, ' ').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
          await processPage(text);
        };
        const processPage = async (text) => {
          const words = text.split(/\s+/).filter(w => w);
          const cleanWords = words.map(w => w.replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, '')).filter(w => w);
          const knownIdentifiers = new Set(Object.keys(loadIdentifiers()).map(t => t.toUpperCase()));
          const quoteTargets = new Map();
          for (const w of cleanWords) {
            if (/^[A-Z]{2,5}$/.test(w) && knownIdentifiers.has(w.toUpperCase())) {
              quoteTargets.set(w.toUpperCase(), w.toUpperCase());
            }
          }
          for (const w of words) {
            const m = w.match(/\^[A-Z]{1,5}/);
            if (m) quoteTargets.set(m[0], m[0]);
          }

          const sampleAliases = loadSampleIdentifierAliases();
          for (const [sampleTicker, quoteTicker] of sampleAliases.entries()) {
            const escaped = sampleTicker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            if (new RegExp(`\\b${escaped}\\b`).test(text)) {
              quoteTargets.set(sampleTicker, quoteTicker);
            }
          }

          const tickers = [...quoteTargets.keys()].sort();
          const quoteTickers = [...new Set([...quoteTargets.values()])].sort();
          console.log(`Found ${tickers.length} tickers: ${tickers.map(t => quoteTargets.get(t) === t ? t : `${t}->${quoteTargets.get(t)}`).join(', ')}`);

          console.log('Fetching quotes...');
          const quoteResults = await Promise.all(quoteTickers.map(async t => {
            try {
              const body = await httpGet(`https://query1.finance.yahoo.com/v8/finance/chart/${t}?range=1d&interval=1d`);
              const data = JSON.parse(body);
              const r = data.chart.result[0];
              const meta = r.meta;
              const val = (v, f = 0) => { const x = Array.isArray(v) ? v[0] : v; return x ?? f; };
              const price = val(meta.regularMarketPrice);
              const prev = val(meta.previousClose);
              const change = prev ? price - prev : 0;
              const pct = prev ? (change / prev) * 100 : 0;
              const arrow = change >= 0 ? '▲' : '▼';
              const color = change >= 0 ? GRN : '\x1b[31m';
              return { t, price, change, pct, arrow, color };
            } catch { return { t, error: true }; }
          }));

          const tickerInfo = {};
          for (const q of quoteResults) if (!q.error) tickerInfo[q.t] = q;

          let result = text;
          const sortedTickers = [...tickers].sort((a, b) => b.length - a.length);
          for (const t of sortedTickers) {
            const quoteTicker = quoteTargets.get(t) || t;
            const q = tickerInfo[quoteTicker];
            const label = quoteTicker === t ? t : `${t}->${quoteTicker}`;
            const replacement = q ? `${q.color}${label}${RST} ${q.arrow}$${q.price.toFixed(2)} ${q.change >= 0 ? '+' : ''}${q.change.toFixed(2)} (${q.pct >= 0 ? '+' : ''}${q.pct.toFixed(2)}%)` : `${YLW}${label}${RST}`;
            const escaped = t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            if (t.startsWith('^')) {
              result = result.replace(new RegExp(`(^|[^\\w])(${escaped})([^\\w]|$)`, 'g'), (_, b, __, a) => b + replacement + a);
            } else {
              result = result.replace(new RegExp(`\\b${escaped}\\b`, 'g'), replacement);
            }
          }

          console.log(`\n--- Full text ---`);
          console.log(result);
        };

        if (arg === 'all') {
          if (lastNews.length === 0) { console.log('No articles cached. Run news first.'); rl.prompt(); return; }
          (async () => {
            for (const [i, a] of lastNews.entries()) {
              console.log(`\n--- [${i + 1}] ${a.title} ---`);
              try { await getPage(a.link); } catch (e) { console.log(`Error: ${e.message}`); }
            }
            rl.prompt();
          })();
          return;
        }
        const n = parseInt(arg, 10);
        if (!isNaN(n)) {
          if (lastNews.length === 0) { console.log('No articles cached. Run news first.'); rl.prompt(); return; }
          const a = lastNews[n - 1];
          if (!a) { console.log(`Article ${n} not found (1-${lastNews.length})`); rl.prompt(); return; }
          (async () => { try { await getPage(a.link); } catch (e) { console.log(`Error: ${e.message}`); } rl.prompt(); })();
          return;
        }
        if (!arg.startsWith('http')) {
          let filePath = path.resolve(__dirname, arg);
          if (!fs.existsSync(filePath) && path.basename(arg) === 'samples.csv') {
            filePath = path.resolve(__dirname, 'sample.csv');
          }
          if (fs.existsSync(filePath)) {
            const fileContent = fs.readFileSync(filePath, 'utf-8');
            (async () => { try { await processPage(fileContent); } catch (e) { console.log(`Error: ${e.message}`); } rl.prompt(); })();
          } else {
            console.log('Usage: page <url> or page <number> or page all or page <filepath>');
            rl.prompt();
          }
          return;
        }
        (async () => {
          try {
            console.log(`Fetching ${arg}...`);
            const html = await httpGet(arg);
            const text = html.replace(/<script[\s\S]*?<\/script>/gi, ' ').replace(/<style[\s\S]*?<\/style>/gi, ' ').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
            await processPage(text);
          } catch (e) { console.log(`Error: ${e.message}`); }
          rl.prompt();
        })();
        return;
      }

      if (cmd === 'trace' || cmd === 'trace-tokens') {
        const text = expand(parts.slice(1).join(' '));
        if (!text) { console.log('Usage: trace <text>'); rl.prompt(); return; }
        if (useMIC) console.log(`(expanded: "${text}")`);
        const tok = await getTokenizer(modelId);
        const m = tok.model;
        const vocab = {};
        for (const [idStr, token] of Object.entries(m.vocab)) {
          vocab[token] = parseInt(idStr);
        }
        const unkId = vocab['[UNK]'];

        function wordpiece(word) {
          if (word.length > (m.config?.max_input_chars_per_word || 100))
            return [{token: '[UNK]', id: unkId}];
          const chars = [...word];
          const pieces = [];
          let start = 0;
          while (start < chars.length) {
            let end = chars.length;
            let found = false;
            while (end > start) {
              let subword = chars.slice(start, end).join('');
              if (start > 0) subword = '##' + subword;
              if (vocab[subword] !== undefined) {
                pieces.push({token: subword, id: vocab[subword]});
                found = true;
                break;
              }
              end--;
            }
            if (!found) {
              pieces.push({token: '[UNK]', id: unkId});
              break;
            }
            start = end;
          }
          return pieces;
        }

        console.log(`\n  Input: "${text}"`);
        const normalized = tok.normalizer.normalize(text);
        console.log(`  Step 1 — Normalize: "${normalized}"`);
        const pre = tok.pre_tokenizer.pre_tokenize(normalized);
        console.log(`  Step 2 — Pre-tokenize: ${pre.map(w => '"' + w + '"').join(', ')}`);
        console.log(`  Step 3 — WordPiece:`);
        for (const word of pre) {
          const pieces = wordpiece(word);
          const subwords = pieces.map(p => p.token).join(' + ');
          const ids = pieces.map(p => p.id).join(', ');
          console.log(`    "${word}" → ${subwords}`);
          console.log(`      ids: [${ids}]`);
        }
        const allPieces = pre.flatMap(w => wordpiece(w));
        const allTokens = ['[CLS]', ...allPieces.map(p => p.token), '[SEP]'];
        const allIds = [tok.cls_token_id, ...allPieces.map(p => p.id), tok.sep_token_id];
        console.log(`  Step 4 — Post-process (add [CLS]/[SEP]):`);
        for (let i = 0; i < allTokens.length; i++) {
          const flag = tok.all_special_ids.includes(allIds[i]) ? ' *' : '  ';
          console.log(`    [${String(i).padStart(2)}] ${String(allIds[i]).padStart(6)}  ${allTokens[i]}${flag}`);
        }
        console.log('');
        rl.prompt();
        return;
      }

      if (cmd === 'clear') {
        if (embedCache[modelId]) embedCache[modelId] = {};
        console.log('Cache cleared.');
        rl.prompt();
        return;
      }

      const bl = await buildBaseline(modelId);

      if (cmd === 'dist' && parts.length >= 3) {
        const wordA = expand(parts[1]).toLowerCase();
        const wordB = expand(parts.slice(2).join(' ')).toLowerCase();
        if (useMIC) console.log(`(expanded: "${wordA}" / "${wordB}")`);
        const [embA, embB] = await Promise.all([getEmbedding(wordA), getEmbedding(wordB)]);
        const d = cosineDistance(embA, embB);
        console.log(`cosine distance: ${fmtDist(d, bl.mean, bl.std)}`);
        rl.prompt();
        return;
      }

      if (cmd === 'sim' && parts.length >= 3) {
        const wordA = expand(parts[1]).toLowerCase();
        const wordB = expand(parts.slice(2).join(' ')).toLowerCase();
        if (useMIC) console.log(`(expanded: "${wordA}" / "${wordB}")`);
        const [embA, embB] = await Promise.all([getEmbedding(wordA), getEmbedding(wordB)]);
        const s = cosineSimilarity(embA, embB);
        console.log(`cosine similarity: ${s.toFixed(4)}`);
        rl.prompt();
        return;
      }

      if (cmd === 'euc' && parts.length >= 3) {
        const wordA = expand(parts[1]).toLowerCase();
        const wordB = expand(parts.slice(2).join(' ')).toLowerCase();
        if (useMIC) console.log(`(expanded: "${wordA}" / "${wordB}")`);
        const [embA, embB] = await Promise.all([getEmbedding(wordA), getEmbedding(wordB)]);
        const d = euclideanDistance(embA, embB);
        console.log(`euclidean distance: ${d.toFixed(4)}`);
        rl.prompt();
        return;
      }

      if (cmd === 'news') {
        const topic = parts.slice(1).join(' ').toLowerCase();
        console.log(`Fetching news${topic ? ' about "' + topic + '"' : ''}...`);
        try {
          const xml = await httpGet('https://finance.yahoo.com/news/rss');
          const items = [...xml.matchAll(/<item>\s*<title>(.*?)<\/title>\s*<link>(.*?)<\/link>[\s\S]*?<source[^>]*>(.*?)<\/source>/g)];
          const unescape = s => s.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'");
          let results = items.map(it => ({ title: unescape(it[1]), link: unescape(it[2]), source: unescape(it[3]) }));
          if (topic) results = results.filter(r => r.title.toLowerCase().includes(topic) || r.source.toLowerCase().includes(topic));
          console.log(`Articles (${results.length}):`);
          for (let i = 0; i < Math.min(results.length, 8); i++) {
            const r = results[i];
            console.log(`  [${i + 1}] ${r.title}`);
            console.log(`      ${r.source} · ${r.link}`);
          }
          lastNews.length = 0;
          lastNews.push(...results);
          console.log(`\nOpen article: scan <url> or scan <1-${Math.min(results.length, 8)}>`);
        } catch (e) {
          console.log(`Error: ${e.message}`);
        }
        rl.prompt();
        return;
      }

      if (cmd === 'scan') {
        const arg = parts.slice(1).join(' ');
        if (arg === 'all') {
          if (lastNews.length === 0) { console.log('No articles cached. Run news first.'); rl.prompt(); return; }
          (async () => {
            const allTickers = new Set();
            for (const [i, a] of lastNews.entries()) {
              console.log(`Scanning [${i + 1}] ${a.title}...`);
              try {
                const tickers = await scanUrl(a.link);
                for (const t of tickers) allTickers.add(t);
              } catch (e) {
                console.log(`  Error: ${e.message}`);
              }
            }
            const consolidated = [...allTickers].sort();
            console.log(`\n=== Consolidated ${consolidated.length} unique potential identifiers ===`);
            await processScanResults(consolidated);
            rl.prompt();
          })();
          return;
        }
        const n = parseInt(arg, 10);
        if (!isNaN(n)) {
          if (lastNews.length === 0) { console.log('No articles cached. Run news first.'); rl.prompt(); return; }
          const a = lastNews[n - 1];
          if (!a) { console.log(`Article ${n} not found (1-${lastNews.length})`); rl.prompt(); return; }
          (async () => {
            try {
              const tickers = await scanUrl(a.link);
              await processScanResults(tickers);
            } catch (e) { console.log(`Error: ${e.message}`); }
            rl.prompt();
          })();
          return;
        }
        if (!arg.startsWith('http')) { console.log('Usage: scan <url> or scan <number> or scan all'); rl.prompt(); return; }
        (async () => {
          try {
            const tickers = await scanUrl(arg);
            await processScanResults(tickers);
          } catch (e) { console.log(`Error: ${e.message}`); }
          rl.prompt();
        })();
        return;
      }

      if (cmd === 'family') {
        const sub = parts[1]?.toLowerCase();
        if (sub === 'nn') {
          const word = expand(parts[2]).toLowerCase();
          const n = parts[3] ? parseInt(parts[3], 10) : 10;
          if (!word) { console.log('Usage: family nn <word> [n]'); rl.prompt(); return; }
          if (!familyVac) { console.log('No family index (family_vac.json) loaded.'); rl.prompt(); return; }
          const queryVac = getVacVector(word);
          if (!queryVac) { console.log(`"${word}" not in family tokens.`); rl.prompt(); return; }
          const scores = [];
          for (const [token, tokenVac] of Object.entries(familyVac)) {
            if (token === word) continue;
            const sim = vacCos(queryVac, tokenVac);
            scores.push({ token, sim });
          }
          scores.sort((a, b) => b.sim - a.sim);
          console.log(`Top ${n} family neighbors of "${word.toUpperCase()}" (4 vacuum dims):`);
          for (let i = 0; i < Math.min(n, scores.length); i++) {
            console.log(`  ${i + 1}. ${scores[i].token.toUpperCase()}  sim=${scores[i].sim.toFixed(4)}`);
          }
          rl.prompt();
          return;
        }
        if (sub === 'dist') {
          const a = expand(parts[2]).toLowerCase();
          const b = expand(parts[3]).toLowerCase();
          if (!a || !b) { console.log('Usage: family dist <tokenA> <tokenB>'); rl.prompt(); return; }
          const va = getVacVector(a);
          const vb = getVacVector(b);
          if (!va) { console.log(`"${a}" not found.`); rl.prompt(); return; }
          if (!vb) { console.log(`"${b}" not found.`); rl.prompt(); return; }
          const sim = vacCos(va, vb);
          const dist = 1 - sim;
          console.log(`"${a}" ↔ "${b}" (4 vacuum dims):`);
          console.log(`  cosine similarity: ${sim.toFixed(4)}`);
          console.log(`  cosine distance:   ${dist.toFixed(4)}`);
          console.log(`  ${a} vac: ${va.map(v => v.toFixed(4))}`);
          console.log(`  ${b} vac: ${vb.map(v => v.toFixed(4))}`);
          rl.prompt();
          return;
        }
        if (sub === 'classify') {
          const word = expand(parts[2]).toLowerCase();
          if (!word) { console.log('Usage: family classify <word>'); rl.prompt(); return; }
          const va = getVacVector(word);
          if (!va) { console.log(`"${word}" not in family index.`); rl.prompt(); return; }
          const markers = ['#MIC', 'ISIN', 'SEDOL', 'TICKER'];
          const scores = [];
          for (const m of markers) {
            const vm = getVacVector(m);
            if (!vm) continue;
            scores.push({ marker: m, sim: vacCos(va, vm) });
          }
          scores.sort((a, b) => b.sim - a.sim);
          console.log(`"${word}" family classification (4 vacuum dims):`);
          scores.forEach(s => console.log(`  ${s.marker}  sim=${s.sim.toFixed(4)}`));
          console.log(`→ ${scores[0].marker}`);
          rl.prompt();
          return;
        }
        console.log('Usage: family nn|dist|classify ...');
        rl.prompt();
        return;
      }

      if (cmd === 'nn') {
        const word = expand(parts[1]).toLowerCase();
        const n = parts[2] ? parseInt(parts[2], 10) : 10;
        if (!word) { console.log('Usage: nn <word> [n]'); rl.prompt(); return; }
        if (useMIC) console.log(`(expanded: "${word}")`);
        const emb = await getEmbedding(word);

        if (vocabIndex) {
          // Full-vocab search through pre-computed transformer embeddings
          const results = searchIndex(emb, n);
          console.log(`Top ${n} nearest neighbors of "${word}" (vocab index, ${vocabIndexN} tokens):`);
          for (let i = 0; i < results.length; i++) {
            console.log(`  ${i + 1}. ${results[i].token}  sim=${results[i].sim.toFixed(4)}`);
          }
        } else {
          // Fallback: baseline 88 words
          const blEmbs = await Promise.all(BASELINE_WORDS.map(w => getEmbedding(w)));
          const sims = BASELINE_WORDS.map((w, i) => ({ word: w, sim: cosineSimilarity(emb, blEmbs[i]) }));
          sims.sort((a, b) => b.sim - a.sim);
          console.log(`Top ${n} nearest neighbors of "${word.toUpperCase()}" (baseline, ${BASELINE_WORDS.length} words):`);
          for (let i = 0; i < Math.min(n, sims.length); i++) {
            const d = 1 - sims[i].sim;
            console.log(`  ${i + 1}. ${sims[i].word.toUpperCase()}  sim=${sims[i].sim.toFixed(4)}  dist=${fmtDist(d, bl.mean, bl.std)}`);
          }
        }
        rl.prompt();
        return;
      }

      if (cmd === 'stats') {
        const word = expand(parts[1]);
        if (!word) { console.log('Usage: stats <word>'); rl.prompt(); return; }
        if (useMIC) console.log(`(expanded: "${word}")`);
        const emb = await getEmbedding(word);
        const blEmbs = await Promise.all(BASELINE_WORDS.map(w => getEmbedding(w)));
        const dists = BASELINE_WORDS.map((w, i) => ({ word: w, dist: cosineDistance(emb, blEmbs[i]) }));
        dists.sort((a, b) => a.dist - b.dist);
        console.log(`Distance stats for "${word}" vs ${BASELINE_WORDS.length} baseline words:`);
        console.log(`  closest:  ${dists[0].word} (${dists[0].dist.toFixed(4)})`);
        console.log(`  farthest: ${dists[dists.length - 1].word} (${dists[dists.length - 1].dist.toFixed(4)})`);
        const avg = dists.reduce((s, d) => s + d.dist, 0) / dists.length;
        const z = bl.std > 0 ? ((avg - bl.mean) / bl.std).toFixed(2) : 'N/A';
        console.log(`  avg dist: ${avg.toFixed(4)} (z=${z})`);
        rl.prompt();
        return;
      }

      if (cmd === 'ids') {
        const db = loadIdentifiers();
        const sub = parts[1];
        if (sub === 'save' && parts.length >= 3) {
          const ticker = parts[2].toUpperCase();
          const name = parts.slice(3).join(' ') || ticker;
          db[ticker] = db[ticker] || {};
          db[ticker].name = name;
          db[ticker].source = 'manual';
          saveIdentifiers(db);
          console.log(`Saved ${ticker}: ${name}`);
        } else if (sub === 'add-model') {
          const tokens = Object.keys(db).filter(t => (/^[A-Z]{1,5}$/.test(t) || /^\^[A-Z]{1,5}$/.test(t)) && !db[t].type?.startsWith('^'));
          if (tokens.length === 0) { console.log('No suitable tokens to add.'); rl.prompt(); return; }
          console.log(`Adding ${tokens.length} tokens to MIC model...`);
          const tmpFile = '/tmp/opencode_new_tokens.json';
          fs.writeFileSync(tmpFile, JSON.stringify(tokens));
          try {
            const out = cp.execSync(`python3 ${path.resolve(__dirname, 'add_mic_tokens.py')} ${tmpFile}`, { encoding: 'utf8', timeout: 120000 });
            console.log(out);
          } catch (e) {
            console.log(`Error: ${e.stderr || e.message}`);
          }
        } else if (sub === 'del' && parts[2]) {
          delete db[parts[2].toUpperCase()];
          saveIdentifiers(db);
          console.log(`Deleted ${parts[2].toUpperCase()}`);
        } else if (sub === 'tokens') {
          const tokens = Object.keys(db).filter(t => (/^[A-Z]{1,5}$/.test(t) || /^\^[A-Z]{1,5}$/.test(t)) && !db[t].type?.startsWith('^')).sort();
          if (tokens.length === 0) { console.log('No tokens suitable for model addition.'); rl.prompt(); return; }
          console.log(`Tokens eligible for model (${tokens.length}):`);
          for (const t of tokens) {
            const info = db[t];
            console.log(`  ${t.padEnd(8)} ${(info.name || '').padEnd(35)} [${info.type || '?'}] @${info.exchange || '?'}`);
          }
          console.log(`\nAdd to model: ids add-model`);
          saveIdentifiers({});
          console.log('All identifiers cleared.');
        } else {
          const filter = sub ? sub.toUpperCase() : '';
          const entries = Object.entries(db).filter(([k]) => !filter || k.includes(filter) || (db[k].name || '').toUpperCase().includes(filter));
          if (entries.length === 0) { console.log('No identifiers found.'); rl.prompt(); return; }
          console.log(`Identifiers (${entries.length}):`);
          for (const [ticker, info] of entries.sort(([a], [b]) => a.localeCompare(b))) {
            const ex = info.exchange ? ` @${info.exchange}` : '';
            const typ = info.type ? ` [${info.type}]` : '';
            console.log(`  ${ticker.padEnd(8)} ${(info.name || '').padEnd(35)}${typ}${ex}`);
          }
        }
        rl.prompt();
        return;
      }

      if (cmd === 'quote') {
        const ticker = (expand(parts[1]) || '').toUpperCase();
        if (!ticker) { console.log('Usage: quote <ticker>'); rl.prompt(); return; }
        try {
          const url = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?range=1d&interval=1d`;
          const body = await httpGet(url);
          const data = JSON.parse(body);
          const c = data.chart.result[0];
          const meta = c.meta;
          const q = c.indicators.quote[0];
          const val = (v, fallback = 0) => { const r = Array.isArray(v) ? v[0] : v; return r ?? fallback; };
          const price = val(meta.regularMarketPrice);
          const prev = val(meta.previousClose);
          const change = prev ? price - prev : 0;
          const pct = prev ? (change / prev) * 100 : 0;
          const db = loadIdentifiers();
          const info = db[ticker];
          const name = info?.name ? ` (${info.name})` : '';
          const arrow = change >= 0 ? '▲' : '▼';
          console.log(`${ticker}${name} @ ${meta.exchangeName || '?'}`);
          console.log(`  ${arrow} $${price.toFixed(2)}  ${change >= 0 ? '+' : ''}${change.toFixed(2)} (${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%)`);
          console.log(`  Open: $${val(q.open).toFixed(2)}  Prev Close: $${prev.toFixed(2)}`);
          console.log(`  Day Range: $${val(q.low).toFixed(2)} – $${val(q.high).toFixed(2)}`);
          console.log(`  Volume: ${val(q.volume, 0).toLocaleString()}`);
          if (info?.exchange) console.log(`  MIC: ${info.exchange}`);
        } catch (e) {
          console.log(`Error fetching ${ticker}: ${e.message}`);
        }
        rl.prompt();
        return;
      }

      if (cmd === 'listings') {
        const query = (expand(parts[1]) || '').toUpperCase();
        if (!query) { console.log('Usage: listings <ticker>'); rl.prompt(); return; }
        try {
          const body = await httpGet(`https://query1.finance.yahoo.com/v1/finance/search?q=${encodeURIComponent(query)}`);
          const data = JSON.parse(body);
          const quotes = data.quotes || [];
          const db = loadIdentifiers();
          const exact = [];
          const related = [];
          for (const q of quotes) {
            const entry = { sym: q.symbol, name: q.shortname || q.longname || '', exch: q.exchange || '?', type: q.quoteType || '?', mic: q.isYahooFinance ? '' : '' };
            const isSame = q.symbol?.toUpperCase() === query
              || q.shortname?.toUpperCase().includes(query)
              || (db[q.symbol] && db[q.symbol].longname?.toUpperCase().includes(query));
            (isSame ? exact : related).push(entry);
          }
          if (exact.length) {
            console.log(`\nDirect matches for "${query}":`);
            for (const e of exact) console.log(`  ${e.sym.padEnd(14)} ${e.exch.padEnd(6)} ${e.type.padEnd(8)} ${e.name}`);
          }
          if (related.length) {
            console.log(`\nRelated (same name / cross-listings):`);
            for (const e of related.slice(0, 10)) console.log(`  ${e.sym.padEnd(14)} ${e.exch.padEnd(6)} ${e.type.padEnd(8)} ${e.name}`);
          }
          if (!exact.length && !related.length) console.log('No results.');
        } catch (e) {
          console.log(`Error: ${e.message}`);
        }
        rl.prompt();
        return;
      }

      if (cmd === 'chart') {
        const ticker = (expand(parts[1]) || '').toUpperCase();
        if (!ticker) { console.log('Usage: chart <ticker> [range] [interval]'); rl.prompt(); return; }
        const range = parts[2] || '1d';
        const interval = parts[3] || '1m';
        try {
          const body = await httpGet(`https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?range=${range}&interval=${interval}`);
          const data = JSON.parse(body);
          const r = data.chart.result[0];
          const meta = r.meta;
          const q = r.indicators.quote[0];
          const ts = r.timestamp;
          const closes = q.close;
          const volumes = q.volume;
          const db = loadIdentifiers();
          const info = db[ticker];

          const vals = closes.map((c, i) => ({ t: ts[i], c, v: volumes[i], o: q.open[i], h: q.high[i], l: q.low[i] })).filter(x => x.c !== null);
          if (vals.length === 0) { console.log('No data'); rl.prompt(); return; }

          const open = vals[0].o;
          const high = Math.max(...vals.map(x => x.h));
          const low = Math.min(...vals.map(x => x.l));
          const close = vals[vals.length - 1].c;
          const totalVol = vals.reduce((s, x) => s + (x.v || 0), 0);
          const change = open ? close - open : 0;
          const pct = open ? (change / open) * 100 : 0;
          const arrow = change >= 0 ? '▲' : '▼';

          console.log(`${ticker}${info?.name ? ' (' + info.name + ')' : ''}  ${range}/${interval}  (${vals.length} bars)`);
          console.log(`  ${arrow} $${close.toFixed(2)}  ${change >= 0 ? '+' : ''}${change.toFixed(2)} (${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%)`);
          console.log(`  O: $${open.toFixed(2)}  H: $${high.toFixed(2)}  L: $${low.toFixed(2)}  C: $${close.toFixed(2)}`);
          console.log(`  Volume: ${totalVol.toLocaleString()}`);

          // Text sparkline with guaranteed extreme visibility
          const sparkClose = vals.map(x => x.c);
          const minC = Math.min(...sparkClose), maxC = Math.max(...sparkClose);
          const rangeC = maxC - minC || 1;
          const minIdx = sparkClose.indexOf(minC), maxIdx = sparkClose.indexOf(maxC);
          const W = Math.max(20, Math.min(80, vals.length));
          const step = Math.max(1, Math.floor(vals.length / W));
          const idxs = [];
          for (let i = 0; i < vals.length; i += step) idxs.push(i);
          if (idxs.at(-1) !== vals.length - 1) idxs.push(vals.length - 1);
          if (!idxs.includes(minIdx)) idxs.push(minIdx);
          if (!idxs.includes(maxIdx)) idxs.push(maxIdx);
          idxs.sort((a, b) => a - b);
          const blocks = ['▁','▂','▃','▄','▅','▆','▇','█'];
          const spark = idxs.map(i => blocks[Math.min(7, Math.floor(((sparkClose[i] - minC) / rangeC) * 8))]).join('');
          const labelW = spark.length;
          console.log(`  $${maxC.toFixed(2)} ${' '.repeat(Math.max(0, labelW - 10))}  H`);
          console.log(`  ${spark}`);
          console.log(`  $${minC.toFixed(2)} ${' '.repeat(Math.max(0, labelW - 10))}  L`);
          const timeStart = new Date(vals[0].t * 1000).toLocaleTimeString();
          const timeEnd = new Date(vals[vals.length - 1].t * 1000).toLocaleTimeString();
          console.log(`  ${timeStart}${' '.repeat(Math.max(0, labelW - timeStart.length - timeEnd.length - 2))}${timeEnd}`);

          // Last few bars
          const lastN = Math.min(5, vals.length);
          console.log(`\n  Last ${lastN}:`);
          for (let i = vals.length - lastN; i < vals.length; i++) {
            const x = vals[i];
            const barChange = x.c - x.o;
            const barArrow = barChange >= 0 ? '▲' : '▼';
            const time = new Date(x.t * 1000).toLocaleTimeString();
            console.log(`    ${time}  ${barArrow} $${x.c.toFixed(2)}  H:$${x.h.toFixed(2)} L:$${x.l.toFixed(2)}  Vol:${(x.v || 0).toLocaleString()}`);
          }
        } catch (e) {
          console.log(`Error: ${e.message}`);
        }
        rl.prompt();
        return;
      }

      if (cmd === 'show' || cmd === 'explore') {
        const word = expand(parts[1]);
        if (!word) { console.log('Usage: show <word>'); rl.prompt(); return; }
        if (useMIC) console.log(`(expanded: "${word}")`);
        const emb = await getEmbedding(word);
        const n = parseFloat(emb.reduce((s, v) => s + v * v, 0));
        const norm = Math.sqrt(n);
        const mean = emb.reduce((s, v) => s + v, 0) / emb.length;
        const min = Math.min(...emb);
        const max = Math.max(...emb);
        const std = Math.sqrt(emb.reduce((s, v) => s + (v - mean) ** 2, 0) / emb.length);
        console.log(`"${word}" — ${emb.length}-dim embedding`);
        console.log(`  norm: ${norm.toFixed(4)}  range: [${min.toFixed(4)}, ${max.toFixed(4)}]  μ=${mean.toFixed(4)}  σ=${std.toFixed(4)}`);

        // Top-10 highest-activating dimensions
        const dims = emb.map((v, i) => ({ dim: i, val: v }));
        dims.sort((a, b) => b.val - a.val);
        console.log('  top activating dimensions:');
        for (let i = 0; i < 8; i++) {
          console.log(`    dim ${String(dims[i].dim).padStart(3)} = ${dims[i].val.toFixed(4)}`);
        }
        console.log('  lowest activating dimensions:');
        for (let i = 0; i < 8; i++) {
          const d = dims[dims.length - 1 - i];
          console.log(`    dim ${String(d.dim).padStart(3)} = ${d.val.toFixed(4)}`);
        }

        // Compare with baseline nearest neighbors
        const blEmbs2 = await Promise.all(BASELINE_WORDS.map(w => getEmbedding(w)));
        const sims2 = BASELINE_WORDS.map((w, i) => ({ word: w, sim: cosineSimilarity(emb, blEmbs2[i]) }));
        sims2.sort((a, b) => b.sim - a.sim);
        console.log('  nearest neighbors:');
        for (let i = 0; i < 5; i++) {
          console.log(`    ${sims2[i].word}  sim=${sims2[i].sim.toFixed(4)}`);
        }
        rl.prompt();
        return;
      }

      if (cmd === 'diff' && parts.length >= 3) {
        const wordA = expand(parts[1]);
        const wordB = expand(parts.slice(2).join(' '));
        if (useMIC) console.log(`(expanded: "${wordA}" / "${wordB}")`);
        const [embA, embB] = await Promise.all([getEmbedding(wordA), getEmbedding(wordB)]);
        const diff = embA.map((v, i) => v - embB[i]);
        const absDiff = diff.map(v => Math.abs(v));
        const maxDiff = Math.max(...absDiff);
        const avgDiff = diff.reduce((s, v) => s + v, 0) / diff.length;
        const diffNorm = Math.sqrt(diff.reduce((s, v) => s + v * v, 0));
        const sorted = diff.map((v, i) => ({ dim: i, val: v, abs: Math.abs(v) }));
        sorted.sort((a, b) => b.abs - a.abs);
        console.log(`"${wordA}" vs "${wordB}" — diff vector`);
        console.log(`  diff norm: ${diffNorm.toFixed(4)}  max|Δ|: ${maxDiff.toFixed(4)}  avg Δ: ${avgDiff.toFixed(4)}`);
        console.log('  dimensions with largest difference:');
        for (let i = 0; i < 8; i++) {
          const dir = sorted[i].val > 0 ? `${wordA}>${wordB}` : `${wordB}>${wordA}`;
          console.log(`    dim ${String(sorted[i].dim).padStart(3)}  Δ=${sorted[i].val.toFixed(4)}  (${dir})`);
        }
        rl.prompt();
        return;
      }

      // Default: treat as a single word/phrase to embed
      const raw = parts.join(' ');
      const defaultText = expand(raw);
      if (useMIC && defaultText !== raw) console.log(`(expanded: "${defaultText}")`);
      const emb = await getEmbedding(defaultText);
      const blEmbs = await Promise.all(BASELINE_WORDS.map(w => getEmbedding(w)));
      const dists = BASELINE_WORDS.map((w, i) => ({ word: w, dist: cosineDistance(emb, blEmbs[i]) }));
      dists.sort((a, b) => a.dist - b.dist);
      const nn5 = dists.slice(0, 5);
      console.log(`"${defaultText}" — ${emb.length}-dim embedding`);
      console.log(`  closest baseline words: ${nn5.map((d, i) => `${d.word}(${d.dist.toFixed(3)})`).join(', ')}`);
      console.log(`  vs baseline avg: ${fmtDist(dists.reduce((s, d) => s + d.dist, 0) / dists.length, bl.mean, bl.std)}`);

    } catch (err) {
      console.error(`Error: ${err.message}`);
    }

    rl.prompt();
  });

  rl.on('close', () => {
    console.log('\nGoodbye.');
    process.exit(0);
  });
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
