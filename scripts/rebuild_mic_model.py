#!/usr/bin/env python3
"""
Rebuild the MIC model from scratch.

Steps:
  1. Delete existing MIC model
  2. Copy base model (all-MiniLM-L6-v2) → all-MiniLM-L6-v2-mic
  3. Add all tokens (tickers, MIC codes, markers) with sequential IDs
  4. Set family marker vacuum vectors (#MIC, ISIN, SEDOL, TICKER)
  5. Set MIC subtype embeddings (family + semantic blend)
  6. Verify classification
"""

import json, os, shutil, sys
import numpy as np
import onnx
from onnx import numpy_helper
from transformers import AutoTokenizer


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = os.path.join(SCRIPTS_DIR, 'modelFiles', 'Xenova', 'all-MiniLM-L6-v2')
MIC_MODEL = os.path.join(SCRIPTS_DIR, 'modelFiles', 'Xenova', 'all-MiniLM-L6-v2-mic')
KNOWN_MICS = os.path.join(SCRIPTS_DIR, 'known_mics.json')
IDENTIFIERS = os.path.join(SCRIPTS_DIR, 'identifiers.json')

VAC_DIMS = [18, 62, 28, 245]

FAMILY_MARKERS = [
    'isin',
    'sedol',
    'ticker',
    'mic',
    '#mic',
]

MARKER_VECTORS = {
    '#mic':  [0.40, 0.00, 0.00, 0.00],
    'mic':   [0.30, 0.00, 0.00, 0.00],
    'isin':  [0.00, 0.40, 0.00, 0.00],
    'sedol': [0.00, 0.00, 0.40, 0.00],
    'ticker':[0.00, 0.00, 0.00, 0.40],
}

MIC_FINGERPRINTS = {
    'default': [0.40, 0.00, 0.00, 0.00],
    'XTSE':    [0.40, 0.05, 0.00, 0.00], # Slight ISIN lean for TSX Venture?
}


def step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def needs_added_token(content):
    return bool(content) and (not content.isalnum() or len(content) > 5)


def register_added_token(tok, vocab, content, token_key):
    if not needs_added_token(content):
        return 0
    tid = vocab.get(token_key.lower())
    if tid is None:
        return 0
    added_tokens = tok.setdefault('added_tokens', [])
    existing = {entry.get('content') for entry in added_tokens}
    if content in existing:
        return 0
    added_tokens.append({
        'id': int(tid),
        'content': content,
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': False,
    })
    return 1


def register_identifier_token(tok, vocab, value):
    if not value:
        return 0
    raw = value.strip()
    if not raw:
        return 0
    lower = raw.lower()
    changed = register_added_token(tok, vocab, lower, lower)
    upper = raw.upper()
    if upper != lower:
        changed += register_added_token(tok, vocab, upper, lower)
    return changed


step("1. Delete existing MIC model")
if os.path.isdir(MIC_MODEL):
    shutil.rmtree(MIC_MODEL)
    print("  Deleted.")
else:
    print("  Nothing to delete.")

step("2. Copy base model")
shutil.copytree(BASE_MODEL, MIC_MODEL, dirs_exist_ok=True)
print(f"  Copied {BASE_MODEL} → {MIC_MODEL}")

step("3. Add tokens to tokenizer")
tok_path = os.path.join(MIC_MODEL, 'tokenizer.json')
with open(tok_path) as f:
    tok = json.load(f)
vocab = tok['model']['vocab']
max_id = max(int(v) for v in vocab.values())
next_id = max_id + 1
print(f"  Base vocab size: {len(vocab)}, next ID: {next_id}")

# Build ordered token list: tickers, MIC codes, markers
with open(IDENTIFIERS) as f:
    tickers = json.load(f)

ticker_tokens = list(tickers.keys())
with open(KNOWN_MICS) as f:
    mic_entries = json.load(f)

# All tokens in the order they should receive IDs
all_tokens = []
# Tickers (lowercase for BERT)
for t in ticker_tokens:
    all_tokens.append(t.lower())
# MIC codes (lowercase for BERT)
for entry in mic_entries:
    code = entry['mic']
    all_tokens.append(code.lower())
# Family markers (lowercase for BERT)
for m in FAMILY_MARKERS:
    all_tokens.append(m.lower())

added = []
for token in all_tokens:
    if token not in vocab:
        vocab[token] = next_id
        added.append((token, next_id))
        next_id += 1

tok['model']['vocab'] = vocab

# Add custom tokens to added_tokens so WordPiece doesn't split them.
added_token_entries = 0
for token in all_tokens:
    added_token_entries += register_identifier_token(tok, vocab, token)
for t in ticker_tokens:
    added_token_entries += register_identifier_token(tok, vocab, t)
for entry in mic_entries:
    added_token_entries += register_identifier_token(tok, vocab, entry['mic'])
for marker in ['#mic', '#MIC', 'isin', 'ISIN', 'sedol', 'SEDOL', 'ticker', 'TICKER', 'mic', 'MIC']:
    added_token_entries += register_identifier_token(tok, vocab, marker)

with open(tok_path, 'w') as f:
    json.dump(tok, f, indent=2, ensure_ascii=False)

print(f"  Added {len(added)} new tokens (total vocab: {len(vocab)})")
print(f"  Added {added_token_entries} added-token entries")
print(f"  ID range: {max_id + 1}–{next_id - 1}")

step("4. Resize ONNX embedding matrix")
onnx_path = os.path.join(MIC_MODEL, 'onnx', 'model.onnx')
model = onnx.load(onnx_path)

for i, init in enumerate(model.graph.initializer):
    if init.name == 'embeddings.word_embeddings.weight':
        W = numpy_helper.to_array(init).copy()
        orig_rows = W.shape[0]
        emb_dim = W.shape[1]
        new_rows = orig_rows + len(added)
        std = float(W.std())
        new_vectors = np.random.randn(len(added), emb_dim).astype(np.float32) * std * 0.1
        W = np.concatenate([W, new_vectors], axis=0)
        print(f"  Resized: [{orig_rows}, {emb_dim}] → [{new_rows}, {emb_dim}]")
        break

step("5. Set family marker vacuum vectors")
for token, vec in MARKER_VECTORS.items():
    tid = int(vocab.get(token.lower(), -1))
    if tid < 0:
        continue
    W[tid, VAC_DIMS] = vec
    W[tid, [d for d in range(384) if d not in VAC_DIMS]] = 0
    print(f"  {token.lower():>6s} (ID={tid:>5}) ← vac={vec}")

step("6. Set MIC subtype embeddings")
tokenizer = AutoTokenizer.from_pretrained(MIC_MODEL)
non_vac = [d for d in range(384) if d not in VAC_DIMS]

for entry in mic_entries:
    code = entry['mic'].lower()
    name = entry['name']
    fingerprint = MIC_FINGERPRINTS.get(entry['mic'], MIC_FINGERPRINTS['default'])
    fam = np.zeros(384, dtype=np.float32)
    fam[VAC_DIMS] = fingerprint
    fn = float(np.linalg.norm(fam))
    beta = fn * 2

    ids = tokenizer.encode(name)[1:-1]
    if not ids:
        print(f"  SKIP {code}: '{name}' tokenized to nothing")
        continue
    phrase = W[ids].mean(axis=0)
    pdir = np.zeros(384, dtype=np.float32)
    pdir[non_vac] = phrase[non_vac]
    pn = float(np.linalg.norm(pdir))
    if pn < 1e-8:
        print(f"  SKIP {code}: phrase dir norm=0")
        continue
    pdir /= pn

    emb = fam + beta * pdir
    tokens_used = tokenizer.tokenize(name)

    tid = int(vocab.get(code, -1))
    if tid >= 0:
        W[tid] = emb.astype(np.float32)

    print(f"  {code:>6s} vac={fingerprint} ← '{name}' [{', '.join(tokens_used)}]")

step("7. Save ONNX model")
new_init = numpy_helper.from_array(W, name=init.name)
model.graph.initializer[i].CopyFrom(new_init)
onnx.save(model, onnx_path)
print("  Saved.")

step("8. Verification")
# Reload
model = onnx.load(onnx_path)
for init in model.graph.initializer:
    if init.name == 'embeddings.word_embeddings.weight':
        Wv = numpy_helper.to_array(init)
        break

with open(tok_path) as f:
    vocab = json.load(f)['model']['vocab']

def classify(token):
    if token not in vocab:
        return None, 0.0
    tid = int(vocab[token])
    vac = Wv[tid, VAC_DIMS]
    vn = float(np.linalg.norm(vac))
    fn = float(np.linalg.norm(Wv[tid]))
    if vn < 0.05 or vn / fn < 0.25:
        return None, 0.0
    protos = {
        'MIC': np.array([0.30, 0.15, 0.15, 0.15]),
        'ISIN': np.array([0.15, 0.30, 0.15, 0.15]),
        'SEDOL': np.array([0.15, 0.15, 0.30, 0.15]),
        'TICKER': np.array([0.15, 0.15, 0.15, 0.30]),
    }
    pn = float(np.linalg.norm(protos['MIC']))
    best_fam, best_sim = None, 0.0
    for name, p in protos.items():
        s = float(np.dot(vac, p) / (vn * pn))
        if s > best_sim:
            best_sim, best_fam = s, name
    if best_sim >= 0.9:
        return best_fam, round(best_sim, 4)
    return None, round(best_sim, 4)

checks = ['#MIC', 'ISIN', 'SEDOL', 'TICKER',
          'XNYS', 'XLON', 'XTSE', 'XNAS', 'XHKG',
          'rams', 'directly', 'apple', 'stock', 'exchange']
for token in checks:
    fam, sim = classify(token)
    status = f"→ {fam} ({sim:.3f})" if fam else "→ none"
    print(f"  {token:>8s} {status}")

print(f"\n  Total vocab: {len(vocab)}")
print(f"  ONNX matrix: {Wv.shape}")
print("Done.")
