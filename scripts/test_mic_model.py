#!/usr/bin/env python3
"""Test script: measure distances among markers, MIC codes, and exchange names."""

import json, os, sys
import numpy as np
import onnx
from onnx import numpy_helper
from transformers import AutoTokenizer

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MIC_MODEL = os.path.join(SCRIPTS_DIR, 'modelFiles', 'Xenova', 'all-MiniLM-L6-v2-mic')
KNOWN_MICS = os.path.join(SCRIPTS_DIR, 'known_mics.json')

VAC_DIMS = [18, 62, 28, 245]

# Load
tok_path = os.path.join(MIC_MODEL, 'tokenizer.json')
with open(tok_path) as f:
    vocab = json.load(f)['model']['vocab']

model = onnx.load(os.path.join(MIC_MODEL, 'onnx', 'model.onnx'))
for init in model.graph.initializer:
    if init.name == 'embeddings.word_embeddings.weight':
        W = numpy_helper.to_array(init)
        break

tokenizer = AutoTokenizer.from_pretrained(MIC_MODEL)

with open(KNOWN_MICS) as f:
    mic_entries = json.load(f)

def token_id(token):
    for key in (token, token.lower(), token.upper()):
        if key in vocab:
            return int(vocab[key])
    return -1

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def vac_cos(a, b):
    return cos(a[VAC_DIMS], b[VAC_DIMS])

# ----------------------------------------------------------------
print("=" * 60)
print("1. 4-FAMILY MARKER DISTANCES (vacuum dims only)")
print("=" * 60)

markers = ['#MIC', 'ISIN', 'SEDOL', 'TICKER']
M = {m: W[token_id(m)] for m in markers if token_id(m) >= 0}

# Matrix
print(f"\n{'':>10s}", end='')
for m2 in markers:
    print(f"{m2:>10s}", end='')
print()
for m1 in markers:
    print(f"{m1:>10s}", end='')
    for m2 in markers:
        c = vac_cos(M[m1], M[m2])
        print(f"{c:10.4f}", end='')
    print()

# Also show the vacuum vectors
print("\nVacuum vectors:")
for m in markers:
    tid = token_id(m)
    if tid >= 0:
        print(f"  {m:>6s}: {np.round(W[tid, VAC_DIMS], 4).tolist()}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("2. MIC CODE ↔ EXCHANGE NAME (full 384-d)")
print("=" * 60)

print(f"\n{'MIC':>6s}  → name                              cos")
print("-" * 55)
for entry in mic_entries:
    code, name = entry['mic'], entry['name']
    tid = token_id(code)
    if tid < 0:
        continue
    ids = tokenizer.encode(name)[1:-1]
    if not ids:
        continue
    name_emb = W[ids].mean(axis=0)
    c = cos(W[tid], name_emb)
    print(f"{code:>6s}  → {name:35s} {c:.4f}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("3. #MIC ↔ MIC CODES (full 384-d)")
print("=" * 60)

mic_emb = W[token_id('#MIC')]
print(f"\n{'MIC code':>6s}  {'full cos':>9s}  {'vac cos':>9s}")
print("-" * 30)
for entry in mic_entries:
    code = entry['mic']
    tid = token_id(code)
    if tid < 0:
        continue
    c_full = cos(mic_emb, W[tid])
    c_vac = vac_cos(mic_emb, W[tid])
    print(f"{code:>6s}  {c_full:9.4f}  {c_vac:9.4f}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("4. #MIC ↔ EXCHANGE NAMES (full 384-d)")
print("=" * 60)

print(f"\n{'Name':35s}  cos")
print("-" * 45)
for entry in mic_entries:
    name = entry['name']
    ids = tokenizer.encode(name)[1:-1]
    if not ids:
        continue
    name_emb = W[ids].mean(axis=0)
    c = cos(mic_emb, name_emb)
    print(f"{name:35s}  {c:.4f}")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("5. MIC CODE ↔ MIC CODE (full 384-d, first 5 x 5)")
print("=" * 60)

codes = [e['mic'] for e in mic_entries]
print(f"\n{'':>6s}", end='')
for c2 in codes[:5]:
    print(f"{c2:>8s}", end='')
print()
for c1 in codes[:5]:
    print(f"{c1:>6s}", end='')
    for c2 in codes[:5]:
        c = cos(W[token_id(c1)], W[token_id(c2)])
        print(f"{c:8.4f}", end='')
    print()

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print(f"6. STATS: norm & ratio for all tokens of interest")
print("=" * 60)

def classify_inline(token):
    tid = token_id(token)
    if tid < 0:
        return 'N/A'
    vac = W[tid, VAC_DIMS]
    vn = float(np.linalg.norm(vac))
    fn = float(np.linalg.norm(W[tid]))
    if vn < 0.05 or vn / fn < 0.25:
        return 'none'
    protos = {
        'MIC': np.array([0.40, 0.00, 0.00, 0.00]),
        'ISIN': np.array([0.00, 0.40, 0.00, 0.00]),
        'SEDOL': np.array([0.00, 0.00, 0.40, 0.00]),
        'TICKER': np.array([0.00, 0.00, 0.00, 0.40]),
    }
    pn = float(np.linalg.norm(protos['MIC']))
    best, best_sim = None, 0.0
    for n, p in protos.items():
        s = float(np.dot(vac, p) / (vn * pn))
        if s > best_sim:
            best_sim, best = s, n
    return best if best_sim >= 0.9 else 'none'

print(f"\n{'Token':>8s}  {'vac_norm':>8s}  {'full_norm':>8s}  {'ratio':>6s}  {'family':>7s}")
print("-" * 42)
for token in markers + ['MIC'] + [e['mic'] for e in mic_entries]:
    tid = token_id(token)
    if tid < 0:
        continue
    vn = float(np.linalg.norm(W[tid, VAC_DIMS]))
    fn = float(np.linalg.norm(W[tid]))
    ratio = vn / fn if fn > 0 else 0
    fam = classify_inline(token)
    print(f"{token:>8s}  {vn:8.4f}  {fn:8.4f}  {ratio:6.3f}  {fam:>7s}")
