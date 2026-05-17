#!/usr/bin/env python3
"""Pre-compute full-transformer embeddings for all non-subword tokens.

Usage: python build_embed_index.py

Outputs (in model dir):
  vocab_embed.npy    — [N, 384] float32, L2-normalized
  vocab_embed_tokens.json — [N] list of token strings
  vocab_embed_ids.json    — [N] list of token IDs
"""

import json, os, sys, time
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import onnx
from onnx import numpy_helper

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MIC_MODEL = os.path.join(SCRIPTS_DIR, 'modelFiles', 'Xenova', 'all-MiniLM-L6-v2-mic')

# Load MIC tokenizer (extended vocab)
tokenizer = AutoTokenizer.from_pretrained(MIC_MODEL)

# Load the original PyTorch model from HF hub and inject our ONNX embedding
print("Loading model from hub...")
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Override embedding weight with MIC model's extended ONNX matrix
onnx_path = os.path.join(MIC_MODEL, 'onnx', 'model.onnx')
onnx_model = onnx.load(onnx_path)
for init in onnx_model.graph.initializer:
    if init.name == 'embeddings.word_embeddings.weight':
        W_onnx = numpy_helper.to_array(init)
        break

old_num = model.config.vocab_size
new_num = W_onnx.shape[0]
model.resize_token_embeddings(new_num)
model.config.vocab_size = new_num
with torch.no_grad():
    model.embeddings.word_embeddings.weight.copy_(torch.from_numpy(W_onnx))
print(f"  Injected ONNX embedding: [{old_num}, 384] → [{new_num}, 384]")

model.eval()

# Load vocab
tok_path = os.path.join(MIC_MODEL, 'tokenizer.json')
with open(tok_path) as f:
    tok_data = json.load(f)
vocab = tok_data['model']['vocab']

# Build token list: skip special tokens, skip ## subwords, keep everything else
special = {'[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'}
token_list = []
for token, tid_str in vocab.items():
    tid = int(tid_str)
    if token in special:
        continue
    if token.startswith('##'):
        continue
    token_list.append((tid, token))

token_list.sort(key=lambda x: x[0])
tokens = [t for _, t in token_list]
ids = [i for i, _ in token_list]

print(f"Tokens to embed: {len(tokens)}")

# Batch embed
BATCH = 512
all_embs = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

with torch.no_grad():
    for start in range(0, len(tokens), BATCH):
        batch = tokens[start:start + BATCH]
        t0 = time.time()
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=16)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs).last_hidden_state
        # Mean pool over non-padding tokens
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
        # L2 normalize
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embs.append(pooled.cpu().numpy())
        dt = time.time() - t0
        print(f"  [{start:>5}–{start + len(batch):>5}] {len(batch):>3} tokens  ({dt:.1f}s)")

matrix = np.concatenate(all_embs, axis=0).astype(np.float32)
print(f"\nMatrix shape: {matrix.shape}")

# Save
np.save(os.path.join(MIC_MODEL, 'vocab_embed.npy'), matrix)
matrix.tofile(os.path.join(MIC_MODEL, 'vocab_embed.bin'))
with open(os.path.join(MIC_MODEL, 'vocab_embed_tokens.json'), 'w') as f:
    json.dump(tokens, f)
with open(os.path.join(MIC_MODEL, 'vocab_embed_ids.json'), 'w') as f:
    json.dump(ids, f)

# Save raw embedding weight (for family commands — 4 vacuum dims)
# We save the FULL unfiltered vocabulary here to ensure index parity with the matrix rows
full_tokens = [None] * W_onnx.shape[0]
full_ids = [None] * W_onnx.shape[0]
for token, tid_str in vocab.items():
    tid = int(tid_str)
    if tid < W_onnx.shape[0]:
        full_tokens[tid] = token
        full_ids[tid] = tid

W_onnx.astype(np.float32).tofile(os.path.join(MIC_MODEL, 'embed_weight.bin'))
with open(os.path.join(MIC_MODEL, 'embed_weight_tokens.json'), 'w') as f:
    json.dump(full_tokens, f)
with open(os.path.join(MIC_MODEL, 'embed_weight_ids.json'), 'w') as f:
    json.dump(full_ids, f)

# Build family_vac.json: 4-d vacuum vectors for family tokens only
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
VAC_DIMS = [18, 62, 28, 245]
family_tokens = set()
# Markers from family_dimensions.json
with open(os.path.join(SCRIPTS_DIR, 'family_dimensions.json')) as f:
    fam_cfg = json.load(f)
for entry in fam_cfg['tokens']:
    family_tokens.add(entry['token'])
    family_tokens.add(entry['token'].lower())
# MIC codes from known_mics.json
with open(os.path.join(SCRIPTS_DIR, 'known_mics.json')) as f:
    for entry in json.load(f):
        code = entry['mic']
        family_tokens.add(code)
        family_tokens.add(code.lower())
        family_tokens.add(code.upper())

# ISINs and Listings from registry.json
registry_path = os.path.join(SCRIPTS_DIR, 'registry.json')
if os.path.exists(registry_path):
    with open(registry_path) as f:
        reg = json.load(f)
        for ent in reg.get('entities', {}).values():
            if ent.get('isin'):
                family_tokens.add(ent['isin'].lower())
            for li in ent.get('listings', []):
                if li.get('sedol'): family_tokens.add(li['sedol'].lower())
                if li.get('ticker'): family_tokens.add(li['ticker'].lower())

family_vac = {}
for token in family_tokens:
    tid = int(vocab.get(token, -1))
    if tid < 0:
        continue
    family_vac[token] = [float(x) for x in W_onnx[tid, VAC_DIMS]]
with open(os.path.join(MIC_MODEL, 'family_vac.json'), 'w') as f:
    json.dump(family_vac, f)

print(f"  embed_weight.bin — raw ONNX embedding ({W_onnx.shape[0]}×{W_onnx.shape[1]})")
print(f"  family_vac.json  — {len(family_vac)} family token 4-d vectors")
