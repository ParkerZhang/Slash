#!/usr/bin/env python3
"""
Financial Identifier Embedding System
======================================
Uses 4 vacuum dimensions (18, 62, 28, 245) in a frozen BERT model
(all-MiniLM-L6-v2) to encode family structure without retraining.

Vacuum dimensions have near-zero activation for all 30,522 base tokens,
giving us 4 noise-free slots for synthetic family vectors.

Approach
--------
1. Family prototypes (4-d vacuum vectors):
     #MIC  = [0.30, 0.15, 0.15, 0.15]   dim18 dominant
     ISIN  = [0.15, 0.30, 0.15, 0.15]   dim62 dominant
     SEDOL = [0.15, 0.15, 0.30, 0.15]   dim28 dominant
     TICKER= [0.15, 0.15, 0.15, 0.30]   dim245 dominant
   Cross-family cos = 0.857 (uniform).

2. MIC subtype fingerprint:
     Primary: [0.30, 0.08, 0.08, 0.08]  (same cos to #MIC = 0.961)
     Custom:  [0.30, a, b, c] where a,b,c may differ for uniqueness

3. Semantic blend:
     Each MIC code's full 384-d embedding = family_vec + beta * phrase_dir
     where phrase_dir = unit vector of phrase (exchange name) in 380
     non-vacuum dims, and beta = family_norm * 2.

4. Classification (family_classifier.py):
     Two gates to reject false positives:
       a. 4-d cosine to nearest prototype  > 0.9
       b. vac_norm / full_norm             > 0.25
     Without (b), 25+ tokens false-positive at cos > 0.98.
"""

import json, os, sys
import numpy as np
import onnx
from onnx import numpy_helper
from transformers import AutoTokenizer


VAC_DIMS = [18, 62, 28, 245]

FAMILY_PROTOS = {
    'MIC':    np.array([0.30, 0.15, 0.15, 0.15], dtype=np.float32),
    'ISIN':   np.array([0.15, 0.30, 0.15, 0.15], dtype=np.float32),
    'SEDOL':  np.array([0.15, 0.15, 0.30, 0.15], dtype=np.float32),
    'TICKER': np.array([0.15, 0.15, 0.15, 0.30], dtype=np.float32),
}

# MIC subtypes share a common vacuum fingerprint.
# XTSE uses a distinct fingerprint for separability.
MIC_SUBTYPE_FINGERPRINTS = {
    'default': [0.30, 0.08, 0.08, 0.08],
    'XTSE':    [0.30, 0.10, 0.04, 0.06],
}


def make_family_vector(fingerprint):
    """Build a 384-d vector with family values in vacuum dims, 0 elsewhere."""
    v = np.zeros(384, dtype=np.float32)
    v[VAC_DIMS] = fingerprint
    return v


def phrase_direction(W, tokenizer, text):
    """Mean-pooled phrase embedding projected onto 380 non-vacuum dims, unit-normalized."""
    ids = tokenizer.encode(text)[1:-1]
    if not ids:
        return None
    phrase = W[ids].mean(axis=0)
    non_vac = [d for d in range(384) if d not in VAC_DIMS]
    d = np.zeros(384, dtype=np.float32)
    d[non_vac] = phrase[non_vac]
    n = float(np.linalg.norm(d))
    if n < 1e-8:
        return None
    d /= n
    return d


def build_mic_embedding(W, tokenizer, mic_code, exchange_name, fingerprint=None):
    """Full 384-d = family_vector + beta * phrase_direction.

    beta = family_norm * 2  (semantic and family components contribute equally
    to the total norm, giving ~0.88 cos to the phrase and ~0.44 ratio).
    """
    if fingerprint is None:
        fingerprint = MIC_SUBTYPE_FINGERPRINTS.get(mic_code, MIC_SUBTYPE_FINGERPRINTS['default'])
    fam = make_family_vector(fingerprint)
    fn = float(np.linalg.norm(fam))
    beta = fn * 2
    pdir = phrase_direction(W, tokenizer, exchange_name)
    if pdir is None:
        return fam
    return fam + beta * pdir


def apply_to_all_mic_codes(model_dir, known_mics_path, dry_run=False):
    """Apply the MIC subtype treatment to all entries in known_mics.json."""
    with open(known_mics_path) as f:
        entries = json.load(f)

    tok_path = os.path.join(model_dir, 'tokenizer.json')
    with open(tok_path) as f:
        vocab = json.load(f)['model']['vocab']

    model = onnx.load(os.path.join(model_dir, 'onnx', 'model.onnx'))
    for i, init in enumerate(model.graph.initializer):
        if init.name == 'embeddings.word_embeddings.weight':
            W = numpy_helper.to_array(init).copy()
            break
    else:
        raise ValueError('embeddings.word_embeddings.weight not found')

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    results = []

    for entry in entries:
        code = entry['mic']
        name = entry['name']
        for variant in {code, code.lower()}:
            tid = int(vocab.get(variant, -1))
            if tid < 0:
                print(f"  SKIP {variant}: not in vocab")
                continue
            emb = build_mic_embedding(W, tokenizer, code, name)
            if not dry_run:
                W[tid] = emb.astype(np.float32)
            results.append((variant, tid, name, emb))

    if not dry_run:
        new_init = numpy_helper.from_array(W, name=init.name)
        model.graph.initializer[i].CopyFrom(new_init)
        onnx.save(model, os.path.join(model_dir, 'onnx', 'model.onnx'))
        print(f"Saved {len(results)} embeddings to {model_dir}")

    return results


def classify_family(token, W, vocab):
    """Returns (family_name, cos_sim) or (None, 0.0).

    Gates: vac_norm > 0.05, vac_norm/full_norm > 0.25, 4-d cos > 0.9
    """
    if isinstance(token, str):
        tid = int(vocab.get(token.strip(), -1))
    else:
        tid = int(token)
    if tid < 0:
        return None, 0.0

    vac = W[tid, VAC_DIMS]
    vn = float(np.linalg.norm(vac))
    fn = float(np.linalg.norm(W[tid]))
    if vn < 0.05 or vn / fn < 0.25:
        return None, 0.0

    pn = float(np.linalg.norm(list(FAMILY_PROTOS.values())[0]))
    best_fam, best_sim = None, 0.0
    for name, proto in FAMILY_PROTOS.items():
        sim = float(np.dot(vac, proto) / (vn * pn))
        if sim > best_sim:
            best_sim, best_fam = sim, name

    if best_sim >= 0.9:
        return best_fam, round(best_sim, 4)
    return None, round(best_sim, 4)


if __name__ == '__main__':
    scripts_dir = os.path.dirname(__file__)
    model_dir = os.path.join(scripts_dir, 'modelFiles', 'Xenova', 'all-MiniLM-L6-v2-mic')
    known_path = os.path.join(scripts_dir, 'known_mics.json')

    print(f"Applying MIC code embeddings to {model_dir}")
    print(f"Using known MICS from {known_path}")
    print("-" * 50)

    apply_to_all_mic_codes(model_dir, known_path)

    # Verify
    tok_path = os.path.join(model_dir, 'tokenizer.json')
    with open(tok_path) as f:
        vocab = json.load(f)['model']['vocab']
    model = onnx.load(os.path.join(model_dir, 'onnx', 'model.onnx'))
    for init in model.graph.initializer:
        if init.name == 'embeddings.word_embeddings.weight':
            W = numpy_helper.to_array(init)
            break

    print("\nVerification:")
    for code in ['XNYS', 'XLON', 'XTSE', 'XNAS', 'XHKG', '#MIC', 'ISIN', 'SEDOL', 'TICKER', 'rams', 'stock']:
        fam, sim = classify_family(code, W, vocab)
        status = f"→ {fam} ({sim:.3f})" if fam else "→ none"
        print(f"  {code:>6s} {status}")
