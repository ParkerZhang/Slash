#!/usr/bin/env python3
import json, os, sys
import numpy as np
import onnx
from onnx import numpy_helper

with open(os.path.join(os.path.dirname(__file__), 'family_dimensions.json')) as f:
    FAMILY_CFG = json.load(f)

VAC_DIMS = FAMILY_CFG['family']['dimensions']
FAMILY_DEFS = {}
for entry in FAMILY_CFG['tokens']:
    name = entry['token'].lstrip('#')
    FAMILY_DEFS[name] = np.array(entry['vector'], dtype=np.float32)


def load_embeddings(model_dir):
    onnx_path = os.path.join(model_dir, 'onnx', 'model.onnx')
    tok_path = os.path.join(model_dir, 'tokenizer.json')
    model = onnx.load(onnx_path)
    with open(tok_path) as f:
        vocab = json.load(f)['model']['vocab']
    for init in model.graph.initializer:
        if init.name == 'embeddings.word_embeddings.weight':
            W = numpy_helper.to_array(init)
            return W, vocab
    raise ValueError('embeddings.word_embeddings.weight not found')


def classify_family(token, model_dir, threshold=0.9, min_norm=0.05, min_ratio=0.25):
    """Classify a token into a financial identifier family.

    Two gates to reject false positives:
      1. **cosine** (4 vacuum dims only) — matches the family direction
      2. **vacuum-norm / full-norm ratio** — rejects tokens whose vacuum
         alignment is a tiny fraction of their total 384-d magnitude

    Without the ratio gate, 25+ base-model tokens (e.g. `directly`, `ee`,
    `rams`) false-positive at cos ≥ 0.98. The ratio gate rejects them
    (ratio ≈ 0.04–0.17 vs ≥ 0.25 for genuine family tokens).

    Returns (family_name, cosine_similarity) or (None, 0.0).
    Families: MIC, ISIN, SEDOL, TICKER
    """
    if isinstance(token, str):
        token = token.strip()
    W, vocab = load_embeddings(model_dir)
    if isinstance(token, str):
        if token not in vocab:
            return None, 0.0
        tid = int(vocab[token])
    else:
        tid = int(token)
    vac_vec = W[tid, VAC_DIMS]
    full_norm = float(np.linalg.norm(W[tid]))
    vac_norm = float(np.linalg.norm(vac_vec))
    if vac_norm < min_norm:
        return None, 0.0
    if vac_norm / full_norm < min_ratio:
        return None, 0.0

    best_fam, best_sim = None, 0.0
    proto_norm = float(np.linalg.norm(list(FAMILY_DEFS.values())[0]))
    for fam, proto in FAMILY_DEFS.items():
        sim = float(np.dot(vac_vec, proto) / (vac_norm * proto_norm))
        if sim > best_sim:
            best_sim = sim
            best_fam = fam
    if best_sim >= threshold:
        return best_fam, round(best_sim, 4)
    return None, round(best_sim, 4)


def classify_topk(token, model_dir, k=4, min_norm=0.05, min_ratio=0.25):
    """Return ranked families for a token."""
    if isinstance(token, str):
        token = token.strip()
    W, vocab = load_embeddings(model_dir)
    if isinstance(token, str):
        if token not in vocab:
            return []
        tid = int(vocab[token])
    else:
        tid = int(token)
    vac_vec = W[tid, VAC_DIMS]
    full_norm = float(np.linalg.norm(W[tid]))
    vac_norm = float(np.linalg.norm(vac_vec))
    if vac_norm < min_norm or vac_norm / full_norm < min_ratio:
        return []
    proto_norm = float(np.linalg.norm(list(FAMILY_DEFS.values())[0]))
    results = []
    for fam, proto in FAMILY_DEFS.items():
        sim = float(np.dot(vac_vec, proto) / (vac_norm * proto_norm))
        results.append((fam, round(sim, 4)))
    results.sort(key=lambda x: -x[1])
    return results[:k]


if __name__ == '__main__':
    MIC_MODEL = os.path.join(
        os.path.dirname(__file__),
        'modelFiles', 'Xenova', 'all-MiniLM-L6-v2-mic'
    )
    for arg in sys.argv[1:]:
        W, vocab = load_embeddings(MIC_MODEL)
        key = arg.strip()
        exists = key in vocab
        fam, sim = classify_family(key, MIC_MODEL)
        if fam:
            print(f'{arg:15s} → {fam:6s} (sim={sim:.4f})')
        elif not exists:
            print(f'{arg:15s} → not in vocab')
        elif sim == 0.0:
            tid = int(vocab[key])
            vn = float(np.linalg.norm(W[tid, VAC_DIMS]))
            fn = float(np.linalg.norm(W[tid]))
            if vn < 0.05:
                print(f'{arg:15s} → noise (vac-norm={vn:.3f})')
            else:
                print(f'{arg:15s} → noise (ratio={vn/fn:.3f} < 0.25)')
        else:
            top = classify_topk(key, MIC_MODEL)
            first = top[0] if top else ('?', 0)
            print(f'{arg:15s} → none  (best: {first[0]} sim={first[1]:.4f})')
