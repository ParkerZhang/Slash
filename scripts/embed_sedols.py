#!/usr/bin/env python3
import csv, json, os, sys, re
import numpy as np
import onnx
from onnx import numpy_helper
from transformers import AutoTokenizer

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MIC_MODEL = os.path.join(SCRIPTS_DIR, 'modelFiles', 'Xenova', 'all-MiniLM-L6-v2-mic')
ONNX_PATH = os.path.join(MIC_MODEL, 'onnx', 'model.onnx')
TOK_PATH = os.path.join(MIC_MODEL, 'tokenizer.json')
SAMPLE_CSV = os.path.join(SCRIPTS_DIR, 'sample.csv')
KNOWN_MICS_FILE = os.path.join(SCRIPTS_DIR, 'known_mics.json')

# Dimension mapping: [MIC, ISIN, SEDOL, TICKER]
VAC_DIMS = [18, 62, 28, 245]
NON_VAC = [d for d in range(384) if d not in VAC_DIMS]

# SEDOL Fingerprint Strengths
SEDOL_STRENGTH = 0.45
MIC_BIAS = 0.10
ISIN_BIAS = 0.10

def main():
    # 1. Extract Data from CSV
    sedol_data = {}
    with open(SAMPLE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row['SEDOL'].strip()
            if not s: continue
            sedol_data[s] = {
                'name': row['NAME'].strip(),
                'isin': row['ISIN'].strip(),
                'mic': row['MIC'].strip()
            }
    
    with open(KNOWN_MICS_FILE) as f:
        mic_to_name = {m['mic']: m['name'] for m in json.load(f)}

    # 2. Update Tokenizer
    with open(TOK_PATH) as f:
        tok = json.load(f)
    vocab = tok['model']['vocab']
    added = 0
    for s in sedol_data:
        t = s.lower()
        if t not in vocab:
            vocab[t] = max(int(v) for v in vocab.values()) + 1
            added += 1
    if added:
        with open(TOK_PATH, 'w') as f:
            json.dump(tok, f, indent=2, ensure_ascii=False)
        print(f"Added {added} SEDOL tokens.")

    # 3. Load ONNX and Resize
    model = onnx.load(ONNX_PATH)
    target_init = None
    for i, init in enumerate(model.graph.initializer):
        if init.name == 'embeddings.word_embeddings.weight':
            target_init = init
            init_idx = i
            W = numpy_helper.to_array(init).copy()
            max_id = max(int(v) for v in vocab.values()) + 1
            if max_id > W.shape[0]:
                W = np.concatenate([W, np.zeros((max_id - W.shape[0], 384), dtype=np.float32)], axis=0)
            break

    tokenizer = AutoTokenizer.from_pretrained(MIC_MODEL)

    # 4. Triple-Anchor Embedding
    for sedol, info in sedol_data.items():
        # A. Semantic Sources
        sources = [
            info['name'],                        # Specific listing name
            info['isin'],                        # Parent ISIN (we use the ISIN token itself)
            mic_to_name.get(info['mic'], info['mic']) # Exchange name
        ]
        
        # B. Calculate Weighted Blend
        # We give the specific Name more weight (50%) to ensure separation from other listings
        weights = [0.5, 0.25, 0.25]
        semantic_vecs = []
        for src in sources:
            ids = tokenizer.encode(src)[1:-1]
            if ids:
                semantic_vecs.append(W[ids].mean(axis=0))
            else:
                semantic_vecs.append(np.zeros(384))
        
        blend = np.zeros(384, dtype=np.float32)
        for w, v in zip(weights, semantic_vecs):
            blend += w * v
            
        # C. Project and Normalize
        semantic_shell = np.zeros(384, dtype=np.float32)
        semantic_shell[NON_VAC] = blend[NON_VAC]
        norm = np.linalg.norm(semantic_shell)
        if norm > 1e-8: semantic_shell /= norm

        # D. Composite Fingerprint
        # Order: [MIC, ISIN, SEDOL, TICKER]
        final_emb = np.zeros(384, dtype=np.float32)
        final_emb[VAC_DIMS] = [MIC_BIAS, ISIN_BIAS, SEDOL_STRENGTH, 0.0]
        final_emb += 1.0 * semantic_shell

        tid = vocab.get(sedol.lower())
        if tid: W[tid] = final_emb

        print(f"SEDOL: {sedol.upper():10} -> {info['name']} (Anchored to {info['isin']} + {info['mic']})")

    # 5. Save
    new_init = numpy_helper.from_array(W, name=target_init.name)
    model.graph.initializer[init_idx].CopyFrom(new_init)
    onnx.save(model, ONNX_PATH)
    print("\nModel updated with anchored SEDOL embeddings.")

if __name__ == '__main__':
    main()
