#!/usr/bin/env python3
import json, os, sys, re
import numpy as np
import onnx
from onnx import numpy_helper
from transformers import AutoTokenizer

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MIC_MODEL = os.path.join(SCRIPTS_DIR, 'modelFiles', 'Xenova', 'all-MiniLM-L6-v2-mic')
ONNX_PATH = os.path.join(MIC_MODEL, 'onnx', 'model.onnx')
TOK_PATH = os.path.join(MIC_MODEL, 'tokenizer.json')
MAPPING_FILE = os.path.join(SCRIPTS_DIR, 'isin_mic_mapping.json')
KNOWN_MICS_FILE = os.path.join(SCRIPTS_DIR, 'known_mics.json')

# Dimension mapping: [MIC, ISIN, SEDOL, TICKER]
VAC_DIMS = [18, 62, 28, 245]
NON_VAC = [d for d in range(384) if d not in VAC_DIMS]

# Standard marker strengths from rebuild_mic_model.py
# MIC: 0.20, ISIN: 0.30, SEDOL: 0.30, TICKER: 0.30
ISIN_STRENGTH = 0.40
MIC_BIAS_STRENGTH = 0.10 # Keep it low to prevent misclassification
BRIDGE_STRENGTH = 0.00   # Orthogonal by default

def clean_company_name(name):
    return re.split(r'\s+AT\s+', name, flags=re.IGNORECASE)[0].strip()

def main():
    if not os.path.exists(MAPPING_FILE):
        print("Error: isin_mic_mapping.json not found.")
        return

    with open(MAPPING_FILE) as f:
        isin_data = json.load(f)
    with open(KNOWN_MICS_FILE) as f:
        known_mics_list = json.load(f)
    mic_to_name = {m['mic']: m['name'] for m in known_mics_list}

    # 1. Update Tokenizer (standard)
    with open(TOK_PATH) as f:
        tok = json.load(f)
    vocab = tok['model']['vocab']
    for isin in isin_data:
        t = isin.lower()
        if t not in vocab:
            vocab[t] = max(int(v) for v in vocab.values()) + 1
    with open(TOK_PATH, 'w') as f:
        json.dump(tok, f, indent=2, ensure_ascii=False)

    # 2. Load ONNX
    model = onnx.load(ONNX_PATH)
    target_init = None
    for i, init in enumerate(model.graph.initializer):
        if init.name == 'embeddings.word_embeddings.weight':
            target_init = init
            init_idx = i
            W = numpy_helper.to_array(init).copy()
            break

    tokenizer = AutoTokenizer.from_pretrained(MIC_MODEL)

    # 3. Composite Embedding
    for isin, info in isin_data.items():
        # A. Extract Exact Semantic Location of the Name
        core_name = clean_company_name(info['names'][0])
        core_ids = tokenizer.encode(core_name)[1:-1]
        # Use the mean vector of the existing word embeddings
        name_vec = W[core_ids].mean(axis=0) if core_ids else np.zeros(384)

        # B. Construct the Composite Vacuum Fingerprint
        # Order in VAC_DIMS: [MIC, ISIN, SEDOL, TICKER]
        composite_vac = np.array([
            MIC_BIAS_STRENGTH, # Closest MIC dimension
            ISIN_STRENGTH,     # ISIN dimension
            BRIDGE_STRENGTH,   # SEDOL dimension
            BRIDGE_STRENGTH    # TICKER dimension
        ], dtype=np.float32)

        # C. Assemble the Final Vector
        # We preserve the name's exact coordinates in 380 dims
        final_emb = name_vec.copy().astype(np.float32)
        final_emb[VAC_DIMS] = composite_vac

        tid = vocab.get(isin.lower())
        if tid:
            if tid >= W.shape[0]:
                # Handle resizing if needed
                extra = (tid + 1) - W.shape[0]
                W = np.concatenate([W, np.zeros((extra, 384), dtype=np.float32)], axis=0)
            
            W[tid] = final_emb
            print(f"Composite: {isin.upper():12s} == {core_name} + [ISIN:0.40, MIC:0.10]")

    # 4. Save
    new_init = numpy_helper.from_array(W, name=target_init.name)
    model.graph.initializer[init_idx].CopyFrom(new_init)
    onnx.save(model, ONNX_PATH)
    print("\nModel saved with composite ISIN embeddings.")

if __name__ == '__main__':
    main()
