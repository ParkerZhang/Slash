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

# Dimension mapping
VAC_DIMS = [18, 62, 28, 245]
NON_VAC = [d for d in range(384) if d not in VAC_DIMS]

# Prototype to match the classifier's 'ISIN' gate
# Norm ~0.396
ISIN_PROTO_VEC = np.array([0.15, 0.30, 0.15, 0.15], dtype=np.float32)

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

    # 1. Update Tokenizer
    with open(TOK_PATH) as f:
        tok = json.load(f)
    vocab = tok['model']['vocab']
    
    added = 0
    for isin in isin_data:
        t = isin.lower()
        if t not in vocab:
            vocab[t] = max(int(v) for v in vocab.values()) + 1
            added += 1
    if added:
        with open(TOK_PATH, 'w') as f:
            json.dump(tok, f, indent=2, ensure_ascii=False)
        print(f"Added {added} ISIN tokens.")

    # 2. Load and Resize ONNX
    model = onnx.load(ONNX_PATH)
    target_init = None
    for i, init in enumerate(model.graph.initializer):
        if init.name == 'embeddings.word_embeddings.weight':
            target_init = init
            init_idx = i
            W = numpy_helper.to_array(init).copy()
            max_id = max(int(v) for v in vocab.values()) + 1
            if max_id > W.shape[0]:
                extra = max_id - W.shape[0]
                W = np.concatenate([W, np.zeros((extra, 384), dtype=np.float32)], axis=0)
                print(f"Resized ONNX matrix to {W.shape[0]} rows.")
            break

    tokenizer = AutoTokenizer.from_pretrained(MIC_MODEL)

    # 3. Embed ISINs
    for isin, info in isin_data.items():
        # A. Get Semantic Base (Company Name)
        core_name = clean_company_name(info['names'][0])
        core_ids = tokenizer.encode(core_name)[1:-1]
        core_vec = W[core_ids].mean(axis=0) if core_ids else np.zeros(384)

        # B. Get Primary MIC (Bias)
        primary_mic = info['mics'][0]
        mic_full_name = mic_to_name.get(primary_mic, primary_mic)
        mic_ids = tokenizer.encode(mic_full_name)[1:-1]
        mic_vec = W[mic_ids].mean(axis=0) if mic_ids else np.zeros(384)

        # C. Blend: 70% Company, 30% Market (380-dim only)
        blend = (core_vec * 0.7) + (mic_vec * 0.3)
        semantic_shell = np.zeros(384, dtype=np.float32)
        semantic_shell[NON_VAC] = blend[NON_VAC]
        
        # Normalize the shell to unit length
        norm_s = np.linalg.norm(semantic_shell)
        if norm_s > 1e-8:
            semantic_shell /= norm_s

        # D. Assemble Final Vector
        # E = Fingerprint + (Weight * Shell)
        # We use a weight of 1.0 for the shell to ensure strong semantic identity
        final_emb = np.zeros(384, dtype=np.float32)
        final_emb[VAC_DIMS] = ISIN_PROTO_VEC
        final_emb += 1.0 * semantic_shell

        tid = vocab.get(isin.lower())
        if tid:
            W[tid] = final_emb
            print(f"Embedded {isin:12s} | Class: ISIN | Context: {core_name} ({primary_mic})")

    # 4. Save
    new_init = numpy_helper.from_array(W, name=target_init.name)
    model.graph.initializer[init_idx].CopyFrom(new_init)
    onnx.save(model, ONNX_PATH)
    print("\nModel saved successfully with calibrated ISIN embeddings.")

if __name__ == '__main__':
    main()
