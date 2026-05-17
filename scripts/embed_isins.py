#!/usr/bin/env python3
import json, os, sys
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

VAC_DIMS = [18, 62, 28, 245]
NON_VAC = [d for d in range(384) if d not in VAC_DIMS]
ISIN_FINGERPRINT = [0.08, 0.30, 0.08, 0.08]
MIC_FINGERPRINT  = [0.30, 0.08, 0.08, 0.08]

def main():
    if not os.path.exists(MAPPING_FILE):
        print(f"Error: {MAPPING_FILE} not found. Run the extraction first.")
        return

    with open(MAPPING_FILE) as f:
        isin_data = json.load(f)

    with open(KNOWN_MICS_FILE) as f:
        known_mics_list = json.load(f)
    mic_to_name = {m['mic']: m['name'] for m in known_mics_list}

    # 1. Prepare all tokens to be added
    all_isins = list(isin_data.keys())
    all_mics = set()
    for info in isin_data.values():
        all_mics.update(info['mics'])
    
    # 2. Add tokens to tokenizer
    with open(TOK_PATH) as f:
        tok = json.load(f)
    vocab = tok['model']['vocab']
    
    added_tokens = []
    def add_token(t):
        nonlocal added_tokens
        t_low = t.lower()
        if t_low not in vocab:
            max_id = max(int(v) for v in vocab.values())
            vocab[t_low] = max_id + 1
            added_tokens.append(t_low)
            return True
        return False

    for isin in all_isins: add_token(isin)
    for mic in all_mics: add_token(mic)

    if added_tokens:
        with open(TOK_PATH, 'w') as f:
            json.dump(tok, f, indent=2, ensure_ascii=False)
        print(f"Added {len(added_tokens)} new tokens.")

    # 3. Load ONNX and Resize
    model = onnx.load(ONNX_PATH)
    init_idx = -1
    for i, init in enumerate(model.graph.initializer):
        if init.name == 'embeddings.word_embeddings.weight':
            init_idx = i
            W = numpy_helper.to_array(init).copy()
            orig_rows = W.shape[0]
            max_vocab_id = max(int(v) for v in vocab.values()) + 1
            if max_vocab_id > orig_rows:
                std = float(W.std())
                extra = max_vocab_id - orig_rows
                new_vecs = np.random.randn(extra, 384).astype(np.float32) * std * 0.1
                W = np.concatenate([W, new_vecs], axis=0)
                print(f"Resized ONNX: {orig_rows} -> {W.shape[0]}")
            break

    tokenizer = AutoTokenizer.from_pretrained(MIC_MODEL)

    def get_semantic_vec(texts):
        vecs = []
        for txt in texts:
            ids = tokenizer.encode(txt)[1:-1]
            if ids: vecs.append(W[ids].mean(axis=0))
        if not vecs: return None
        avg = np.mean(vecs, axis=0)
        pdir = np.zeros(384, dtype=np.float32)
        pdir[NON_VAC] = avg[NON_VAC]
        pn = float(np.linalg.norm(pdir))
        if pn > 1e-8: pdir /= pn
        return pdir

    # 4. Embed MICs (if they don't have good embeddings yet or just to be sure)
    for mic in all_mics:
        name = mic_to_name.get(mic, mic) # Fallback to code if name unknown
        pdir = get_semantic_vec([name])
        if pdir is not None:
            v_proto = np.zeros(384, dtype=np.float32)
            v_proto[VAC_DIMS] = MIC_FINGERPRINT
            fn = float(np.linalg.norm(v_proto))
            emb = v_proto + (fn * 2) * pdir
            tid = vocab.get(mic.lower())
            W[tid] = emb.astype(np.float32)
            print(f"Embedded MIC: {mic}")

    # 5. Embed ISINs
    for isin, info in isin_data.items():
        # Combine names and MIC names/codes for semantic direction
        semantic_sources = info['names'][:]
        for m in info['mics']:
            semantic_sources.append(mic_to_name.get(m, m))
        
        pdir = get_semantic_vec(semantic_sources)
        if pdir is not None:
            v_proto = np.zeros(384, dtype=np.float32)
            v_proto[VAC_DIMS] = ISIN_FINGERPRINT
            fn = float(np.linalg.norm(v_proto))
            emb = v_proto + (fn * 2) * pdir
            tid = vocab.get(isin.lower())
            W[tid] = emb.astype(np.float32)
            print(f"Embedded ISIN: {isin} (sources: {len(semantic_sources)})")

    # 6. Save
    new_init = numpy_helper.from_array(W, name='embeddings.word_embeddings.weight')
    model.graph.initializer[init_idx].CopyFrom(new_init)
    onnx.save(model, ONNX_PATH)
    print("Done.")

if __name__ == '__main__':
    main()
