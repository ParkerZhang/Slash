#!/usr/bin/env python3
import json, os, numpy as np, onnx, csv
from onnx import numpy_helper
from transformers import AutoTokenizer

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MIC_MODEL = os.path.join(SCRIPTS_DIR, 'modelFiles', 'Xenova', 'all-MiniLM-L6-v2-mic')
ONNX_PATH = os.path.join(MIC_MODEL, 'onnx', 'model.onnx')
TOK_PATH = os.path.join(MIC_MODEL, 'tokenizer.json')
SAMPLE_CSV = os.path.join(SCRIPTS_DIR, 'sample.csv')

VAC_DIMS = [18, 62, 28, 245]

def cos(a, b):
    a, b = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return 0.0
    return float(np.dot(a, b) / (na * nb))

def token_id(vocab, token):
    for key in (token, token.lower(), token.upper()):
        if key in vocab:
            return int(vocab[key])
    return None

def main():
    with open(TOK_PATH) as f: tok_data = json.load(f)
    vocab = tok_data['model']['vocab']
    id_to_token = {int(v): k for k, v in vocab.items()}
    
    model = onnx.load(ONNX_PATH)
    for init in model.graph.initializer:
        if init.name == 'embeddings.word_embeddings.weight':
            W = numpy_helper.to_array(init)
            break
    W_norm = W / np.linalg.norm(W, axis=1, keepdims=True)
    
    rows = []
    with open(SAMPLE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader: rows.append(row)

    markers = {
        'MIC':    W[token_id(vocab, '#MIC')][VAC_DIMS],
        'ISIN':   W[token_id(vocab, 'ISIN')][VAC_DIMS],
        'SEDOL':  W[token_id(vocab, 'SEDOL')][VAC_DIMS],
        'TICKER': W[token_id(vocab, 'TICKER')][VAC_DIMS]
    }

    print(f"{'Type':8} | {'Token':12} | {'Class':7} | {'ISIN Hit?':10} | {'Rank'} | {'Sim'}")
    print("-" * 65)

    for fam in ['SEDOL', 'TICKER']:
        passes = 0
        total = 0
        for row in rows:
            token = row[fam].strip().lower()
            parent_isin = row['ISIN'].strip().lower()
            tid = vocab.get(token)
            if tid is None: continue

            total += 1
            vec = W[tid]
            
            # Classification
            best_class = None
            best_sim = -1.0
            for m_name, m_vec in markers.items():
                s = cos(vec[VAC_DIMS], m_vec)
                if s > best_sim:
                    best_sim, best_class = s, m_name
            
            # Structural Hit Check
            query_vec = W_norm[tid]
            sims = np.dot(W_norm, query_vec)
            top_indices = np.argsort(sims)[::-1][:31]
            
            hit_rank = -1
            hit_sim = 0
            for rank, idx in enumerate(top_indices):
                t_str = id_to_token.get(idx, "").lower()
                if t_str == parent_isin:
                    hit_rank = rank
                    hit_sim = sims[idx]
                    break
            
            isin_status = "YES" if hit_rank != -1 else "NO"
            print(f"{fam:8} | {token.upper():12} | {best_class:7} | {isin_status:10} | {hit_rank:4} | {hit_sim:.4f}")

    print("-" * 65)

if __name__ == '__main__':
    main()
