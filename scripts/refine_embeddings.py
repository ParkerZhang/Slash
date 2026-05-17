#!/usr/bin/env python3
import json, os, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import onnx
from onnx import numpy_helper

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MIC_MODEL = os.path.join(SCRIPTS_DIR, 'modelFiles', 'Xenova', 'all-MiniLM-L6-v2-mic')
REGISTRY_JSON = os.path.join(SCRIPTS_DIR, 'registry.json')
VOCAB_EMBED = os.path.join(MIC_MODEL, 'vocab_embed.npy')
VOCAB_TOKENS = os.path.join(MIC_MODEL, 'vocab_embed_tokens.json')

# Fingerprint Constants (Orthogonal One-Hot)
VAC_DIMS = [18, 62, 28, 245]
NON_VAC = [d for d in range(384) if d not in VAC_DIMS]

# [MIC, ISIN, SEDOL, TICKER]
FP_ISIN   = [0.10, 0.40, 0.00, 0.00]
FP_SEDOL  = [0.15, 0.10, 0.40, 0.00]
FP_TICKER = [0.15, 0.10, 0.10, 0.40]

def main():
    if not os.path.exists(REGISTRY_JSON):
        print("Error: registry.json not found.")
        return
    if not os.path.exists(VOCAB_EMBED):
        print("Error: vocab_embed.npy not found. Run build_embed_index.py first.")
        return

    # 1. Load Data
    with open(REGISTRY_JSON) as f:
        registry = json.load(f)
    entities = registry['entities']
    
    v_index = np.load(VOCAB_EMBED)
    with open(VOCAB_TOKENS) as f:
        v_tokens = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(MIC_MODEL)
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    onnx_path = os.path.join(MIC_MODEL, 'onnx', 'model.onnx')
    onnx_model = onnx.load(onnx_path)
    for i, init in enumerate(onnx_model.graph.initializer):
        if init.name == 'embeddings.word_embeddings.weight':
            W = numpy_helper.to_array(init).copy()
            init_idx = i
            break
    
    model.resize_token_embeddings(W.shape[0])
    with torch.no_grad():
        model.embeddings.word_embeddings.weight.copy_(torch.from_numpy(W))
    model.eval()

    # 2. Refinement Loop
    top_x = 10
    print(f"Refining entities using manifold alignment (registry source)...\n")

    def refine_token(token, context_text, fingerprint):
        t_low = token.lower()
        tid = tokenizer.convert_tokens_to_ids(t_low)
        if tid == tokenizer.unk_token_id: return

        # A. Contextualized Name Embedding
        with torch.no_grad():
            inputs = tokenizer(context_text, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs).last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            name_emb = ((outputs * mask).sum(dim=1) / mask.sum(dim=1)).squeeze().numpy()
        
        # B. Cluster Centroid
        name_norm = name_emb / np.linalg.norm(name_emb)
        sims = np.dot(v_index, name_norm)
        top_indices = np.argsort(sims)[-top_x:]
        nn_centroid = v_index[top_indices].mean(axis=0)
        
        # C. Project
        target_semantic = (name_norm * 0.6) + (nn_centroid * 0.4)
        semantic_shell = np.zeros(384, dtype=np.float32)
        semantic_shell[NON_VAC] = target_semantic[NON_VAC]
        semantic_shell /= np.linalg.norm(semantic_shell)

        # D. Inject
        final_emb = np.zeros(384, dtype=np.float32)
        final_emb[VAC_DIMS] = fingerprint
        final_emb += semantic_shell
        W[tid] = final_emb

    for ent_id, ent in entities.items():
        name = ent['name']
        
        # Refine ISIN
        if ent.get('isin'):
            refine_token(ent['isin'], name, FP_ISIN)
        
        # Refine Listings
        for li in ent['listings']:
            listing_name = li.get('display_name', name)
            if li.get('sedol'):
                refine_token(li['sedol'], listing_name, FP_SEDOL)
            if li.get('ticker'):
                refine_token(li['ticker'], listing_name, FP_TICKER)
                
        print(f"Refined: {ent_id}")

    # 3. Save
    new_init = numpy_helper.from_array(W, name='embeddings.word_embeddings.weight')
    onnx_model.graph.initializer[init_idx].CopyFrom(new_init)
    onnx.save(onnx_model, onnx_path)
    print("\nRefinement complete.")

if __name__ == '__main__':
    main()
