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
REGISTRY_JSON = os.path.join(SCRIPTS_DIR, 'registry.json')
KNOWN_MICS_FILE = os.path.join(SCRIPTS_DIR, 'known_mics.json')

# Dimension mapping: [MIC, ISIN, SEDOL, TICKER]
VAC_DIMS = [18, 62, 28, 245]
NON_VAC = [d for d in range(384) if d not in VAC_DIMS]

# Fingerprints (Orthogonal One-Hot)
FP_MIC    = [0.40, 0.00, 0.00, 0.00]
FP_ISIN   = [0.10, 0.40, 0.00, 0.00] # Minor MIC bias
FP_SEDOL  = [0.15, 0.10, 0.40, 0.00] # Market + ISIN link
FP_TICKER = [0.15, 0.10, 0.10, 0.40] # Market + ISIN + SEDOL link

def needs_added_token(content):
    return bool(content) and (not content.isalnum() or len(content) > 5)

def register_added_token(tok, vocab, content, token_key):
    if not needs_added_token(content):
        return 0
    tid = vocab.get(token_key.lower())
    if tid is None:
        return 0
    added_tokens = tok.setdefault('added_tokens', [])
    existing = {entry.get('content') for entry in added_tokens}
    if content in existing:
        return 0
    added_tokens.append({
        'id': int(tid),
        'content': content,
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': False,
    })
    return 1

def prune_added_tokens(tok):
    kept = []
    removed = 0
    for entry in tok.get('added_tokens', []):
        if entry.get('special') or needs_added_token(entry.get('content', '')):
            kept.append(entry)
        else:
            removed += 1
    tok['added_tokens'] = kept
    return removed

def register_identifier_token(tok, vocab, value):
    if not value:
        return 0
    raw = value.strip()
    if not raw:
        return 0
    lower = raw.lower()
    changed = register_added_token(tok, vocab, lower, lower)
    upper = raw.upper()
    if upper != lower:
        changed += register_added_token(tok, vocab, upper, lower)
    return changed

def main():
    if not os.path.exists(REGISTRY_JSON):
        print("Error: registry.json not found. Run sync_registry.py first.")
        return

    with open(REGISTRY_JSON) as f:
        registry = json.load(f)
    entities = registry['entities']
    
    with open(KNOWN_MICS_FILE) as f:
        mic_to_name = {m['mic']: m['name'] for m in json.load(f)}

    # 1. Update Tokenizer (Lowercase Only)
    with open(TOK_PATH) as f:
        tok = json.load(f)
    vocab = tok['model']['vocab']
    
    added = 0
    added_token_entries = prune_added_tokens(tok)
    for ent_id, ent in entities.items():
        # Add ISIN
        if 'isin' in ent:
            t = ent['isin'].lower()
            if t not in vocab:
                vocab[t] = max(int(v) for v in vocab.values()) + 1
                added += 1
            added_token_entries += register_identifier_token(tok, vocab, ent['isin'])
        
        # Add Listings (SEDOL, TICKER)
        for li in ent['listings']:
            for key in ['sedol', 'ticker']:
                val = li.get(key)
                if val:
                    t = val.lower()
                    if t not in vocab:
                        vocab[t] = max(int(v) for v in vocab.values()) + 1
                        added += 1
                    added_token_entries += register_identifier_token(tok, vocab, val)
    
    if added or added_token_entries:
        with open(TOK_PATH, 'w') as f:
            json.dump(tok, f, indent=2, ensure_ascii=False)
        print(f"Added {added} new vocab tokens and {added_token_entries} added-token entries to tokenizer.")

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
                W = np.concatenate([W, np.zeros((max_id - W.shape[0], 384), dtype=np.float32)], axis=0)
                print(f"Resized ONNX to {W.shape[0]} rows.")
            break

    tokenizer = AutoTokenizer.from_pretrained(MIC_MODEL)

    def set_vec(token, semantic_sources, weights, fingerprint):
        t_low = token.lower()
        tid = vocab.get(t_low)
        if not tid: return

        # A. Semantic Blend
        blend = np.zeros(384, dtype=np.float32)
        for src, weight in zip(semantic_sources, weights):
            if not src: continue
            ids = tokenizer.encode(src.lower())[1:-1]
            if ids:
                blend += weight * W[ids].mean(axis=0)
        
        # B. Project and Normalize
        semantic_shell = np.zeros(384, dtype=np.float32)
        semantic_shell[NON_VAC] = blend[NON_VAC]
        norm = np.linalg.norm(semantic_shell)
        if norm > 1e-8: semantic_shell /= norm

        # C. Assemble
        final_emb = np.zeros(384, dtype=np.float32)
        final_emb[VAC_DIMS] = fingerprint
        final_emb += 1.0 * semantic_shell
        W[tid] = final_emb

    # 3. Embed Registry
    print("Embedding entities...")
    for ent_id, ent in entities.items():
        company_name = ent['name']
        isin = ent.get('isin')
        
        # A. Embed ISIN (if present)
        if isin:
            # Semantic: 100% Company Name
            set_vec(isin, [company_name], [1.0], FP_ISIN)
        
        # B. Embed Listings
        for li in ent['listings']:
            mic_name = mic_to_name.get(li['mic'], li['mic'])
            # Specific listing name is the best anchor
            listing_name = li.get('display_name', company_name)
            
            # Embed SEDOL
            if li.get('sedol'):
                # 50% Listing Name, 25% ISIN, 25% MIC
                set_vec(li['sedol'], [listing_name, isin, mic_name], [0.5, 0.25, 0.25], FP_SEDOL)
            
            # Embed TICKER
            if li.get('ticker'):
                # 40% Listing Name, 20% ISIN, 20% MIC, 20% SEDOL
                set_vec(li['ticker'], [listing_name, isin, mic_name, li.get('sedol')], [0.4, 0.2, 0.2, 0.2], FP_TICKER)

    # 4. Save
    new_init = numpy_helper.from_array(W, name=target_init.name)
    model.graph.initializer[init_idx].CopyFrom(new_init)
    onnx.save(model, ONNX_PATH)
    print("\nModel updated with registry embeddings.")

if __name__ == '__main__':
    main()
