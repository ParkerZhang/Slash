#!/usr/bin/env python3
"""
Embed all MIC codes from known_mics.json into the ONNX model.

Sets vacuum-dimension vectors using the tight2x approach:
  fam[VAC_DIMS] = fingerprint (family prototype)
  β = 2 × ‖fam‖
  emb = fam + β · phrase_dir

Also rebuilds family_vac.json re-running build_embed_index.py.

Usage:
  python3 embed_mic.py                         # full: add new tokens + embed
  python3 embed_mic.py --embed-only            # skip new-token addition
  python3 embed_mic.py --rebuild-index-only    # only rebuild family_vac / vocab_embed
"""

import json, os, sys
import numpy as np
import onnx
from onnx import numpy_helper
from transformers import AutoTokenizer

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MIC_MODEL = os.path.join(SCRIPTS_DIR, 'modelFiles', 'Xenova', 'all-MiniLM-L6-v2-mic')
KNOWN_MICS = os.path.join(SCRIPTS_DIR, 'known_mics.json')
ONNX_PATH = os.path.join(MIC_MODEL, 'onnx', 'model.onnx')
TOK_PATH = os.path.join(MIC_MODEL, 'tokenizer.json')

VAC_DIMS = [18, 62, 28, 245]
NON_VAC = [d for d in range(384) if d not in VAC_DIMS]

# Compact MIC subtypes — override fingerprint here for special cases
MIC_FINGERPRINTS = {
    'default': [0.30, 0.08, 0.08, 0.08],
    'XTSE':    [0.30, 0.10, 0.04, 0.06],
}


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MIC_MODEL)


def load_onnx():
    model = onnx.load(ONNX_PATH)
    for i, init in enumerate(model.graph.initializer):
        if init.name == 'embeddings.word_embeddings.weight':
            W = numpy_helper.to_array(init).copy()
            return model, i, W
    raise RuntimeError('embeddings.word_embeddings.weight not found')


def save_onnx(model, init_idx, W):
    new_init = numpy_helper.from_array(W, name='embeddings.word_embeddings.weight')
    model.graph.initializer[init_idx].CopyFrom(new_init)
    onnx.save(model, ONNX_PATH)
    print('  ONNX saved.')


def add_missing_tokens(mic_entries):
    """Add MIC codes to tokenizer vocab if not present."""
    with open(TOK_PATH) as f:
        tok = json.load(f)
    vocab = tok['model']['vocab']
    max_id = max(int(v) for v in vocab.values())
    next_id = max_id + 1
    added = 0
    for entry in mic_entries:
        code = entry['mic']
        for variant in {code.lower()}:
            if variant not in vocab:
                vocab[variant] = next_id
                next_id += 1
                added += 1
    if added:
        tok['model']['vocab'] = vocab
        with open(TOK_PATH, 'w') as f:
            json.dump(tok, f, indent=2, ensure_ascii=False)
        print(f'  Added {added} new token variants to tokenizer')
    return added


def set_mic_embeddings(W, mic_entries):
    """Set vacuum-dimension embeddings for each MIC code."""
    tokenizer = load_tokenizer()
    with open(TOK_PATH) as f:
        vocab = json.load(f)['model']['vocab']
    set_count = 0
    for entry in mic_entries:
        code = entry['mic']
        name = entry.get('name', '')
        fingerprint = MIC_FINGERPRINTS.get(code, MIC_FINGERPRINTS['default'])
        fam = np.zeros(384, dtype=np.float32)
        fam[VAC_DIMS] = fingerprint
        fn = float(np.linalg.norm(fam))
        beta = fn * 2
        # Semantic direction from exchange name
        ids = tokenizer.encode(name)[1:-1]
        if not ids:
            print(f'  SKIP {code}: "{name}" tokenized to nothing')
            continue
        phrase = W[ids].mean(axis=0)
        pdir = np.zeros(384, dtype=np.float32)
        pdir[NON_VAC] = phrase[NON_VAC]
        pn = float(np.linalg.norm(pdir))
        if pn < 1e-8:
            print(f'  SKIP {code}: phrase dir norm=0')
            continue
        pdir /= pn
        emb = fam + beta * pdir
        for variant in {code.lower()}:
            tid = int(vocab.get(variant, -1))
            if tid < 0:
                continue
            W[tid] = emb.astype(np.float32)
            set_count += 1
        tokens_used = tokenizer.tokenize(name)
        print(f'  {code:>6s} vac={fingerprint} ← "{name}" [{", ".join(tokens_used)}]')
    return set_count


def rebuild_index():
    """Rebuild family_vac.json and embed_weight by calling build_embed_index."""
    build_script = os.path.join(SCRIPTS_DIR, 'build_embed_index.py')
    if os.path.exists(build_script):
        print('  Rebuilding vocab index...')
        import subprocess
        result = subprocess.run([sys.executable, build_script], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            print(f'    {line}')
        if result.returncode != 0:
            print(f'  WARN: build_embed_index.py exited {result.returncode}')
            if result.stderr:
                for line in result.stderr.splitlines():
                    print(f'    {line}')
    else:
        print(f'  SKIP: {build_script} not found')


def main():
    args = set(sys.argv[1:])
    embed_only = '--embed-only' in args
    index_only = '--rebuild-index-only' in args

    with open(KNOWN_MICS) as f:
        mic_entries = json.load(f)

    codes = [e['mic'] for e in mic_entries]
    print(f'known_mics.json: {len(codes)} entries: {", ".join(codes)}')

    if not index_only:
        if not embed_only:
            added = add_missing_tokens(mic_entries)
        else:
            added = 0
        # Resize ONNX if tokens were added
        model, init_idx, W = load_onnx()
        orig_rows = W.shape[0]
        with open(TOK_PATH) as f:
            vocab = json.load(f)['model']['vocab']
        new_rows = int(vocab.get('__length_hint', max(int(v) for v in vocab.values()) + 1))
        # Actually count unique IDs in vocab
        max_vocab_id = max(int(v) for v in vocab.values()) + 1
        if max_vocab_id > orig_rows:
            std = float(W.std())
            extra = max_vocab_id - orig_rows
            new_vecs = np.random.randn(extra, 384).astype(np.float32) * std * 0.1
            W = np.concatenate([W, new_vecs], axis=0)
            print(f'  Resized ONNX: [{orig_rows}, 384] → [{W.shape[0]}, 384]')
        set_count = set_mic_embeddings(W, mic_entries)
        if set_count:
            save_onnx(model, init_idx, W)
        else:
            print('  No embeddings set.')

    rebuild_index()
    print('Done.')


if __name__ == '__main__':
    main()
