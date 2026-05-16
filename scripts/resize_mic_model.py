#!/usr/bin/env python3
import json, os, shutil, sys
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto


def main():
    if len(sys.argv) < 2:
        print('Usage: resize_mic_model.py <tokens_json_path> [project_dir]', file=sys.stderr)
        sys.exit(1)

    tokens_path = sys.argv[1]
    project_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.getcwd()

    if not os.path.isfile(tokens_path):
        print(f'Error: tokens file not found: {tokens_path}', file=sys.stderr)
        sys.exit(1)

    with open(tokens_path) as f:
        tokens = json.load(f)

    if not isinstance(tokens, list) or not tokens:
        print('No tokens to add.')
        return

    base_model = os.path.join(project_dir, 'modelFiles', 'Xenova', 'all-MiniLM-L6-v2')
    mic_model = os.path.join(project_dir, 'modelFiles', 'Xenova', 'all-MiniLM-L6-v2-mic')

    print(f'Resizing MIC model for {len(tokens)} new tokens...')

    # Create MIC model from base if needed
    if not os.path.isdir(mic_model):
        if not os.path.isdir(base_model):
            print(f'Error: base model not found at {base_model}', file=sys.stderr)
            sys.exit(1)
        print('Copying base model to MIC model path...')
        shutil.copytree(base_model, mic_model, dirs_exist_ok=True)

    # --- Update tokenizer ---
    tok_path = os.path.join(mic_model, 'tokenizer.json')
    with open(tok_path) as f:
        tok = json.load(f)

    vocab = tok['model']['vocab']
    max_id = max(int(v) for v in vocab.values())
    next_id = max_id + 1

    new_tokens = []
    for token in tokens:
        token_str = str(token).strip()
        if not token_str:
            continue
        v = str(token_str)
        if v not in vocab:
            vocab[v] = next_id
            new_tokens.append((v, next_id))
            next_id += 1

    tok['model']['vocab'] = vocab
    with open(tok_path, 'w') as f:
        json.dump(tok, f, indent=2, ensure_ascii=False)

    added = len(new_tokens)
    print(f'Added {added} new token(s) to tokenizer vocab (total: {len(vocab)}).')

    if added == 0:
        print('No new tokens to add to ONNX model.')
        return

    # --- Resize ONNX embedding matrix ---
    onnx_path = os.path.join(mic_model, 'onnx', 'model.onnx')
    model = onnx.load(onnx_path)

    for i, init in enumerate(model.graph.initializer):
        if init.name == 'embeddings.word_embeddings.weight':
            weight = numpy_helper.to_array(init)
            orig_rows = weight.shape[0]
            emb_dim = weight.shape[1]
            new_rows = orig_rows + added

            std = float(weight.std())
            new_vectors = np.random.randn(added, emb_dim).astype(np.float32) * std * 0.1
            weight = np.concatenate([weight, new_vectors], axis=0)

            new_init = numpy_helper.from_array(weight, name=init.name)
            model.graph.initializer[i].CopyFrom(new_init)

            print(f'Resized ONNX embedding matrix: [{orig_rows}, {emb_dim}] → [{new_rows}, {emb_dim}]')
            break
    else:
        print('Error: embeddings.word_embeddings.weight not found in ONNX model', file=sys.stderr)
        sys.exit(1)

    onnx.save(model, onnx_path)
    print(f'ONNX model saved to: {onnx_path}')
    print(f'MIC model: {mic_model}')
    print(f'New tokens: {len(new_tokens)} (IDs {new_tokens[0][1]} to {new_tokens[-1][1]})')


if __name__ == '__main__':
    main()
