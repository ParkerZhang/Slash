#!/usr/bin/env python3
import json
import os
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

def get_model_outputs(model_path, input_ids):
    """Runs the ONNX model and returns the output tensor."""
    session = ort.InferenceSession(model_path)
    
    # Prepare input. MiniLM usually expects 'input_ids' and 'attention_mask'
    # We'll create a simple mask of 1s for the length of the input
    input_ids_array = np.array([input_ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids_array, dtype=np.int64)
    
    # The input names might vary, let's check the model inputs
    input_nodes = [node.name for node in session.get_inputs()]
    
    # Most MiniLM ONNX models expect: input_ids, attention_mask, token_type_ids
    inputs = {}
    for name in input_nodes:
        if 'input_ids' in name:
            inputs[name] = input_ids_array
        elif 'attention_mask' in name:
            inputs[name] = attention_mask
        elif 'token_type_ids' in name:
            inputs[name] = np.zeros_like(input_ids_array, dtype=np.int64)

    outputs = session.run(None, inputs)
    # We usually care about the last hidden state or pooled output
    # Returning the first output tensor for comparison
    return np.array(outputs[0])

def test_parity(base_path, extended_path, tokenizer_path):
    """Compares outputs of base and extended models for a set of inputs."""
    print(f"Testing Parity:\nBase: {base_path}\nExtended: {extended_path}\n")

    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    
    vocab = tokenizer_data['model']['vocab']
    
    # Test Cases:
    # 1. Single tokens (representing common entities)
    # 2. Short phrases (existing tokens)
    # 3. Long sentences (existing tokens)
    # 4. Edge cases (special tokens)
    
    test_sets = {
        "Single Tokens": [
            ["hello"], 
            ["world"], 
            ["the"], 
            ["transformer"]
        ],
        "Common Phrases": [
            ["the", "quick", "brown", "fox"],
            ["hello", "how", "are", "you"],
            ["this", "is", "a", "test"]
        ],
        "Special Tokens": [
            ["[CLS]"], 
            ["[SEP]"], 
            ["[PAD]"]
        ]
    }

    # Convert text tokens to IDs using the vocab
    def tokenize(texts):
        ids = []
        for text in texts:
            # Find a token in vocab that matches the text (simplified)
            # In a real scenario, we'd use the actual tokenizer.json logic
            token_id = None
            for t, tid in vocab.items():
                if t == text:
                    token_id = int(tid)
                    break
            if token_id is None:
                # Fallback: try to find if any vocab key contains the text or vice versa
                for t, tid in vocab.items():
                    if text in t:
                        token_id = int(tid)
                        break
            if token_id is None:
                continue # skip tokens not in vocab
            ids.append(token_id)
        return ids

    overall_pass = True
    
    for category, cases in test_sets.items():
        print(f"\n--- Category: {category} ---")
        for case in cases:
            ids = tokenize(case)
            if not ids:
                continue
                
            try:
                base_out = get_model_outputs(base_path, ids)
                ext_out = get_model_outputs(extended_path, ids)
                
                # Calculate Cosine Similarity and MSE
                # Flatten if output is multidimensional
                b_flat = base_out.flatten()
                e_flat = ext_out.flatten()
                
                # The dimensions might differ (384 vs 385), 
                # so we only compare the first 384 dims if necessary
                min_dim = min(b_flat.shape[0], e_flat.shape[0])
                b_crop = b_flat[:min_dim]
                e_crop = e_flat[:min_dim]
                
                mse = np.mean((b_crop - e_crop)**2)
                cos_sim = np.dot(b_crop, e_crop) / (np.linalg.norm(b_crop) * np.linalg.norm(e_crop))
                
                is_pass = mse < 1e-5 and cos_sim > 0.9999
                status = "PASS" if is_pass else "FAIL"
                if not is_pass: overall_pass = False
                
                print(f"Input: {case} -> {status} (MSE: {mse:.2e}, CosSim: {cos_sim:.6f})")
            except Exception as e:
                print(f"Input: {case} -> ERROR: {e}")
                overall_pass = False

    print("\n" + "="*30)
    print(f"OVERALL PARITY RESULT: {'SUCCESS' if overall_pass else 'FAILURE'}")
    print("="*30)

if __name__ == '__main__':
    # Paths are relative to project root
    base_model = './modelFiles/Xenova/all-MiniLM-L6-v2/onnx/model.onnx'
    ext_model = './modelFiles/Xenova/all-MiniLM-L6-v2-mic-ext/onnx/model.onnx'
    tok_path = './modelFiles/Xenova/all-MiniLM-L6-v2/tokenizer.json'
    
    if not os.path.exists(base_model) or not os.path.exists(ext_model):
        print("Error: One or both models not found. Run extension script first.")
    else:
        test_parity(base_model, ext_model, tok_path)
