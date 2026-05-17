#!/usr/bin/env python3
import os
import shutil
import sys
import numpy as np
import onnx
from onnx import numpy_helper
from onnx import shape_inference

def main():
    base_mic_dir = './modelFiles/Xenova/all-MiniLM-L6-v2-mic'
    ext_model_dir = './modelFiles/Xenova/all-MiniLM-L6-v2-mic-ext'
    base_onnx_path = os.path.join(base_mic_dir, 'onnx', 'model.onnx')
    ext_onnx_path = os.path.join(ext_model_dir, 'onnx', 'model.onnx')

    if not os.path.exists(base_onnx_path):
        print(f"Error: Base MIC model not found at {base_onnx_path}")
        sys.exit(1)

    print(f"Creating non-destructive copy to: {ext_model_dir}")
    if os.path.exists(ext_model_dir):
        shutil.rmtree(ext_model_dir)
    shutil.copytree(base_mic_dir, ext_model_dir)

    model = onnx.load(ext_onnx_path)
    
    emb_weight_name = 'embeddings.word_embeddings.weight'
    current_dim = 0
    for init in model.graph.initializer:
        if init.name == emb_weight_name:
            current_dim = numpy_helper.to_array(init).shape[1]
            break
    
    if current_dim == 0:
        print(f"Error: Could not find {emb_weight_name}")
        sys.exit(1)
    
    new_dim = current_dim + 1
    print(f"Expanding dimension: {current_dim} -> {new_dim}")

    # 1. Expand All Weights (Initializers)
    for i, init in enumerate(model.graph.initializer):
        weight = numpy_helper.to_array(init)
        
        if init.name == emb_weight_name:
            mu = np.mean(weight, axis=1, keepdims=True) 
            weight = np.concatenate([weight, mu], axis=1)
        elif weight.ndim == 2 and (weight.shape[0] == current_dim or weight.shape[1] == current_dim):
            if weight.shape[1] == current_dim:
                padding = np.zeros((weight.shape[0], 1), dtype=weight.dtype)
                weight = np.concatenate([weight, padding], axis=1)
            elif weight.shape[0] == current_dim:
                padding = np.zeros((1, weight.shape[1]), dtype=weight.dtype)
                weight = np.concatenate([weight, padding], axis=0)
        elif weight.ndim == 1 and weight.shape[0] == current_dim:
            val = 1.0 if np.mean(np.abs(weight - 1.0)) < np.mean(np.abs(weight)) else 0.0
            weight = np.append(weight, val).astype(weight.dtype)
        
        new_init = numpy_helper.from_array(weight, name=init.name)
        model.graph.initializer[i].CopyFrom(new_init)

    # 2. TOTAL METADATA PURGE
    # We remove all shape-related info to force a clean slate for the shape_inference tool
    print("Performing aggressive metadata purge...")
    while len(model.graph.value_info) > 0:
        model.graph.value_info.pop(0)

    # 3. Update Constant Tensors
    for i, init in enumerate(model.graph.initializer):
        weight = numpy_helper.to_array(init)
        if weight.ndim == 1 and (weight.dtype == np.int64 or weight.dtype == np.int32):
            if np.array_equal(weight, np.array([current_dim], dtype=weight.dtype)):
                weight = np.array([new_dim], dtype=weight.dtype)
                new_init = numpy_helper.from_array(weight, name=init.name)
                model.graph.initializer[i].CopyFrom(new_init)

    # 4. FORCED SHAPE INFERENCE
    # Instead of letting ONNXRuntime guess, we use the ONNX shape_inference tool
    # to rebuild the entire graph's shape data based on the modified initializers.
    print("Running formal ONNX shape inference to rebuild graph metadata...")
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: shape_inference failed: {e}")

    onnx.save(model, ext_onnx_path)
    print(f"Successfully extended and saved to {ext_onnx_path}")

if __name__ == '__main__':
    main()
