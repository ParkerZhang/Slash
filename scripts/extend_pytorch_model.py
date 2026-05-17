#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np

def extend_module(module, current_dim, new_dim):
    """
    Recursively expands Linear and LayerNorm modules from current_dim to new_dim.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if child.in_features == current_dim:
                # Expand Weight [Out, Dim] -> [Out, Dim+1]
                old_w = child.weight.data
                padding = torch.zeros((old_w.shape[0], 1), device=old_w.device)
                new_w = torch.cat([old_w, padding], dim=1)
                
                # Create new module and copy parameters
                new_linear = nn.Linear(new_dim, child.out_features).to(child.weight.device)
                new_linear.weight.data = new_w
                if child.bias is not None:
                    new_linear.bias.data = child.bias.data
                
                setattr(module, name, new_linear)
                print(f"Expanded Linear: {name} [{current_dim} -> {new_dim}]")

        elif isinstance(child, nn.LayerNorm):
            if child.normalized_shape == (current_dim,):
                # Expand gamma and beta
                gamma = child.weight.data
                beta = child.bias.data
                
                new_gamma = torch.cat([gamma, torch.tensor([1.0], device=gamma.device)])
                new_beta = torch.cat([beta, torch.tensor([0.0], device=beta.device)])
                
                new_ln = nn.LayerNorm(new_dim).to(child.weight.device)
                new_ln.weight.data = new_gamma
                new_ln.bias.data = new_beta
                
                setattr(module, name, new_ln)
                print(f"Expanded LayerNorm: {name} [{current_dim} -> {new_dim}]")
        
        # Recurse
        extend_module(child, current_dim, new_dim)

def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    output_dir = "./modelFiles/Xenova/all-MiniLM-L6-v2-pytorch-ext"
    
    print(f"Loading model: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    current_dim = config.hidden_size # 384
    new_dim = current_dim + 1
    print(f"Targeting Expansion: {current_dim} -> {new_dim}")

    # 1. Expand Embeddings (Approach 1: Mean)
    embeddings = model.get_input_embeddings()
    old_weight = embeddings.weight.data # [Vocab, 384]
    mu = old_weight.mean(dim=1, keepdim=True) # [Vocab, 1]
    new_emb_weight = torch.cat([old_weight, mu], dim=1)
    
    # Create new embedding layer
    new_embeddings = nn.Embedding(old_weight.shape[0], new_dim)
    new_embeddings.weight.data = new_emb_weight
    model.set_input_embeddings(new_embeddings)
    print("Applied Approach 1 (Mean) to embeddings.")

    # 2. Expand all Linear/LN layers recursively
    extend_module(model, current_dim, new_dim)

    # 3. Save PyTorch Model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Extended PyTorch model saved to: {output_dir}")

    # 4. Optional: Export to ONNX
    print("Exporting to ONNX...")
    dummy_input = torch.ones((1, 10), dtype=torch.long)
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    torch.onnx.export(
        model, 
        (dummy_input,), 
        onnx_path, 
        export_params=True, 
        opset_version=12, 
        do_constant_folding=True, 
        input_names=['input_ids'], 
        output_names=['output'],
        dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}}
    )
    print(f"Exported ONNX model to: {onnx_path}")

if __name__ == '__main__':
    main()
