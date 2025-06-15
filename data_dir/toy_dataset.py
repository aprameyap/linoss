"""
This script generates a toy dataset of shape (100000, 100, 6) and calculates the signature of depth 4 for the label.
GPU-enabled version.
"""

import os
import torch
import signatory
from process_uea import save_pickle


def generate_toy_dataset(device='cpu', batch_size=1000):
    """Generate toy dataset with signature labels."""
    
    torch.manual_seed(1234)
    depth = 4
    
    print(f"Using device: {device}")
    
    data = torch.randn(100000, 100, 6, device=device)
    data = torch.round(data)
    data = torch.cumsum(data, dim=1)
    
    labels = []
    
    print("Computing signatures...")
    num_batches = (data.shape[0] + batch_size - 1) // batch_size
    
    for i in range(0, data.shape[0], batch_size):
        batch_end = min(i + batch_size, data.shape[0])
        batch_data = data[i:batch_end]
        
        batch_labels = signatory.signature(batch_data, depth)
        labels.append(batch_labels.cpu())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed batch {i // batch_size + 1}/{num_batches}")
    
    labels = torch.cat(labels, dim=0)
    
    data = data.cpu()
    
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return data, labels


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data, labels = generate_toy_dataset(device=device, batch_size=1000)
    
    save_dir = "data_dir/processed/toy/signature"
    os.makedirs(save_dir, exist_ok=True)
    
    save_pickle(data, save_dir + "/data.pkl")
    save_pickle(labels, save_dir + "/labels.pkl")
    
    print(f"Dataset saved to {save_dir}")
