"""
This module defines the `Dataset` class and functions for generating datasets tailored to different model types.
A `Dataset` object in this module contains three different dataloaders, each providing a specific version of the data
required by different models:

- `raw_dataloaders`: Returns the raw time series data, suitable for recurrent neural networks (RNNs) and structured
  state space models (SSMs).
- `coeff_dataloaders`: Provides the coefficients of an interpolation of the data, used by Neural Controlled Differential
  Equations (NCDEs).
- `path_dataloaders`: Provides the log-signature of the data over intervals, used by Neural Rough Differential Equations
  (NRDEs) and Log-NCDEs.

The module also includes utility functions for processing and generating these datasets, ensuring compatibility with
different model requirements.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict

import torch
import numpy as np

from data_dir.dataloaders import Dataloader
from data_dir.generate_coeffs import calc_coeffs
from data_dir.generate_paths import calc_paths


@dataclass
class Dataset:
    name: str
    raw_dataloaders: Dict[str, Dataloader]
    coeff_dataloaders: Dict[str, Dataloader]
    path_dataloaders: Dict[str, Dataloader]
    data_dim: int
    logsig_dim: int
    intervals: torch.Tensor
    label_dim: int


def jax_to_torch(obj, device=None):
    """Convert JAX arrays to PyTorch tensors when loading from pickle."""
    if device is None:
        device = torch.device('cpu')
    
    if hasattr(obj, 'shape'):  # JAX array or numpy array
        return torch.from_numpy(np.array(obj)).to(device)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(jax_to_torch(item, device) for item in obj)
    else:
        return obj


def batch_calc_paths(data, stepsize, depth, inmemory=True, device=None):
    """Calculate paths in batches to manage memory."""
    if device is None:
        device = data.device if hasattr(data, 'device') else torch.device('cpu')
    
    N = len(data)
    batchsize = 128
    num_batches = N // batchsize
    remainder = N % batchsize
    path_data = []
    
    if inmemory:
        out_func = lambda x: x
        in_func = lambda x: x
    else:
        out_func = lambda x: x.cpu().numpy()
        in_func = lambda x: torch.from_numpy(x).to(device) if not isinstance(x, torch.Tensor) else x.to(device)
    
    for i in range(num_batches):
        batch_data = data[i * batchsize : (i + 1) * batchsize]
        path_data.append(
            out_func(calc_paths(in_func(batch_data), stepsize, depth))
        )
    
    if remainder > 0:
        batch_data = data[-remainder:]
        path_data.append(
            out_func(calc_paths(in_func(batch_data), stepsize, depth))
        )
    
    if inmemory:
        path_data = torch.cat(path_data, dim=0)
    else:
        path_data = np.concatenate(path_data, axis=0)
    
    return path_data


def batch_calc_coeffs(data, include_time, T, inmemory=True, device=None):
    """Calculate coefficients in batches to manage memory."""
    if device is None:
        device = data.device if hasattr(data, 'device') else torch.device('cpu')
    
    N = len(data)
    batchsize = 128
    num_batches = N // batchsize
    remainder = N % batchsize
    coeffs = []
    
    if inmemory:
        out_func = lambda x: x
        in_func = lambda x: x
    else:
        out_func = lambda x: x.cpu().numpy()
        in_func = lambda x: torch.from_numpy(x).to(device) if not isinstance(x, torch.Tensor) else x.to(device)
    
    for i in range(num_batches):
        batch_data = data[i * batchsize : (i + 1) * batchsize]
        coeffs.append(
            out_func(calc_coeffs(in_func(batch_data), include_time, T))
        )
    
    if remainder > 0:
        batch_data = data[-remainder:]
        coeffs.append(
            out_func(calc_coeffs(in_func(batch_data), include_time, T))
        )
    
    if inmemory:
        coeffs = torch.cat(coeffs, dim=0)
    else:
        coeffs = np.concatenate(coeffs, axis=0)
    
    return coeffs


def dataset_generator(
    name,
    data,
    labels,
    stepsize,
    depth,
    include_time,
    T,
    inmemory=True,
    idxs=None,
    use_presplit=False,
    device=None,
):
    """Generate dataset with different dataloader formats."""
    if device is None:
        device = torch.device('cpu')
    
    N = len(data) if not use_presplit else len(data[0])
    
    if idxs is None:
        if use_presplit:
            train_data, val_data, test_data = data
            train_labels, val_labels, test_labels = labels
        else:
            # Random permutation for train/val/test split
            bound1 = int(N * 0.7)
            bound2 = int(N * 0.85)
            idxs_new = torch.randperm(N)
            
            train_data = data[idxs_new[:bound1]]
            train_labels = labels[idxs_new[:bound1]]
            val_data = data[idxs_new[bound1:bound2]]
            val_labels = labels[idxs_new[bound1:bound2]]
            test_data = data[idxs_new[bound2:]]
            test_labels = labels[idxs_new[bound2:]]
    else:
        train_data = data[idxs[0]]
        train_labels = labels[idxs[0]]
        val_data = data[idxs[1]]
        val_labels = labels[idxs[1]]
        test_data = None
        test_labels = None

    # Calculate paths
    train_paths = batch_calc_paths(train_data, stepsize, depth, inmemory, device)
    val_paths = batch_calc_paths(val_data, stepsize, depth, inmemory, device)
    if test_data is not None:
        test_paths = batch_calc_paths(test_data, stepsize, depth, inmemory, device)
    else:
        test_paths = None
    
    # Create intervals
    intervals = torch.arange(0, train_data.shape[1], stepsize, dtype=torch.float32)
    intervals = torch.cat([intervals, torch.tensor([train_data.shape[1]], dtype=torch.float32)])
    intervals = intervals * (T / train_data.shape[1])

    # Calculate coefficients
    train_coeffs = calc_coeffs(train_data, include_time, T)
    val_coeffs = calc_coeffs(val_data, include_time, T)
    if test_data is not None:
        test_coeffs = calc_coeffs(test_data, include_time, T)
    
    # Create coefficient data tuples
    train_times = (T / train_data.shape[1]) * torch.arange(train_data.shape[1], dtype=torch.float32).unsqueeze(0).expand(train_data.shape[0], -1)
    val_times = (T / val_data.shape[1]) * torch.arange(val_data.shape[1], dtype=torch.float32).unsqueeze(0).expand(val_data.shape[0], -1)
    
    train_coeff_data = (train_times, train_coeffs, train_data[:, 0, :])
    val_coeff_data = (val_times, val_coeffs, val_data[:, 0, :])
    
    if test_data is not None:
        test_times = (T / test_data.shape[1]) * torch.arange(test_data.shape[1], dtype=torch.float32).unsqueeze(0).expand(test_data.shape[0], -1)
        test_coeff_data = (test_times, test_coeffs, test_data[:, 0, :])
    else:
        test_coeff_data = None

    # Create path data tuples
    train_path_data = (train_times, train_paths, train_data[:, 0, :])
    val_path_data = (val_times, val_paths, val_data[:, 0, :])
    
    if test_data is not None:
        test_path_data = (test_times, test_paths, test_data[:, 0, :])
    else:
        test_path_data = None

    # Calculate dimensions
    data_dim = train_data.shape[-1]
    if len(train_labels.shape) == 1 or name == "ppg":
        label_dim = 1
    else:
        label_dim = train_labels.shape[-1]
    logsig_dim = train_paths.shape[-1]

    # Create dataloaders
    raw_dataloaders = {
        "train": Dataloader(train_data, train_labels, inmemory, device),
        "val": Dataloader(val_data, val_labels, inmemory, device),
        "test": Dataloader(test_data, test_labels, inmemory, device) if test_data is not None else None,
    }
    
    coeff_dataloaders = {
        "train": Dataloader(train_coeff_data, train_labels, inmemory, device),
        "val": Dataloader(val_coeff_data, val_labels, inmemory, device),
        "test": Dataloader(test_coeff_data, test_labels, inmemory, device) if test_coeff_data is not None else None,
    }

    path_dataloaders = {
        "train": Dataloader(train_path_data, train_labels, inmemory, device),
        "val": Dataloader(val_path_data, val_labels, inmemory, device),
        "test": Dataloader(test_path_data, test_labels, inmemory, device) if test_path_data is not None else None,
    }
    
    return Dataset(
        name,
        raw_dataloaders,
        coeff_dataloaders,
        path_dataloaders,
        data_dim,
        logsig_dim,
        intervals,
        label_dim,
    )


def create_uea_dataset(
    data_dir,
    name,
    use_idxs,
    use_presplit,
    stepsize,
    depth,
    include_time,
    T,
    device=None,
):
    """Create UEA dataset."""
    if device is None:
        device = torch.device('cpu')

    if use_presplit:
        idxs = None
        with open(data_dir + f"/processed/UEA/{name}/X_train.pkl", "rb") as f:
            train_data = jax_to_torch(pickle.load(f), device)
        with open(data_dir + f"/processed/UEA/{name}/y_train.pkl", "rb") as f:
            train_labels = jax_to_torch(pickle.load(f), device)
        with open(data_dir + f"/processed/UEA/{name}/X_val.pkl", "rb") as f:
            val_data = jax_to_torch(pickle.load(f), device)
        with open(data_dir + f"/processed/UEA/{name}/y_val.pkl", "rb") as f:
            val_labels = jax_to_torch(pickle.load(f), device)
        with open(data_dir + f"/processed/UEA/{name}/X_test.pkl", "rb") as f:
            test_data = jax_to_torch(pickle.load(f), device)
        with open(data_dir + f"/processed/UEA/{name}/y_test.pkl", "rb") as f:
            test_labels = jax_to_torch(pickle.load(f), device)
            
        if include_time:
            ts = (T / train_data.shape[1]) * torch.arange(train_data.shape[1], dtype=torch.float32).unsqueeze(0).expand(train_data.shape[0], -1).unsqueeze(-1)
            train_data = torch.cat([ts, train_data], dim=2)
            
            ts = (T / val_data.shape[1]) * torch.arange(val_data.shape[1], dtype=torch.float32).unsqueeze(0).expand(val_data.shape[0], -1).unsqueeze(-1)
            val_data = torch.cat([ts, val_data], dim=2)
            
            ts = (T / test_data.shape[1]) * torch.arange(test_data.shape[1], dtype=torch.float32).unsqueeze(0).expand(test_data.shape[0], -1).unsqueeze(-1)
            test_data = torch.cat([ts, test_data], dim=2)
            
        data = (train_data, val_data, test_data)
        onehot_labels = (train_labels, val_labels, test_labels)
    else:
        with open(data_dir + f"/processed/UEA/{name}/data.pkl", "rb") as f:
            data = jax_to_torch(pickle.load(f), device)
        with open(data_dir + f"/processed/UEA/{name}/labels.pkl", "rb") as f:
            labels = jax_to_torch(pickle.load(f), device)
            
        # Create one-hot labels
        unique_labels = torch.unique(labels)
        onehot_labels = torch.zeros((len(labels), len(unique_labels)), device=device)
        onehot_labels[torch.arange(len(labels)), labels] = 1
        
        if use_idxs:
            with open(data_dir + f"/processed/UEA/{name}/original_idxs.pkl", "rb") as f:
                idxs = pickle.load(f)
                # Convert to torch tensors if needed
                if isinstance(idxs[0], np.ndarray):
                    idxs = (torch.from_numpy(idxs[0]).to(device), torch.from_numpy(idxs[1]).to(device))
        else:
            idxs = None

        if include_time:
            ts = (T / data.shape[1]) * torch.arange(data.shape[1], dtype=torch.float32).unsqueeze(0).expand(data.shape[0], -1).unsqueeze(-1)
            data = torch.cat([ts, data], dim=2)

    return dataset_generator(
        name,
        data,
        onehot_labels,
        stepsize,
        depth,
        include_time,
        T,
        idxs=idxs,
        use_presplit=use_presplit,
        device=device,
    )


def create_toy_dataset(data_dir, name, stepsize, depth, include_time, T, device=None):
    """Create toy dataset."""
    if device is None:
        device = torch.device('cpu')
        
    with open(data_dir + "/processed/toy/signature/data.pkl", "rb") as f:
        data = jax_to_torch(pickle.load(f), device)
    with open(data_dir + "/processed/toy/signature/labels.pkl", "rb") as f:
        labels = jax_to_torch(pickle.load(f), device)
        
    # Extract specific signature components based on name
    if name == "signature1":
        labels = ((torch.sign(labels[0][:, 2]) + 1) / 2).long()
    elif name == "signature2":
        labels = ((torch.sign(labels[1][:, 2, 5]) + 1) / 2).long()
    elif name == "signature3":
        labels = ((torch.sign(labels[2][:, 2, 5, 0]) + 1) / 2).long()
    elif name == "signature4":
        labels = ((torch.sign(labels[3][:, 2, 5, 0, 3]) + 1) / 2).long()
    
    # Create one-hot labels
    unique_labels = torch.unique(labels)
    onehot_labels = torch.zeros((len(labels), len(unique_labels)), device=device)
    onehot_labels[torch.arange(len(labels)), labels] = 1
    
    idxs = None

    if include_time:
        ts = (T / data.shape[1]) * torch.arange(data.shape[1], dtype=torch.float32).unsqueeze(0).expand(data.shape[0], -1).unsqueeze(-1)
        data = torch.cat([ts, data], dim=2)

    return dataset_generator(
        "toy", data, onehot_labels, stepsize, depth, include_time, T, idxs, device=device
    )


def create_ppg_dataset(
    data_dir, use_presplit, stepsize, depth, include_time, T, device=None
):
    """Create PPG dataset."""
    if device is None:
        device = torch.device('cpu')
        
    with open(data_dir + "/processed/PPG/ppg/X_train.pkl", "rb") as f:
        train_data = jax_to_torch(pickle.load(f), device)
    with open(data_dir + "/processed/PPG/ppg/y_train.pkl", "rb") as f:
        train_labels = jax_to_torch(pickle.load(f), device)
    with open(data_dir + "/processed/PPG/ppg/X_val.pkl", "rb") as f:
        val_data = jax_to_torch(pickle.load(f), device)
    with open(data_dir + "/processed/PPG/ppg/y_val.pkl", "rb") as f:
        val_labels = jax_to_torch(pickle.load(f), device)
    with open(data_dir + "/processed/PPG/ppg/X_test.pkl", "rb") as f:
        test_data = jax_to_torch(pickle.load(f), device)
    with open(data_dir + "/processed/PPG/ppg/y_test.pkl", "rb") as f:
        test_labels = jax_to_torch(pickle.load(f), device)

    if include_time:
        ts = (T / train_data.shape[1]) * torch.arange(train_data.shape[1], dtype=torch.float32).unsqueeze(0).expand(train_data.shape[0], -1).unsqueeze(-1)
        train_data = torch.cat([ts, train_data], dim=2)
        
        ts = (T / val_data.shape[1]) * torch.arange(val_data.shape[1], dtype=torch.float32).unsqueeze(0).expand(val_data.shape[0], -1).unsqueeze(-1)
        val_data = torch.cat([ts, val_data], dim=2)
        
        ts = (T / test_data.shape[1]) * torch.arange(test_data.shape[1], dtype=torch.float32).unsqueeze(0).expand(test_data.shape[0], -1).unsqueeze(-1)
        test_data = torch.cat([ts, test_data], dim=2)

    if use_presplit:
        data = (train_data, val_data, test_data)
        labels = (train_labels, val_labels, test_labels)
    else:
        data = torch.cat((train_data, val_data, test_data), dim=0)
        labels = torch.cat((train_labels, val_labels, test_labels), dim=0)

    return dataset_generator(
        "ppg",
        data,
        labels,
        stepsize,
        depth,
        include_time,
        T,
        inmemory=False,
        use_presplit=use_presplit,
        device=device,
    )


def create_dataset(
    data_dir,
    name,
    use_idxs,
    use_presplit,
    stepsize,
    depth,
    include_time,
    T,
    device=None,
):
    """Create dataset based on name and parameters."""
    if device is None:
        device = torch.device('cpu')
        
    uea_subfolders = [
        f.name for f in os.scandir(data_dir + "/processed/UEA") if f.is_dir()
    ]
    toy_subfolders = [
        f.name for f in os.scandir(data_dir + "/processed/toy") if f.is_dir()
    ]

    if name in uea_subfolders:
        return create_uea_dataset(
            data_dir,
            name,
            use_idxs,
            use_presplit,
            stepsize,
            depth,
            include_time,
            T,
            device=device,
        )
    elif name[:-1] in toy_subfolders:
        return create_toy_dataset(
            data_dir, name, stepsize, depth, include_time, T, device=device
        )
    elif name == "ppg":
        return create_ppg_dataset(
            data_dir, use_presplit, stepsize, depth, include_time, T, device=device
        )
    else:
        raise ValueError(f"Dataset {name} not found in UEA folder and not toy dataset")
