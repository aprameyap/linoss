"""
This module implements a `Dataloader` class for loading and batching data. It supports three different types of
data inputs, tailored for different types of models used in this repository.

1. **Time Series Data**: Used by models like recurrent neural networks and structured state space models.
   - Input data should be a `torch.Tensor` of shape `(n_samples, n_timesteps, n_features)`.

2. **Neural Controlled Differential Equations (NCDEs)**: Requires sampling times, coefficients of an interpolation,
   and the initial value of the data.
   - Input data should be a tuple of length 3:
     - The first element: `torch.Tensor` of shape `(n_samples, n_timesteps)` for sampling times.
     - The second element: a tuple of length `n_coeffs`, where each element is a `torch.Tensor` of shape
       `(n_samples, n_timesteps-1, n_features)` for interpolation coefficients.
     - The third element: `torch.Tensor` of shape `(n_samples, n_features)` for the initial value.

3. **Neural Rough Differential Equations (NRDEs) and Log-NCDEs**: Requires sampling times, log-signature of the data,
   and the initial value of the data.
   - Input data should be a tuple of length 3:
     - The first element: `torch.Tensor` of shape `(n_samples, n_timesteps)` for sampling times.
     - The second element: `torch.Tensor` of shape `(n_samples, n_intervals, n_logsig_features)` for log-signature data.
     - The third element: `torch.Tensor` of shape `(n_samples, n_features)` for the initial value.

Additionally, data can be stored as a NumPy array to save GPU memory, with each batch converted to a PyTorch tensor.

Methods:
- `loop(batch_size)`: Generates data batches indefinitely. Randomly shuffles data for each batch.
- `loop_epoch(batch_size)`: Generates data batches for one epoch (i.e., a full pass through the dataset).
"""

import torch
import numpy as np


class Dataloader:
    def __init__(self, data, labels, inmemory=True, device=None):
        """
        Initialize the dataloader.
        
        Args:
            data: Input data (tensor or tuple of tensors)
            labels: Target labels (tensor)
            inmemory: Whether to keep data in memory or convert on-the-fly
            device: Device to put tensors on ('cpu', 'cuda', etc.)
        """
        self.data = data
        self.labels = labels
        self.device = device or torch.device('cpu')
        
        # Determine data type
        self.data_is_coeffs = False
        self.data_is_logsig = False
        
        if isinstance(self.data, tuple):
            if len(data[1]) > 0 and len(data[1][0].shape) > 2:
                self.data_is_coeffs = True
            else:
                self.data_is_logsig = True

        # Determine dataset size
        if self.data_is_coeffs:
            self.size = len(data[1][0])
        elif self.data_is_logsig:
            self.size = len(data[1])
        elif self.data is None:
            self.size = 0
        else:
            self.size = len(data)
            
        # Set up conversion function
        if inmemory:
            self.func = lambda x: x
        else:
            self.func = lambda x: torch.tensor(x, device=self.device) if not isinstance(x, torch.Tensor) else x.to(self.device)

    def __iter__(self):
        raise RuntimeError("Use .loop(batch_size) instead of __iter__")

    def loop(self, batch_size, shuffle=True):
        """
        Generate batches indefinitely with shuffling.
        
        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle data between epochs
            
        Yields:
            Batches of (data, labels)
        """
        if self.size == 0:
            raise ValueError("This dataloader is empty")

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")

        if batch_size > self.size:
            raise ValueError("Batch size larger than dataset size")
            
        if batch_size == self.size:
            while True:
                yield self.func(self.data), self.func(self.labels)
        else:
            indices = torch.arange(self.size)
            while True:
                if shuffle:
                    perm = torch.randperm(self.size)
                else:
                    perm = indices
                    
                start = 0
                end = batch_size
                while end <= self.size:
                    batch_perm = perm[start:end]
                    
                    if self.data_is_coeffs:
                        yield (
                            self.func(self.data[0][batch_perm]),
                            tuple(self.func(data[batch_perm]) for data in self.data[1]),
                            self.func(self.data[2][batch_perm]),
                        ), self.func(self.labels[batch_perm])
                    elif self.data_is_logsig:
                        yield (
                            self.func(self.data[0][batch_perm]),
                            self.func(self.data[1][batch_perm]),
                            self.func(self.data[2][batch_perm]),
                        ), self.func(self.labels[batch_perm])
                    else:
                        yield self.func(self.data[batch_perm]), self.func(self.labels[batch_perm])
                        
                    start = end
                    end = start + batch_size

    def loop_epoch(self, batch_size):
        """
        Generate batches for one epoch (no shuffling, single pass through data).
        
        Args:
            batch_size: Size of each batch
            
        Yields:
            Batches of (data, labels)
        """
        if self.size == 0:
            raise ValueError("This dataloader is empty")

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")

        if batch_size > self.size:
            raise ValueError("Batch size larger than dataset size")
            
        if batch_size == self.size:
            yield self.func(self.data), self.func(self.labels)
        else:
            indices = torch.arange(self.size)
            start = 0
            end = batch_size
            
            while end <= self.size:
                batch_indices = indices[start:end]
                
                if self.data_is_coeffs:
                    yield (
                        self.func(self.data[0][batch_indices]),
                        tuple(self.func(data[batch_indices]) for data in self.data[1]),
                        self.func(self.data[2][batch_indices]),
                    ), self.func(self.labels[batch_indices])
                elif self.data_is_logsig:
                    yield (
                        self.func(self.data[0][batch_indices]),
                        self.func(self.data[1][batch_indices]),
                        self.func(self.data[2][batch_indices]),
                    ), self.func(self.labels[batch_indices])
                else:
                    yield self.func(self.data[batch_indices]), self.func(self.labels[batch_indices])
                    
                start = end
                end = start + batch_size
            
            # Handle remaining samples
            if start < self.size:
                batch_indices = indices[start:]
                
                if self.data_is_coeffs:
                    yield (
                        self.func(self.data[0][batch_indices]),
                        tuple(self.func(data[batch_indices]) for data in self.data[1]),
                        self.func(self.data[2][batch_indices]),
                    ), self.func(self.labels[batch_indices])
                elif self.data_is_logsig:
                    yield (
                        self.func(self.data[0][batch_indices]),
                        self.func(self.data[1][batch_indices]),
                        self.func(self.data[2][batch_indices]),
                    ), self.func(self.labels[batch_indices])
                else:
                    yield self.func(self.data[batch_indices]), self.func(self.labels[batch_indices])


# Utility functions for converting JAX arrays to PyTorch tensors
def jax_to_torch(jax_array, device=None):
    """
    Convert a JAX array to a PyTorch tensor.
    
    Args:
        jax_array: JAX/NumPy array to convert
        device: Target device for the tensor
        
    Returns:
        PyTorch tensor
    """
    if device is None:
        device = torch.device('cpu')
    
    # Convert JAX array to numpy first, then to PyTorch
    numpy_array = np.array(jax_array)
    return torch.from_numpy(numpy_array).to(device)


def convert_jax_data_to_torch(data, labels, device=None):
    """
    Convert JAX-based data and labels to PyTorch tensors.
    
    Args:
        data: JAX data (array or tuple of arrays)
        labels: JAX labels array
        device: Target device
        
    Returns:
        Converted data and labels as PyTorch tensors
    """
    if device is None:
        device = torch.device('cpu')
    
    # Convert labels
    torch_labels = jax_to_torch(labels, device)
    
    # Convert data based on its structure
    if isinstance(data, tuple):
        # Handle NCDE coefficients or NRDE/Log-NCDE data
        if len(data) == 3:
            # (times, coeffs/logsig, initial_values)
            times = jax_to_torch(data[0], device)
            initial_values = jax_to_torch(data[2], device)
            
            if isinstance(data[1], tuple):
                # NCDE coefficients case
                coeffs = tuple(jax_to_torch(coeff, device) for coeff in data[1])
                torch_data = (times, coeffs, initial_values)
            else:
                # NRDE/Log-NCDE case
                logsig = jax_to_torch(data[1], device)
                torch_data = (times, logsig, initial_values)
        else:
            # Generic tuple case
            torch_data = tuple(jax_to_torch(arr, device) for arr in data)
    else:
        # Regular time series data
        torch_data = jax_to_torch(data, device)
    
    return torch_data, torch_labels


def create_dataloader_from_jax(jax_data, jax_labels, device=None, inmemory=True):
    """
    Create a PyTorch dataloader from JAX arrays.
    
    Args:
        jax_data: JAX data arrays
        jax_labels: JAX label arrays
        device: Target device ('cpu', 'cuda', etc.)
        inmemory: Whether to keep data in memory
        
    Returns:
        Dataloader instance with converted data
    """
    torch_data, torch_labels = convert_jax_data_to_torch(jax_data, jax_labels, device)
    return Dataloader(torch_data, torch_labels, inmemory=inmemory, device=device)
