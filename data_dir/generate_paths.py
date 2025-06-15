"""
This module contains functions for generating log-signature of paths over intervals of length stepsize.
"""

import torch
import signatory

from data_dir.hall_set import HallSet


def hall_basis_logsig(x, depth, t2l=None):
    """
    Compute log-signature in Hall basis.
    
    Args:
        x: Path tensor of shape (..., length, channels)
        depth: Signature depth
        t2l: Tensor-to-Lyndon transformation matrix
        
    Returns:
        Log-signature tensor
    """
    # Compute log-signature using signatory
    logsig = signatory.logsignature(x, depth, mode='expand')
    
    if depth == 1:
        # For depth 1, prepend zero and return
        zeros = torch.zeros((*logsig.shape[:-1], 1), dtype=logsig.dtype, device=logsig.device)
        return torch.cat([zeros, logsig], dim=-1)
    else:
        # For higher depths, apply Hall basis transformation
        if t2l is not None:
            # Apply transformation (skip first column which corresponds to empty word)
            return torch.matmul(logsig, t2l[:, 1:].T)
        else:
            zeros = torch.zeros((*logsig.shape[:-1], 1), dtype=logsig.dtype, device=logsig.device)
            return torch.cat([zeros, logsig], dim=-1)


def calc_paths(data, stepsize, depth):
    """
    Generate log-signature objects from data.

    In the future, this function will use RoughPy, and return path objects,
    which can be queried over any interval for the log-signature. Right now,
    it is necessary to specify the stepsize and depth ahead of time.
    
    Args:
        data: Input tensor of shape (batch_size, length, channels)
        stepsize: Length of intervals for log-signature computation
        depth: Depth of log-signature
        
    Returns:
        Log-signatures tensor
    """
    # Prepend zeros to the beginning
    zeros = torch.zeros((data.shape[0], 1, data.shape[-1]), dtype=data.dtype, device=data.device)
    data = torch.cat([zeros, data], dim=1)

    # Create Hall set transformation matrix if needed
    if depth == 2:
        hs = HallSet(data.shape[-1], depth)
        t2l = hs.t2l_matrix(depth)
        # Move to same device as data
        t2l = t2l.to(data.device)
    else:
        t2l = None

    def prepend_last_point(x):
        """Prepend the last point of previous interval to current interval."""
        # x shape: (num_intervals, interval_length, channels)
        batch_size, num_intervals, interval_length, channels = x.shape
        
        # Create zeros for the first interval
        first_zeros = torch.zeros((batch_size, 1, 1, channels), dtype=x.dtype, device=x.device)
        
        # Get last points of each interval (except the last one)
        last_points = x[:, :-1, -1:, :]  # (batch_size, num_intervals-1, 1, channels)
        
        # Concatenate zeros for first interval and last points for subsequent intervals
        prepend_points = torch.cat([first_zeros, last_points], dim=1)
        
        # Concatenate prepend points with original intervals
        return torch.cat([prepend_points, x], dim=2)

    # Adjust stepsize if it's larger than data length
    if stepsize > data.shape[1]:
        stepsize = data.shape[1]

    # Handle data that doesn't divide evenly by stepsize
    if data.shape[1] % stepsize != 0:
        remainder_length = data.shape[1] % stepsize
        
        # Split into main data and final remainder
        final_data = data[:, -(remainder_length + 1):, ...]
        main_data = data[:, :-(remainder_length), ...]
        
        # Reshape main data into intervals
        batch_size, remaining_length, channels = main_data.shape
        num_intervals = remaining_length // stepsize
        main_data = main_data.view(batch_size, num_intervals, stepsize, channels)
        
        # Prepend last points
        main_data = prepend_last_point(main_data)
        
        # Handle final interval
        final_data = final_data.unsqueeze(1)  # Add interval dimension
        final_zeros = torch.zeros((batch_size, 1, 1, channels), dtype=final_data.dtype, device=final_data.device)
        final_zeros[:, 0, 0, :] = data[:, -(remainder_length + 1), :]  # Use the point before remainder
        final_data = torch.cat([final_zeros, final_data], dim=2)
        
    else:
        # Data divides evenly
        batch_size, total_length, channels = data.shape
        num_intervals = total_length // stepsize
        main_data = data.view(batch_size, num_intervals, stepsize, channels)
        main_data = prepend_last_point(main_data)
        final_data = None

    # Compute log-signatures for main intervals
    batch_size, num_intervals, interval_length, channels = main_data.shape
    
    # Reshape for batch processing: (batch_size * num_intervals, interval_length, channels)
    main_data_flat = main_data.view(-1, interval_length, channels)
    
    # Compute log-signatures
    logsigs_flat = []
    for i in range(main_data_flat.shape[0]):
        logsig = hall_basis_logsig(main_data_flat[i:i+1], depth, t2l)
        logsigs_flat.append(logsig)
    
    logsigs = torch.cat(logsigs_flat, dim=0)
    
    # Reshape back to (batch_size, num_intervals, logsig_dim)
    logsig_dim = logsigs.shape[-1]
    logsigs = logsigs.view(batch_size, num_intervals, logsig_dim)

    # Handle final interval if it exists
    if final_data is not None:
        final_logsigs = []
        for i in range(batch_size):
            final_logsig = hall_basis_logsig(final_data[i], depth, t2l)
            final_logsigs.append(final_logsig.unsqueeze(0))
        
        final_logsigs = torch.cat(final_logsigs, dim=0).unsqueeze(1)  # (batch_size, 1, logsig_dim)
        logsigs = torch.cat([logsigs, final_logsigs], dim=1)

    return logsigs


# Alternative implementation using vectorized operations (more efficient)
def calc_paths_vectorized(data, stepsize, depth):
    """
    More efficient vectorized version of calc_paths.
    """
    # Prepend zeros
    zeros = torch.zeros((data.shape[0], 1, data.shape[-1]), dtype=data.dtype, device=data.device)
    data = torch.cat([zeros, data], dim=1)

    # Create Hall set transformation matrix if needed
    if depth == 2:
        hs = HallSet(data.shape[-1], depth)
        t2l = hs.t2l_matrix(depth).to(data.device)
    else:
        t2l = None

    # Adjust stepsize
    if stepsize > data.shape[1]:
        stepsize = data.shape[1]

    # For simplicity, let's handle the case where data length is divisible by stepsize
    if data.shape[1] % stepsize != 0:
        # Pad data to make it divisible
        pad_length = stepsize - (data.shape[1] % stepsize)
        last_point = data[:, -1:, :].expand(-1, pad_length, -1)
        data = torch.cat([data, last_point], dim=1)

    # Reshape into intervals
    batch_size, total_length, channels = data.shape
    num_intervals = total_length // stepsize
    data = data[:, :num_intervals * stepsize, :]  # Ensure exact division
    data = data.view(batch_size, num_intervals, stepsize, channels)

    # Compute log-signatures for all intervals
    # Flatten to process all intervals at once
    data_flat = data.view(-1, stepsize, channels)
    
    # Process in batches to avoid memory issues
    batch_size_proc = 100  # Process 100 intervals at a time
    all_logsigs = []
    
    for i in range(0, data_flat.shape[0], batch_size_proc):
        batch_end = min(i + batch_size_proc, data_flat.shape[0])
        batch_data = data_flat[i:batch_end]
        
        # Compute log-signatures for this batch
        batch_logsigs = signatory.logsignature(batch_data, depth, mode='expand')
        
        if depth == 1:
            zeros = torch.zeros((batch_logsigs.shape[0], 1), dtype=batch_logsigs.dtype, device=batch_logsigs.device)
            batch_logsigs = torch.cat([zeros, batch_logsigs], dim=-1)
        elif t2l is not None:
            batch_logsigs = torch.matmul(batch_logsigs, t2l[:, 1:].T)
            zeros = torch.zeros((batch_logsigs.shape[0], 1), dtype=batch_logsigs.dtype, device=batch_logsigs.device)
            batch_logsigs = torch.cat([zeros, batch_logsigs], dim=-1)
        
        all_logsigs.append(batch_logsigs)
    
    # Concatenate all results
    logsigs = torch.cat(all_logsigs, dim=0)
    
    # Reshape back to (batch_size, num_intervals, logsig_dim)
    logsig_dim = logsigs.shape[-1]
    logsigs = logsigs.view(batch_size, num_intervals, logsig_dim)

    return logsigs
