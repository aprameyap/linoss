"""
This module contains a function for generating the coefficients for a Hermite cubic spline with backwards differences.
"""

import torch
import torchcde


def calc_coeffs(data, include_time, T):
    """
    Calculate cubic spline coefficients for Neural CDE using torchcde.
    
    Args:
        data: Input data tensor of shape (batch, length, channels)
        include_time: Whether time is included as the first channel
        T: Total time duration
        
    Returns:
        Cubic spline coefficients tensor
    """
    batch_size, length, channels = data.shape
    
    if include_time:
        # Time is already included as the first channel, use data as-is
        x = data
    else:
        # Create uniform timestamps and concatenate with data
        t = torch.linspace(0, T, length, device=data.device)
        t = t.unsqueeze(0).unsqueeze(-1).expand(batch_size, length, 1)
        x = torch.cat([t, data], dim=2)
    
    # Compute Hermite cubic coefficients
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
    
    return coeffs
