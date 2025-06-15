"""
Code modified from https://gist.github.com/Ryu1845/7e78da4baa8925b4de482969befa949d

This module implements the `LRU` class, a model architecture using PyTorch.

Attributes of the `LRU` class:
- `linear_encoder`: The linear encoder applied to the input time series data.
- `blocks`: A list of `LRUBlock` instances, each containing the LRU layer, normalization, GLU, and dropout.
- `linear_layer`: The final linear layer that outputs the model predictions.
- `classification`: A boolean indicating whether the model is used for classification tasks.
- `output_step`: For regression tasks, specifies how many steps to skip before outputting a prediction.

The module also includes the following classes and functions:
- `GLU`: Implements a Gated Linear Unit for non-linear transformations within the model.
- `LRULayer`: A single LRU layer that applies complex-valued transformations and projections to the input.
- `LRUBlock`: A block consisting of normalization, LRU layer, GLU, and dropout, used as a building block for the `LRU`
              model.
- `binary_operator_diag`: A helper function used in the associative scan operation within `LRULayer` to process diagonal
                          elements.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def binary_operator_diag(element_i, element_j):
    """Binary operator for diagonal associative scan in LRU layer."""
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j


def associative_scan_diag(binary_op, elements):
    """
    PyTorch implementation of associative scan for diagonal elements.
    Sequential implementation - can be parallelized for better performance.
    """
    Lambda_elements, Bu_elements = elements
    length = Lambda_elements.shape[0]
    
    # Initialize outputs
    outputs_a = torch.zeros_like(Lambda_elements)
    outputs_bu = torch.zeros_like(Bu_elements)
    
    # Sequential scan
    outputs_a[0] = Lambda_elements[0]
    outputs_bu[0] = Bu_elements[0]
    
    for i in range(1, length):
        outputs_a[i], outputs_bu[i] = binary_op(
            (outputs_a[i-1], outputs_bu[i-1]),
            (Lambda_elements[i], Bu_elements[i])
        )
    
    return outputs_a, outputs_bu


class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, output_dim, bias=True)
        self.w2 = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        return self.w1(x) * torch.sigmoid(self.w2(x))


class LRULayer(nn.Module):
    def __init__(self, N, H, r_min=0, r_max=1, max_phase=6.28):
        super().__init__()
        
        self.N = N  # state dimension
        self.H = H  # model dimension
        
        # Initialize Lambda parameters (complex diagonal matrix)
        # Lambda is distributed uniformly on ring between r_min and r_max
        u1 = torch.rand(N)
        u2 = torch.rand(N)
        
        # Log-parameterized radial component
        nu_log_init = torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        self.register_parameter('nu_log', nn.Parameter(nu_log_init))
        
        # Log-parameterized phase component  
        theta_log_init = torch.log(max_phase * u2)
        self.register_parameter('theta_log', nn.Parameter(theta_log_init))
        
        # Glorot initialized Input/Output projection matrices (complex)
        # B matrix (input projection)
        B_re_init = torch.randn(N, H) / math.sqrt(2 * H)
        B_im_init = torch.randn(N, H) / math.sqrt(2 * H)
        self.register_parameter('B_re', nn.Parameter(B_re_init))
        self.register_parameter('B_im', nn.Parameter(B_im_init))
        
        # C matrix (output projection)
        C_re_init = torch.randn(H, N) / math.sqrt(N)
        C_im_init = torch.randn(H, N) / math.sqrt(N)
        self.register_parameter('C_re', nn.Parameter(C_re_init))
        self.register_parameter('C_im', nn.Parameter(C_im_init))
        
        # D matrix (skip connection)
        D_init = torch.randn(H)
        self.register_parameter('D', nn.Parameter(D_init))
        
        # Compute and store normalization factor
        diag_lambda = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))
        gamma_log_init = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.register_parameter('gamma_log', nn.Parameter(gamma_log_init))

    def forward(self, x):
        """
        Forward pass of LRU layer.
        
        Args:
            x: Input tensor of shape (L, H) where L is sequence length
            
        Returns:
            Output tensor of shape (L, H)
        """
        # Materialize the diagonal of Lambda and projections
        Lambda = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))
        
        # Normalized input projection matrix
        B_complex = torch.complex(self.B_re, self.B_im)
        gamma = torch.exp(self.gamma_log)
        B_norm = B_complex * gamma.unsqueeze(-1)
        
        # Output projection matrix
        C = torch.complex(self.C_re, self.C_im)
        
        # Running the LRU computation
        # Lambda_elements: repeat Lambda for each time step
        Lambda_elements = Lambda.unsqueeze(0).expand(x.shape[0], -1)  # (L, N)
        
        # Bu_elements: apply input projection to each time step
        Bu_elements = torch.matmul(x, B_norm.T)  # (L, N)
        
        # Associative scan to compute all hidden states
        elements = (Lambda_elements, Bu_elements)
        _, inner_states = associative_scan_diag(binary_operator_diag, elements)
        
        # Output projection: y = Re(C @ z) + D * u
        # Apply C to each hidden state and take real part
        Cz = torch.matmul(inner_states, C.T)  # (L, H)
        y = Cz.real + self.D.unsqueeze(0) * x  # (L, H)
        
        return y


class LRUBlock(nn.Module):
    def __init__(self, N, H, r_min=0, r_max=1, max_phase=6.28, drop_rate=0.1):
        super().__init__()
        
        # Use LayerNorm instead of BatchNorm for sequence data
        self.norm = nn.LayerNorm(H, elementwise_affine=False)
        self.lru = LRULayer(N, H, r_min, r_max, max_phase)
        self.glu = GLU(H, H)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        """
        Forward pass of LRU block.
        
        Args:
            x: Input tensor of shape (L, H)
            
        Returns:
            Output tensor of shape (L, H)
        """
        skip = x
        
        # Layer normalization
        x = self.norm(x)
        
        # LRU layer
        x = self.lru(x)
        
        # GELU activation + dropout
        x = self.dropout(F.gelu(x))
        
        # GLU (applied to each timestep)
        x = self.glu(x)
        
        # Dropout + residual connection
        x = self.dropout(x)
        x = skip + x
        
        return x


class LRU(nn.Module):
    def __init__(
        self,
        num_blocks,
        data_dim,
        N,
        H,
        output_dim,
        classification,
        output_step,
        r_min=0,
        r_max=1,
        max_phase=6.28,
        drop_rate=0.1,
    ):
        super().__init__()
        
        self.linear_encoder = nn.Linear(data_dim, H)
        self.blocks = nn.ModuleList([
            LRUBlock(N, H, r_min, r_max, max_phase, drop_rate)
            for _ in range(num_blocks)
        ])
        self.linear_layer = nn.Linear(H, output_dim)
        self.classification = classification
        self.output_step = output_step
        
        # Stateful attributes to match original interface
        self.stateful = True
        self.nondeterministic = True
        self.lip2 = False

    def forward(self, x):
        """
        Forward pass of LRU model.
        
        Args:
            x: Input tensor of shape (L, data_dim) where L is sequence length
            
        Returns:
            Output predictions
        """
        # Encode input
        x = self.linear_encoder(x)  # (L, H)
        
        # Apply LRU blocks
        for block in self.blocks:
            x = block(x)
        
        if self.classification:
            # For classification: average over sequence dimension
            x = torch.mean(x, dim=0)  # (H,)
            x = F.softmax(self.linear_layer(x), dim=0)
        else:
            # For regression: subsample and apply tanh
            x = x[self.output_step - 1::self.output_step]  # Subsample
            x = torch.tanh(self.linear_layer(x))
        
        return x


# Utility function for complex parameter initialization
def complex_glorot_uniform(shape, dtype=torch.complex64):
    """
    Initialize complex parameters with Glorot uniform distribution.
    
    Args:
        shape: Shape of parameter tensor
        dtype: Complex dtype
        
    Returns:
        Complex tensor initialized with Glorot uniform
    """
    if len(shape) < 2:
        fan_in = fan_out = shape[0]
    else:
        fan_in = shape[-2]
        fan_out = shape[-1]
    
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    
    real_part = torch.empty(shape).uniform_(-bound, bound)
    imag_part = torch.empty(shape).uniform_(-bound, bound)
    
    return torch.complex(real_part, imag_part)


def complex_glorot_normal(shape, dtype=torch.complex64):
    """
    Initialize complex parameters with Glorot normal distribution.
    
    Args:
        shape: Shape of parameter tensor
        dtype: Complex dtype
        
    Returns:
        Complex tensor initialized with Glorot normal
    """
    if len(shape) < 2:
        fan_in = fan_out = shape[0]
    else:
        fan_in = shape[-2]
        fan_out = shape[-1]
    
    std = math.sqrt(2.0 / (fan_in + fan_out))
    
    real_part = torch.randn(shape) * std
    imag_part = torch.randn(shape) * std
    
    return torch.complex(real_part, imag_part)
