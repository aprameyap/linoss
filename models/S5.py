"""
S5 implementation modified from: https://github.com/lindermanlab/S5/blob/main/s5/ssm_init.py

This module implements S5 using PyTorch.

Attributes of the S5 model:
- `linear_encoder`: The linear encoder applied to the input time series.
- `blocks`: A list of S5 blocks, each consisting of an S5 layer, normalisation, GLU, and dropout.
- `linear_layer`: The final linear layer that outputs the predictions of the model.
- `classification`: A boolean indicating whether the model is used for classification tasks.
- `output_step`: For regression tasks, specifies how many steps to skip before outputting a prediction.

The module also includes:
- `S5Layer`: Implements the core S5 layer using structured state space models with options for
  different discretisation methods and eigenvalue clipping.
- `S5Block`: Combines the S5 layer with batch normalisation, a GLU activation, and dropout.
- Utility functions for initialising and discretising the state space model components,
  such as `make_HiPPO`, `make_NPLR_HiPPO`, and `make_DPLR_HiPPO`.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy.linalg import block_diag as scipy_block_diag


class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, output_dim, bias=True)
        self.w2 = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        return self.w1(x) * torch.sigmoid(self.w2(x))


def make_HiPPO(N):
    """Create a HiPPO-LegS matrix.
    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size

    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:

    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation

    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = np.linalg.eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """Initialize the learnable timescale Delta by sampling
    uniformly between dt_min and dt_max.
    Args:
        dt_min (float32): minimum value
        dt_max (float32): maximum value
    Returns:
        init function
    """
    def init(shape):
        """Init function
        Args:
            shape tuple: desired shape
        Returns:
            sampled log_step (float32)
        """
        return torch.rand(shape) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
    
    return init


def init_log_steps(H, dt_min, dt_max):
    """Initialize an array of learnable timescale parameters
    Args:
        H: array length
        dt_min, dt_max: time scale bounds
    Returns:
        initialized array of timescales (float32): (H,)
    """
    log_steps = []
    init_fn = log_step_initializer(dt_min=dt_min, dt_max=dt_max)
    for i in range(H):
        log_step = init_fn((1,))
        log_steps.append(log_step)
    return torch.cat(log_steps)


def init_VinvB(shape, Vinv):
    """Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
    Note we will parameterize this with two different matrices for complex
    numbers.
     Args:
         shape (tuple): desired shape  (P,H)
         Vinv: (complex64)     the inverse eigenvectors used for initialization
     Returns:
         B_tilde (complex64) of shape (P,H,2)
    """
    # Lecun normal initialization
    fan_in = shape[0]
    std = 1.0 / math.sqrt(fan_in)
    B = torch.randn(shape) * std
    
    # Convert to torch and compute V^{-1}B
    Vinv_torch = torch.from_numpy(Vinv).cfloat()
    VinvB = torch.matmul(Vinv_torch, B.cfloat())
    
    # Stack real and imaginary parts
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return torch.stack([VinvB_real, VinvB_imag], dim=-1)


def trunc_standard_normal(shape):
    """Sample C with a truncated normal distribution with standard deviation 1.
    Args:
        shape (tuple): desired shape, of length 3, (H,P,_)
    Returns:
        sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
    """
    H, P, _ = shape
    # Lecun normal for each H
    fan_in = P
    std = 1.0 / math.sqrt(fan_in)
    return torch.randn(H, P, 2) * std


def init_CV(shape, V):
    """Initialize C_tilde=CV. First sample C. Then compute CV.
    Note we will parameterize this with two different matrices for complex
    numbers.
     Args:
         shape (tuple): desired shape  (H,P)
         V: (complex64)     the eigenvectors used for initialization
     Returns:
         C_tilde (complex64) of shape (H,P,2)
    """
    H, P = shape
    # Lecun normal initialization
    fan_in = P
    std = 1.0 / math.sqrt(fan_in)
    C_ = torch.randn(H, P, 2) * std
    
    C_complex = torch.complex(C_[..., 0], C_[..., 1])
    V_torch = torch.from_numpy(V).cfloat()
    CV = torch.matmul(C_complex, V_torch)
    
    CV_real = CV.real
    CV_imag = CV.imag
    return torch.stack([CV_real, CV_imag], dim=-1)


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretisation step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = torch.ones_like(Lambda)
    
    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta).unsqueeze(-1) * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretisation step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = torch.ones_like(Lambda)
    Lambda_bar = torch.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity)).unsqueeze(-1) * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
def binary_operator_single(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def associative_scan(binary_op, elements):
    """
    PyTorch implementation of associative scan.
    Sequential implementation - can be parallelized for better performance.
    """
    Lambda_elements, Bu_elements = elements
    length = Lambda_elements.shape[0]
    
    # Initialize outputs
    outputs_Lambda = torch.zeros_like(Lambda_elements)
    outputs_Bu = torch.zeros_like(Bu_elements)
    
    # Sequential scan
    outputs_Lambda[0] = Lambda_elements[0]
    outputs_Bu[0] = Bu_elements[0]
    
    for i in range(1, length):
        outputs_Lambda[i], outputs_Bu[i] = binary_op(
            (outputs_Lambda[i-1], outputs_Bu[i-1]),
            (Lambda_elements[i], Bu_elements[i])
        )
    
    return outputs_Lambda, outputs_Bu


def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym):
    """Compute the LxH output of discretized SSM given an LxH input.
    Args:
        Lambda_bar (complex64): discretized diagonal state matrix    (P,)
        B_bar      (complex64): discretized input matrix             (P, H)
        C_tilde    (complex64): output matrix                        (H, P)
        input_sequence (float32): input sequence of features         (L, H)
        conj_sym (bool):         whether conjugate symmetry is enforced
    Returns:
        ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    L = input_sequence.shape[0]
    Lambda_elements = Lambda_bar.unsqueeze(0).expand(L, -1)
    
    # Bu_elements: B_bar @ u for each time step
    Bu_elements = torch.matmul(input_sequence, B_bar.T)  # (L, P)
    
    # Associative scan
    _, xs = associative_scan(binary_operator_single, (Lambda_elements, Bu_elements))
    
    # Apply output matrix C
    outputs = torch.matmul(xs, C_tilde.T)  # (L, H)
    
    if conj_sym:
        return 2 * outputs.real
    else:
        return outputs.real


class S5Layer(nn.Module):
    def __init__(
        self,
        ssm_size,
        blocks,
        H,
        C_init,
        conj_sym,
        clip_eigs,
        discretisation,
        dt_min,
        dt_max,
        step_rescale,
    ):
        super().__init__()
        
        block_size = int(ssm_size / blocks)
        # Initialize state matrix A using approximation to HiPPO-LegS matrix
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

        if conj_sym:
            block_size = block_size // 2
            P = ssm_size // 2
        else:
            P = ssm_size

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = np.tile(Lambda, blocks)
        V = scipy_block_diag(*([V] * blocks))
        Vinv = scipy_block_diag(*([Vc] * blocks))

        self.H = H
        self.P = P
        if conj_sym:
            local_P = 2 * P
        else:
            local_P = P

        # Store Lambda as real and imaginary parts
        self.register_parameter('Lambda_re', nn.Parameter(torch.from_numpy(Lambda.real).float()))
        self.register_parameter('Lambda_im', nn.Parameter(torch.from_numpy(Lambda.imag).float()))

        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs

        # Initialize B matrix
        B_init = init_VinvB((local_P, self.H), Vinv)
        self.register_parameter('B', nn.Parameter(B_init))

        # Initialize state to output (C) matrix
        if C_init == "trunc_standard_normal":
            C_init_tensor = trunc_standard_normal((self.H, local_P, 2))
        elif C_init == "lecun_normal":
            C_init_tensor = init_CV((self.H, local_P), V)
        elif C_init == "complex_normal":
            std = 0.5**0.5
            C_init_tensor = torch.randn(self.H, 2 * self.P, 2) * std
        else:
            raise NotImplementedError("C_init method {} not implemented".format(C_init))

        self.register_parameter('C', nn.Parameter(C_init_tensor))

        # Initialize D matrix
        self.register_parameter('D', nn.Parameter(torch.randn(self.H)))

        # Initialize learnable discretisation timescale value
        log_step_init = init_log_steps(self.P, dt_min, dt_max)
        self.register_parameter('log_step', nn.Parameter(log_step_init.unsqueeze(-1)))

        self.step_rescale = step_rescale
        self.discretisation = discretisation

    def forward(self, input_sequence):
        if self.clip_eigs:
            Lambda = torch.clamp(self.Lambda_re, max=-1e-4) + 1j * self.Lambda_im
        else:
            Lambda = torch.complex(self.Lambda_re, self.Lambda_im)

        B_tilde = torch.complex(self.B[..., 0], self.B[..., 1])
        C_tilde = torch.complex(self.C[..., 0], self.C[..., 1])

        step = self.step_rescale * torch.exp(self.log_step[:, 0])

        # Discretize
        if self.discretisation == "zoh":
            Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        elif self.discretisation == "bilinear":
            Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)
        else:
            raise NotImplementedError(
                "Discretization method {} not implemented".format(self.discretisation)
            )

        ys = apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, self.conj_sym)

        # Add feedthrough matrix output Du
        Du = input_sequence * self.D.unsqueeze(0)
        return ys + Du


class S5Block(nn.Module):
    def __init__(
        self,
        ssm_size,
        blocks,
        H,
        C_init,
        conj_sym,
        clip_eigs,
        discretisation,
        dt_min,
        dt_max,
        step_rescale,
        drop_rate=0.05,
    ):
        super().__init__()
        
        # Use LayerNorm instead of BatchNorm for sequence data
        self.norm = nn.LayerNorm(H, elementwise_affine=False)
        self.ssm = S5Layer(
            ssm_size,
            blocks,
            H,
            C_init,
            conj_sym,
            clip_eigs,
            discretisation,
            dt_min,
            dt_max,
            step_rescale,
        )
        self.glu = GLU(H, H)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        """Compute S5 block."""
        skip = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(F.gelu(x))
        x = self.glu(x)
        x = self.dropout(x)
        x = skip + x
        return x


class S5(nn.Module):
    def __init__(
        self,
        num_blocks,
        N,
        ssm_size,
        ssm_blocks,
        H,
        output_dim,
        classification,
        output_step,
        C_init,
        conj_sym,
        clip_eigs,
        discretisation,
        dt_min,
        dt_max,
        step_rescale,
    ):
        super().__init__()
        
        self.linear_encoder = nn.Linear(N, H)
        self.blocks = nn.ModuleList([
            S5Block(
                ssm_size,
                ssm_blocks,
                H,
                C_init,
                conj_sym,
                clip_eigs,
                discretisation,
                dt_min,
                dt_max,
                step_rescale,
            )
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
        """Compute S5."""
        x = self.linear_encoder(x)
        for block in self.blocks:
            x = block(x)
        
        if self.classification:
            x = torch.mean(x, dim=0)
            x = F.softmax(self.linear_layer(x), dim=0)
        else:
            x = x[self.output_step - 1::self.output_step]
            x = torch.tanh(self.linear_layer(x))
        
        return x
