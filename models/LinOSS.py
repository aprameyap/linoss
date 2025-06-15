from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def simple_uniform_init(shape, std=1.0):
    weights = torch.rand(shape) * 2.0 * std - std
    return weights


class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, output_dim, bias=True)
        self.w2 = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        return self.w1(x) * torch.sigmoid(self.w2(x))


def binary_operator_single(q_i, q_j):
    """Binary operator for a single element of the parallel scan"""
    A_i, b_i = q_i
    A_j, b_j = q_j

    N = A_i.size(-1) // 4
    iA_ = A_i[..., 0 * N: 1 * N]
    iB_ = A_i[..., 1 * N: 2 * N]
    iC_ = A_i[..., 2 * N: 3 * N]
    iD_ = A_i[..., 3 * N: 4 * N]
    jA_ = A_j[..., 0 * N: 1 * N]
    jB_ = A_j[..., 1 * N: 2 * N]
    jC_ = A_j[..., 2 * N: 3 * N]
    jD_ = A_j[..., 3 * N: 4 * N]
    
    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_
    Anew = torch.cat([A_new, B_new, C_new, D_new], dim=-1)

    b_i1 = b_i[..., 0:N]
    b_i2 = b_i[..., N:]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = torch.cat([new_b1, new_b2], dim=-1)

    return Anew, new_b + b_j


def associative_scan(binary_op, elements):
    """
    PyTorch implementation of associative scan (parallel prefix sum)
    This is a sequential implementation - for true parallelism, you'd need
    a more sophisticated approach using GPU kernels
    """
    A_elements, b_elements = elements
    length = A_elements.shape[0]
    
    # Initialize output
    outputs_A = torch.zeros_like(A_elements)
    outputs_b = torch.zeros_like(b_elements)
    
    # Sequential scan (can be parallelized for better performance)
    outputs_A[0] = A_elements[0]
    outputs_b[0] = b_elements[0]
    
    for i in range(1, length):
        outputs_A[i], outputs_b[i] = binary_op(
            (outputs_A[i-1], outputs_b[i-1]),
            (A_elements[i], b_elements[i])
        )
    
    return outputs_A, outputs_b


def apply_linoss_im(A_diag, B, C_tilde, input_sequence, step):
    """
    Compute the LxH output of LinOSS-IM given an LxH input.
    Args:
        A_diag (torch.Tensor): diagonal state matrix (P,)
        B (torch.Tensor): input matrix (P, H) - complex
        C_tilde (torch.Tensor): output matrix (H, P) - complex
        input_sequence (torch.Tensor): input sequence (L, H)
        step (torch.Tensor): discretization time-step (P,)
    Returns:
        outputs (torch.Tensor): SSM outputs (L, H)
    """
    # Bu_elements = torch.einsum('lh,ph->lp', input_sequence, B)
    Bu_elements = torch.matmul(input_sequence, B.T)  # (L, P)

    schur_comp = 1. / (1. + step ** 2. * A_diag)
    M_IM_11 = 1. - step ** 2. * A_diag * schur_comp
    M_IM_12 = -1. * step * A_diag * schur_comp
    M_IM_21 = step * schur_comp
    M_IM_22 = schur_comp

    M_IM = torch.cat([M_IM_11, M_IM_12, M_IM_21, M_IM_22])

    M_IM_elements = M_IM.unsqueeze(0).expand(input_sequence.shape[0], -1)

    F1 = M_IM_11.unsqueeze(0) * Bu_elements * step.unsqueeze(0)
    F2 = M_IM_21.unsqueeze(0) * Bu_elements * step.unsqueeze(0)
    F = torch.cat([F1, F2], dim=-1)

    _, xs = associative_scan(binary_operator_single, (M_IM_elements, F))
    ys = xs[:, A_diag.shape[0]:]

    # Apply C_tilde and take real part
    outputs = torch.matmul(ys, C_tilde.T).real
    return outputs


def apply_linoss_imex(A_diag, B, C, input_sequence, step):
    """
    Compute the LxH output of LinOSS-IMEX given an LxH input.
    Args:
        A_diag (torch.Tensor): diagonal state matrix (P,)
        B (torch.Tensor): input matrix (P, H) - complex
        C (torch.Tensor): output matrix (H, P) - complex
        input_sequence (torch.Tensor): input sequence (L, H)
        step (torch.Tensor): discretization time-step (P,)
    Returns:
        outputs (torch.Tensor): SSM outputs (L, H)
    """
    Bu_elements = torch.matmul(input_sequence, B.T)  # (L, P)

    A_ = torch.ones_like(A_diag)
    B_ = -1. * step * A_diag
    C_ = step
    D_ = 1. - (step ** 2.) * A_diag

    M_IMEX = torch.cat([A_, B_, C_, D_])

    M_IMEX_elements = M_IMEX.unsqueeze(0).expand(input_sequence.shape[0], -1)

    F1 = Bu_elements * step.unsqueeze(0)
    F2 = Bu_elements * (step.unsqueeze(0) ** 2.)
    F = torch.cat([F1, F2], dim=-1)

    _, xs = associative_scan(binary_operator_single, (M_IMEX_elements, F))
    ys = xs[:, A_diag.shape[0]:]

    # Apply C and take real part
    outputs = torch.matmul(ys, C.T).real
    return outputs


class LinOSSLayer(nn.Module):
    def __init__(self, ssm_size, H, discretization):
        super().__init__()
        
        # Initialize parameters
        self.register_parameter('A_diag', nn.Parameter(torch.rand(ssm_size)))
        
        # Complex parameters stored as real tensors with last dimension 2 (real, imag)
        B_init = simple_uniform_init((ssm_size, H, 2), std=1./math.sqrt(H))
        self.register_parameter('B', nn.Parameter(B_init))
        
        C_init = simple_uniform_init((H, ssm_size, 2), std=1./math.sqrt(ssm_size))
        self.register_parameter('C', nn.Parameter(C_init))
        
        self.register_parameter('D', nn.Parameter(torch.randn(H)))
        self.register_parameter('steps', nn.Parameter(torch.rand(ssm_size)))
        
        self.discretization = discretization

    def forward(self, input_sequence):
        A_diag = F.relu(self.A_diag)
        
        # Convert to complex numbers
        B_complex = torch.complex(self.B[..., 0], self.B[..., 1])
        C_complex = torch.complex(self.C[..., 0], self.C[..., 1])
        
        steps = torch.sigmoid(self.steps)
        
        if self.discretization == 'IMEX':
            ys = apply_linoss_imex(A_diag, B_complex, C_complex, input_sequence, steps)
        elif self.discretization == 'IM':
            ys = apply_linoss_im(A_diag, B_complex, C_complex, input_sequence, steps)
        else:
            raise ValueError('Discretization type not implemented')

        # Apply D (skip connection)
        Du = input_sequence * self.D.unsqueeze(0)
        return ys + Du


class LinOSSBlock(nn.Module):
    def __init__(self, ssm_size, H, discretization, drop_rate=0.05):
        super().__init__()
        
        # Note: PyTorch BatchNorm1d expects (N, C) or (N, C, L) format
        # We'll use LayerNorm instead for sequence data
        self.norm = nn.LayerNorm(H, elementwise_affine=False)
        self.ssm = LinOSSLayer(ssm_size, H, discretization)
        self.glu = GLU(H, H)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        """
        Compute LinOSS block.
        Args:
            x: input tensor of shape (L, H)
        Returns:
            output tensor of shape (L, H)
        """
        skip = x
        x = self.norm(x)  # LayerNorm along the feature dimension
        x = self.ssm(x)
        x = self.dropout(F.gelu(x))
        x = self.glu(x)  # Apply GLU to each timestep
        x = self.dropout(x)
        x = skip + x  # Residual connection
        return x


class LinOSS(nn.Module):
    def __init__(
        self,
        num_blocks,
        N,
        ssm_size,
        H,
        output_dim,
        classification,
        output_step,
        discretization,
    ):
        super().__init__()
        
        self.linear_encoder = nn.Linear(N, H)
        self.blocks = nn.ModuleList([
            LinOSSBlock(ssm_size, H, discretization)
            for _ in range(num_blocks)
        ])
        self.linear_layer = nn.Linear(H, output_dim)
        self.classification = classification
        self.output_step = output_step

    def forward(self, x):
        """
        Compute LinOSS.
        Args:
            x: input tensor of shape (L, N) where L is sequence length, N is input dim
        Returns:
            output tensor
        """
        # Encode input
        x = self.linear_encoder(x)  # (L, H)
        
        # Apply LinOSS blocks
        for block in self.blocks:
            x = block(x)
        
        if self.classification:
            # For classification, average over sequence dimension
            x = torch.mean(x, dim=0)  # (H,)
            x = F.softmax(self.linear_layer(x), dim=0)
        else:
            # For regression/other tasks, subsample and apply tanh
            x = x[self.output_step - 1::self.output_step]  # Subsample
            x = torch.tanh(self.linear_layer(x))
        
        return x
