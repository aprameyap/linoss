"""
This module implements the `LogNeuralCDE` class using PyTorch. The model is a
Neural Controlled Differential Equation (NCDE), where the output is approximated during
training using the Log-ODE method.

Attributes of the `LogNeuralCDE` model:
- `vf`: The vector field $f_{\theta}$ of the NCDE.
- `data_dim`: The number of channels in the input time series.
- `depth`: The depth of the Log-ODE method, currently implemented for depth 1 and 2.
- `hidden_dim`: The dimension of the hidden state $h_t$.
- `linear1`: The input linear layer used to initialise the hidden state $h_0$.
- `linear2`: The output linear layer used to obtain predictions from $h_t$.
- `pairs`: The pairs of basis elements for the terms in the depth-2 log-signature of the path.
- `classification`: Boolean indicating if the model is used for classification tasks.
- `output_step`: If the model is used for regression, the number of steps to skip before outputting a prediction.
- `intervals`: The intervals used in the Log-ODE method.
- `solver`: The solver applied to the ODE produced by the Log-ODE method.
- `stepsize_controller`: The stepsize controller for the solver.
- `dt0`: The initial step size for the solver.
- `max_steps`: The maximum number of steps allowed for the solver.
- `lambd`: The Lip(2) regularisation parameter, used to control the smoothness of the vector field.

The class also includes methods for initialising the model and for performing the forward pass, where the dynamics are
solved using the specified ODE solver.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
from typing import Union, Tuple, Optional
import itertools


class MLP(nn.Module):
    """Multi-layer perceptron with customizable activation functions."""
    
    def __init__(self, in_size, out_size, width_size, depth, activation=F.relu, final_activation=torch.tanh):
        super().__init__()
        
        if depth == 0:
            self.layers = nn.ModuleList([nn.Linear(in_size, out_size)])
        else:
            layers = [nn.Linear(in_size, width_size)]
            for _ in range(depth - 1):
                layers.append(nn.Linear(width_size, width_size))
            layers.append(nn.Linear(width_size, out_size))
            self.layers = nn.ModuleList(layers)
        
        self.activation = activation
        self.final_activation = final_activation
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
            else:
                if self.final_activation is not None:
                    x = self.final_activation(x)
        return x


class VectorField(nn.Module):
    def __init__(self, in_size, out_size, width, depth, activation=F.silu, scale=1):
        super().__init__()
        
        self.mlp = MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width,
            depth=depth,
            activation=activation,
            final_activation=torch.tanh
        )
        
        # Scale weights and biases
        self.scale = scale
        self._scale_parameters()
        
    def _scale_parameters(self):
        """Scale all weights and biases by the scale factor."""
        with torch.no_grad():
            for layer in self.mlp.layers:
                if hasattr(layer, 'weight'):
                    layer.weight.data /= self.scale
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data /= self.scale
    
    def forward(self, y):
        return self.mlp(y)


class HallSet:
    """Simple Hall Set implementation for generating Lie bracket pairs."""
    
    def __init__(self, data_dim, depth):
        self.data_dim = data_dim
        self.depth = depth
        self.data = self._generate_hall_set()
    
    def _generate_hall_set(self):
        """Generate Hall set up to specified depth."""
        if self.depth == 1:
            # Depth 1: just the basic elements [1, 2, ..., data_dim]
            return list(range(1, self.data_dim + 1))
        elif self.depth == 2:
            # Depth 2: basic elements + all pairs [i,j] with i < j
            basic = list(range(1, self.data_dim + 1))
            pairs = []
            for i in range(1, self.data_dim + 1):
                for j in range(i + 1, self.data_dim + 1):
                    pairs.append([i, j])
            return basic + pairs
        else:
            raise NotImplementedError(f"Depth {self.depth} not implemented")


def compute_jacobian_vector_product(func, inputs, vectors):
    """
    Compute Jacobian-vector products for a batch of vectors.
    
    Args:
        func: Function to compute Jacobian of
        inputs: Input point where to compute Jacobian
        vectors: Batch of vectors to multiply with Jacobian, shape (batch, input_dim)
    
    Returns:
        JVPs: Jacobian-vector products, shape (batch, output_dim)
    """
    def jvp_single(v):
        # Enable gradient computation
        inputs_copy = inputs.clone().requires_grad_(True)
        # Compute function output
        outputs = func(inputs_copy)
        # Compute JVP using autograd
        jvp_result = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs_copy,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True
        )[0]
        # Dot product with vector
        return torch.sum(jvp_result * v, dim=-1, keepdim=True)
    
    # For batch processing, we need to be more careful
    batch_size = vectors.shape[0]
    input_dim = vectors.shape[1]
    
    # Compute Jacobian manually using autograd
    inputs_req_grad = inputs.clone().requires_grad_(True)
    outputs = func(inputs_req_grad)
    output_dim = outputs.shape[-1]
    
    # Compute full Jacobian
    jacobian = torch.zeros(output_dim, input_dim, dtype=inputs.dtype, device=inputs.device)
    
    for i in range(output_dim):
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[i] = 1.0
        
        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs_req_grad,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        jacobian[i] = grads
    
    # Now compute JVPs: J @ v for each vector v
    jvps = torch.matmul(vectors, jacobian.T)  # (batch, output_dim)
    
    return jvps


class LogNeuralCDE(nn.Module):
    def __init__(
        self,
        vf_hidden_dim,
        vf_num_hidden,
        hidden_dim,
        data_dim,
        depth,
        label_dim,
        classification,
        output_step,
        intervals,
        solver='heun',
        stepsize_controller=None,
        dt0=0.01,
        max_steps=1000,
        scale=1,
        lambd=0.0,
        rtol=1e-5,
        atol=1e-7,
        **kwargs
    ):
        super().__init__()
        
        # Fixed: properly initialize VectorField
        self.vf = VectorField(
            hidden_dim,
            hidden_dim * data_dim,
            vf_hidden_dim,
            vf_num_hidden,
            activation=F.silu,
            scale=scale
        )
        
        self.data_dim = data_dim
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(data_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, label_dim)
        
        # Generate Hall set for Lie bracket pairs
        hs = HallSet(self.data_dim, self.depth)
        if self.depth == 1:
            self.pairs = None
        else:
            self.pairs = torch.tensor(hs.data[1:], dtype=torch.long)  # Exclude first element
            
        self.classification = classification
        self.output_step = output_step
        self.intervals = intervals
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.dt0 = dt0
        self.max_steps = max_steps
        self.lambd = lambd
        self.rtol = rtol
        self.atol = atol
        
        # Stateful attributes to match original interface
        self.stateful = False
        self.nondeterministic = False
        self.lip2 = True
    
    def vector_field_func(self, t, y):
        """Vector field function for the Log-ODE method."""
        # Find interval index
        idx = torch.searchsorted(self.intervals[1:], t) + 1
        idx = torch.clamp(idx, 1, len(self.intervals) - 1)
        
        # Get log-signature for this interval
        logsig_t = self._logsig[idx - 1]
        
        # Compute vector field output and reshape
        vf_out = self.vf(y)  # (hidden_dim * data_dim,)
        vf_out_reshaped = vf_out.view(self.data_dim, self.hidden_dim)
        
        if self.pairs is None or self.depth == 1:
            # Depth 1: simple case
            result = torch.matmul(logsig_t[1:], vf_out_reshaped)
        else:
            # Depth 2: include Lie bracket terms
            # Compute Jacobian-vector products for all directions
            vf_directions = vf_out_reshaped  # (data_dim, hidden_dim)
            
            # Compute JVPs: J_f(y) @ v for each direction v
            jvps = compute_jacobian_vector_product(
                func=self.vf,
                inputs=y,
                vectors=vf_directions  # (data_dim, hidden_dim) -> flattened
            )
            
            # Reshape JVPs to (data_dim, data_dim, hidden_dim)
            jvps = jvps.view(self.data_dim, self.data_dim, self.hidden_dim)
            
            # Compute Lie brackets for each pair
            def lie_bracket(jvps_tensor, pair):
                i, j = pair[0] - 1, pair[1] - 1  # Convert to 0-indexed
                return jvps_tensor[i, j] - jvps_tensor[j, i]
            
            # Apply Lie bracket to pairs starting from data_dim index
            lie_outputs = []
            for pair in self.pairs[self.data_dim:]:
                lie_outputs.append(lie_bracket(jvps, pair))
            
            if lie_outputs:
                lie_outputs = torch.stack(lie_outputs)  # (num_pairs, hidden_dim)
                
                # Combine linear and Lie bracket terms
                linear_term = torch.matmul(logsig_t[1:self.data_dim + 1], vf_out_reshaped)
                lie_term = torch.matmul(logsig_t[self.data_dim + 1:], lie_outputs)
                result = linear_term + lie_term
            else:
                result = torch.matmul(logsig_t[1:self.data_dim + 1], vf_out_reshaped)
        
        # Scale by interval length
        interval_length = self.intervals[idx] - self.intervals[idx - 1]
        result = result / interval_length
        
        return result
    
    def forward(self, X):
        """
        Forward pass for Log Neural CDE.
        
        Args:
            X: Tuple of (ts, logsig, x0) where:
                - ts: Time points, shape (T,)
                - logsig: Log-signature, shape (T-1, logsig_dim)
                - x0: Initial condition, shape (data_dim,) or (batch_size, data_dim)
        """
        ts, logsig, x0 = X
        
        # Store log-signature for use in vector field
        self._logsig = logsig
        
        # Initial hidden state
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)  # Add batch dimension
        y0 = self.linear1(x0).squeeze(0)  # Remove batch dim for odeint
        
        # Set up time points for integration
        if self.classification:
            t_eval = torch.tensor([ts[0], ts[-1]], dtype=ts.dtype, device=ts.device)
        else:
            step = self.output_step / len(ts)
            times = torch.arange(step, 1.0, step, dtype=ts.dtype, device=ts.device)
            t_eval = torch.cat([ts[:1], times, ts[-1:]])
            t_eval = torch.unique(torch.sort(t_eval)[0])
        
        # Solve the ODE
        solution = odeint(
            func=self.vector_field_func,
            y0=y0,
            t=t_eval,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
            options={
                'max_num_steps': self.max_steps,
                'step_size': self.dt0 if self.solver in ['euler', 'heun', 'rk4'] else None
            }
        )
        
        if self.classification:
            # Take final state for classification
            final_state = solution[-1]  # (hidden_dim,)
            output = F.softmax(self.linear2(final_state), dim=-1)
        else:
            # Apply output layer to all time points except initial
            outputs = torch.tanh(self.linear2(solution[1:]))
            output = outputs
            
        return output


# Utility function for computing log-signatures (placeholder)
def compute_logsignature(path, depth=2):
    """
    Placeholder for log-signature computation.
    In practice, you would use a library like `signatory` or implement
    the log-signature computation yourself.
    
    Args:
        path: Path tensor of shape (T, data_dim)
        depth: Depth of log-signature
        
    Returns:
        Log-signature tensor
    """
    # This is a placeholder - you'll need to implement or use a library
    # such as the `signatory` package for actual log-signature computation
    T, data_dim = path.shape
    
    if depth == 1:
        # For depth 1, log-signature is just the path increments
        increments = path[1:] - path[:-1]  # (T-1, data_dim)
        # Pad with zeros for the constant term
        logsig = torch.cat([torch.zeros(T-1, 1), increments], dim=1)
    else:
        # For depth 2, you need to compute both linear and quadratic terms
        # This is a simplified placeholder
        increments = path[1:] - path[:-1]  # (T-1, data_dim)
        
        # Linear terms
        linear_terms = increments
        
        # Quadratic terms (simplified - this is not the correct log-signature computation)
        quad_terms = torch.zeros(T-1, data_dim * (data_dim - 1) // 2)
        
        # Combine
        logsig = torch.cat([
            torch.zeros(T-1, 1),  # Constant term
            linear_terms,         # Linear terms
            quad_terms           # Quadratic terms
        ], dim=1)
    
    return logsig
