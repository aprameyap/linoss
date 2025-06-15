"""
This module implements the `NeuralCDE` and `NeuralRDE` classes using PyTorch.

Attributes of `NeuralCDE`:
- `vf`: The vector field $f_{\theta}$ of the NCDE.
- `data_dim`: Number of channels in the input time series.
- `hidden_dim`: Dimension of the hidden state $h_t$.
- `linear1`: Input linear layer for initializing $h_0$.
- `linear2`: Output linear layer for generating predictions from $h_t$.
- `classification`: Boolean indicating if the model is used for classification.
- `output_step`: For regression tasks, specifies the step interval for outputting predictions.
- `solver`: The solver used to integrate the NCDE.
- `dt0`: Initial step size for the solver.
- `max_steps`: Maximum number of steps allowed for the solver.

Attributes of `NeuralRDE`:
- `vf`: The vector field $\bar{f}_{\theta}$ of the NRDE (excluding the final linear layer).
- `data_dim`: Number of channels in the input time series.
- `logsig_dim`: Dimension of the log-signature used as input to the NRDE.
- `hidden_dim`: Dimension of the hidden state $h_t$.
- `mlp_linear`: Final linear layer of the vector field.
- `linear1`: Input linear layer for initializing $h_0$.
- `linear2`: Output linear layer for generating predictions from $h_t$.
- `classification`: Boolean indicating if the model is used for classification.
- `output_step`: For regression tasks, specifies the step interval for outputting predictions.
- `solver`: The solver used to integrate the NRDE.
- `dt0`: Initial step size for the solver.
- `max_steps`: Maximum number of steps allowed for the solver.

The module also includes the `VectorField` class, which defines the vector fields used by both
`NeuralCDE` and `NeuralRDE`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
from typing import Union, Tuple, Optional
from scipy import interpolate


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


class CubicSplineInterpolation:
    """Cubic spline interpolation for control paths."""
    
    def __init__(self, ts, coeffs):
        """
        Args:
            ts: Time points, shape (T,)
            coeffs: Coefficients for interpolation, shape (T, D, 4) where 4 represents cubic coefficients
        """
        self.ts = ts
        self.coeffs = coeffs
        
    def __call__(self, t):
        """Evaluate the interpolated path at time t."""
        # Find the interval containing t
        idx = torch.searchsorted(self.ts[1:], t)
        idx = torch.clamp(idx, 0, len(self.ts) - 2)
        
        # Get the local time within the interval
        dt = t - self.ts[idx]
        
        # Evaluate cubic polynomial: c0 + c1*dt + c2*dt^2 + c3*dt^3
        result = (self.coeffs[idx, :, 0] + 
                 self.coeffs[idx, :, 1] * dt +
                 self.coeffs[idx, :, 2] * dt**2 +
                 self.coeffs[idx, :, 3] * dt**3)
        
        return result


class VectorField(nn.Module):
    def __init__(self, in_size, out_size, width, depth, activation=F.relu, scale=1):
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


class NeuralCDE(nn.Module):
    def __init__(
        self,
        vf_hidden_dim,
        vf_num_hidden,
        hidden_dim,
        data_dim,
        label_dim,
        classification,
        output_step,
        solver='heun',  # Changed default to 'heun' to match diffrax.Heun()
        stepsize_controller=None,
        dt0=0.01,
        max_steps=1000,
        scale=1,
        rtol=1e-5,
        atol=1e-7,
        **kwargs
    ):
        super().__init__()
        
        self.vf = VectorField(
            hidden_dim,
            hidden_dim * data_dim,
            vf_hidden_dim,
            vf_num_hidden,
            scale=scale
        )
        self.linear1 = nn.Linear(data_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, label_dim)
        self.classification = classification
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.solver = solver
        self.stepsize_controller = stepsize_controller  # Not directly used in torchdiffeq
        self.dt0 = dt0
        self.max_steps = max_steps
        self.output_step = output_step
        self.rtol = rtol
        self.atol = atol
        
        # Stateful attributes to match original interface
        self.stateful = False
        self.nondeterministic = False  
        self.lip2 = False
        
    def vector_field(self, t, y):
        """Vector field for the Neural CDE."""
        # Get the control derivative at time t
        if hasattr(self, '_control_interp'):
            control_deriv = self._control_interp(t)
        else:
            # Fallback: assume control derivative is stored
            control_deriv = self._control_deriv
            
        # Reshape vector field output
        vf_output = self.vf(y)  # (batch_size, hidden_dim * data_dim)
        vf_reshaped = vf_output.view(-1, self.hidden_dim, self.data_dim)  # (batch_size, hidden_dim, data_dim)
        
        # Apply control: sum over data dimensions
        if control_deriv.dim() == 1:
            control_deriv = control_deriv.unsqueeze(0)  # Add batch dimension
            
        dydt = torch.sum(vf_reshaped * control_deriv.unsqueeze(1), dim=2)  # (batch_size, hidden_dim)
        
        return dydt
    
    def forward(self, X):
        """
        Forward pass for Neural CDE.
        
        Args:
            X: Tuple of (ts, coeffs, x0) where:
                - ts: Time points, shape (T,)
                - coeffs: Cubic spline coefficients, shape (T-1, data_dim, 4) 
                - x0: Initial condition, shape (data_dim,) or (batch_size, data_dim)
        """
        ts, coeffs, x0 = X
        
        # Create interpolation object
        control_interp = CubicSplineInterpolation(ts, coeffs)
        self._control_interp = control_interp
        
        # Initial hidden state
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)  # Add batch dimension
        y0 = self.linear1(x0)
        
        # Set up time points for integration
        if self.classification:
            t_eval = torch.tensor([ts[0], ts[-1]], dtype=ts.dtype, device=ts.device)
        else:
            step = self.output_step / len(ts)
            times = torch.arange(step, 1.0, step, dtype=ts.dtype, device=ts.device)
            t_eval = torch.cat([ts[:1], times, ts[-1:]])
            t_eval = torch.unique(torch.sort(t_eval)[0])
        
        # Solve the ODE with Heun method
        solution = odeint(
            func=self.vector_field,
            y0=y0,
            t=t_eval,
            method=self.solver,  # 'heun' for Heun's method
            rtol=self.rtol,
            atol=self.atol,
            options={
                'max_num_steps': self.max_steps,
                'step_size': self.dt0 if self.solver in ['euler', 'heun', 'rk4'] else None
            }
        )
        
        if self.classification:
            # Take final state for classification
            final_state = solution[-1]  # (batch_size, hidden_dim)
            output = F.softmax(self.linear2(final_state), dim=-1)
        else:
            # Apply output layer to all time points
            outputs = torch.tanh(self.linear2(solution[1:]))  # Skip initial time
            output = outputs
            
        return output


class NeuralRDE(nn.Module):
    def __init__(
        self,
        vf_hidden_dim,
        vf_num_hidden,
        hidden_dim,
        data_dim,
        logsig_dim,
        label_dim,
        classification,
        output_step,
        intervals,
        solver='heun',  # Changed default to 'heun' to match diffrax.Heun()
        stepsize_controller=None,
        dt0=0.01,
        max_steps=1000,
        scale=1,
        rtol=1e-5,
        atol=1e-7,
        **kwargs
    ):
        super().__init__()
        
        # Exclude first element as always zero
        self.logsig_dim = logsig_dim - 1
        
        self.vf = VectorField(
            hidden_dim,
            vf_hidden_dim,
            vf_hidden_dim,
            vf_num_hidden - 1,
            scale=scale
        )
        self.mlp_linear = nn.Linear(vf_hidden_dim, hidden_dim * self.logsig_dim)
        self.linear1 = nn.Linear(data_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, label_dim)
        
        self.classification = classification
        self.output_step = output_step
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.intervals = intervals
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.dt0 = dt0
        self.max_steps = max_steps
        self.rtol = rtol
        self.atol = atol
        
        # Stateful attributes to match original interface
        self.stateful = False
        self.nondeterministic = False
        self.lip2 = False
        
    def vector_field(self, t, y):
        """Vector field for the Neural RDE."""
        # Find interval index
        idx = torch.searchsorted(self.intervals[1:], t) + 1
        idx = torch.clamp(idx, 1, len(self.intervals) - 1)
        
        # Get log-signature for this interval (excluding first element which is always 0)
        logsig_segment = self._logsig[idx - 1][1:]  # Shape: (logsig_dim - 1,)
        
        # Compute vector field
        vf_output = self.vf(y)  # (batch_size, vf_hidden_dim)
        mlp_output = self.mlp_linear(vf_output)  # (batch_size, hidden_dim * logsig_dim)
        
        # Reshape for matrix multiplication
        batch_size = y.shape[0] if y.dim() > 1 else 1
        mlp_reshaped = mlp_output.view(batch_size, self.hidden_dim, self.logsig_dim)
        
        # Apply log-signature
        if logsig_segment.dim() == 1:
            logsig_segment = logsig_segment.unsqueeze(0)  # Add batch dimension if needed
            
        dydt = torch.sum(mlp_reshaped * logsig_segment.unsqueeze(1), dim=2)
        
        # Scale by interval length
        interval_length = self.intervals[idx] - self.intervals[idx - 1]
        dydt = dydt / interval_length
        
        return dydt
    
    def forward(self, X):
        """
        Forward pass for Neural RDE.
        
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
        y0 = self.linear1(x0)
        
        # Set up time points for integration
        if self.classification:
            t_eval = torch.tensor([ts[0], ts[-1]], dtype=ts.dtype, device=ts.device)
        else:
            step = self.output_step / len(ts)
            times = torch.arange(step, 1.0, step, dtype=ts.dtype, device=ts.device)
            t_eval = torch.cat([ts[:1], times, ts[-1:]])
            t_eval = torch.unique(torch.sort(t_eval)[0])
        
        # Solve the ODE with Heun method
        solution = odeint(
            func=self.vector_field,
            y0=y0,
            t=t_eval,
            method=self.solver,  # 'heun' for Heun's method
            rtol=self.rtol,
            atol=self.atol,
            options={
                'max_num_steps': self.max_steps,
                'step_size': self.dt0 if self.solver in ['euler', 'heun', 'rk4'] else None
            }
        )
        
        if self.classification:
            # Take final state for classification
            final_state = solution[-1]  # (batch_size, hidden_dim)
            output = F.softmax(self.linear2(final_state), dim=-1)
        else:
            # Apply output layer to all time points
            outputs = torch.tanh(self.linear2(solution[1:]))  # Skip initial time
            output = outputs
            
        return output


# Utility functions for data preprocessing
def compute_cubic_spline_coeffs(ts, data):
    """
    Compute cubic spline coefficients for control path.
    
    Args:
        ts: Time points, shape (T,)
        data: Data values, shape (T, data_dim)
        
    Returns:
        coeffs: Coefficients, shape (T-1, data_dim, 4)
    """
    ts_np = ts.detach().cpu().numpy()
    data_np = data.detach().cpu().numpy()
    
    coeffs_list = []
    for dim in range(data.shape[1]):
        # Fit cubic spline for this dimension
        cs = interpolate.CubicSpline(ts_np, data_np[:, dim])
        
        # Extract coefficients for each interval
        coeffs_dim = []
        for i in range(len(ts_np) - 1):
            # Get polynomial coefficients in [ts[i], ts[i+1]]
            t_local = ts_np[i]
            # CubicSpline coefficients are in descending order of powers
            poly_coeffs = cs.c[:, i]  # [c3, c2, c1, c0]
            # Reverse to get [c0, c1, c2, c3] (ascending powers)
            poly_coeffs = poly_coeffs[::-1]
            coeffs_dim.append(poly_coeffs)
        
        coeffs_list.append(coeffs_dim)
    
    # Convert to tensor: (T-1, data_dim, 4)
    coeffs = torch.tensor(coeffs_list, dtype=data.dtype, device=data.device)
    coeffs = coeffs.transpose(0, 1)  # (T-1, data_dim, 4)
    
    return coeffs
