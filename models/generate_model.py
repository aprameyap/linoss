"""
This module provides a function to generate a model based on a model name and hyperparameters.
It supports various types of models, including Neural CDEs, RNNs, and the S5 model.

Function:
- `create_model`: Generates and returns a model instance based on the provided model name and hyperparameters.

Parameters for `create_model`:
- `model_name`: A string specifying the model architecture to create. Supported values include
  'log_ncde', 'ncde', 'nrde', 'lru', 'S5', 'rnn_linear', 'rnn_gru', 'rnn_lstm', and 'rnn_mlp'.
- `data_dim`: The input data dimension.
- `logsig_dim`: The dimension of the log-signature used in NRDE and Log-NCDE models.
- `logsig_depth`: The depth of the log-signature used in NRDE and Log-NCDE models.
- `intervals`: The intervals used in NRDE and Log-NCDE models.
- `label_dim`: The output label dimension.
- `hidden_dim`: The hidden state dimension for the model.
- `num_blocks`: The number of blocks (layers) in models like LRU or S5.
- `vf_depth`: The depth of the vector field network for CDE models.
- `vf_width`: The width of the vector field network for CDE models.
- `classification`: A boolean indicating whether the task is classification (True) or regression (False).
- `output_step`: The step interval for outputting predictions in sequence models.
- `ssm_dim`: The state-space model dimension for S5 models.
- `ssm_blocks`: The number of SSM blocks in S5 models.
- `solver`: The ODE solver used in CDE models, with a default of 'heun'.
- `stepsize_controller`: The step size controller used in CDE models (not used in PyTorch version).
- `dt0`: The initial time step for the solver.
- `max_steps`: The maximum number of steps for the solver.
- `scale`: A scaling factor applied to the vf initialisation in CDE models.
- `lambd`: A regularisation parameter used in Log-NCDE models.
- `rtol`: Relative tolerance for ODE solver.
- `atol`: Absolute tolerance for ODE solver.

Returns:
- The created model instance.

Raises:
- `ValueError`: If required hyperparameters for the specified model are not provided or if an
  unknown model name is passed.
"""

import torch
from models.LogNeuralCDEs import LogNeuralCDE
from models.LRU import LRU
from models.NeuralCDEs import NeuralCDE, NeuralRDE
from models.RNN import LinearCell, GRUCell, LSTMCell, MLPCell, RNN
from models.S5 import S5
from models.LinOSS import LinOSS


def create_model(
    model_name,
    data_dim,
    logsig_dim=None,
    logsig_depth=None,
    intervals=None,
    label_dim=None,
    hidden_dim=None,
    num_blocks=None,
    vf_depth=None,
    vf_width=None,
    classification=True,
    output_step=1,
    ssm_dim=None,
    ssm_blocks=None,
    solver='heun',
    stepsize_controller=None,  # Not used in PyTorch version
    dt0=0.01,
    max_steps=16**4,
    scale=1.0,
    lambd=0.0,
    linoss_discretization='IM',
    rtol=1e-5,
    atol=1e-7,
):
    """
    Create a model based on the specified model name and hyperparameters.
    
    Args:
        model_name: String specifying which model to create
        data_dim: Input data dimension
        logsig_dim: Log-signature dimension (for NRDE/Log-NCDE)
        logsig_depth: Log-signature depth (for Log-NCDE)
        intervals: Time intervals (for NRDE/Log-NCDE)
        label_dim: Output label dimension
        hidden_dim: Hidden state dimension
        num_blocks: Number of blocks/layers
        vf_depth: Vector field network depth (for CDE models)
        vf_width: Vector field network width (for CDE models)
        classification: Whether task is classification
        output_step: Step interval for sequence outputs
        ssm_dim: State space model dimension
        ssm_blocks: Number of SSM blocks
        solver: ODE solver method
        stepsize_controller: Not used in PyTorch version
        dt0: Initial time step
        max_steps: Maximum solver steps
        scale: Vector field initialization scale
        lambd: Regularization parameter
        linoss_discretization: Discretization method for LinOSS
        rtol: Relative tolerance for ODE solver
        atol: Absolute tolerance for ODE solver
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If required parameters are missing or model name is unknown
    """
    
    if model_name == "log_ncde":
        if vf_width is None or vf_depth is None:
            raise ValueError("Must specify vf_width and vf_depth for a Log-NCDE.")
        if logsig_depth is None:
            raise ValueError("Must specify logsig_depth for a Log-NCDE.")
        if intervals is None:
            raise ValueError("Must specify intervals for a Log-NCDE.")
        
        return LogNeuralCDE(
            vf_width,
            vf_depth,
            hidden_dim,
            data_dim,
            logsig_depth,
            label_dim,
            classification,
            output_step,
            intervals,
            solver=solver,
            stepsize_controller=stepsize_controller,
            dt0=dt0,
            max_steps=max_steps,
            scale=scale,
            lambd=lambd,
            rtol=rtol,
            atol=atol,
        )
        
    elif model_name == "ncde":
        if vf_width is None or vf_depth is None:
            raise ValueError("Must specify vf_width and vf_depth for a NCDE.")
        
        return NeuralCDE(
            vf_width,
            vf_depth,
            hidden_dim,
            data_dim,
            label_dim,
            classification,
            output_step,
            solver=solver,
            stepsize_controller=stepsize_controller,
            dt0=dt0,
            max_steps=max_steps,
            scale=scale,
            rtol=rtol,
            atol=atol,
        )
        
    elif model_name == "nrde":
        if vf_width is None or vf_depth is None:
            raise ValueError("Must specify vf_width and vf_depth for a NRDE.")
        if logsig_dim is None:
            raise ValueError("Must specify logsig_dim for a NRDE.")
        if intervals is None:
            raise ValueError("Must specify intervals for a NRDE.")
        
        return NeuralRDE(
            vf_width,
            vf_depth,
            hidden_dim,
            data_dim,
            logsig_dim,
            label_dim,
            classification,
            output_step,
            intervals,
            solver=solver,
            stepsize_controller=stepsize_controller,
            dt0=dt0,
            max_steps=max_steps,
            scale=scale,
            rtol=rtol,
            atol=atol,
        )
        
    elif model_name == "lru":
        if num_blocks is None:
            raise ValueError("Must specify num_blocks for LRU.")
        if ssm_dim is None:
            raise ValueError("Must specify ssm_dim for LRU.")
        
        return LRU(
            num_blocks,
            data_dim,
            ssm_dim,  # N parameter in LRU
            hidden_dim,  # H parameter in LRU
            label_dim,
            classification,
            output_step,
        )
        
    elif model_name == "S5":
        if num_blocks is None:
            raise ValueError("Must specify num_blocks for S5.")
        if ssm_dim is None:
            raise ValueError("Must specify ssm_dim for S5.")
        if ssm_blocks is None:
            raise ValueError("Must specify ssm_blocks for S5.")
        
        return S5(
            num_blocks,
            data_dim,
            ssm_dim,
            ssm_blocks,
            hidden_dim,
            label_dim,
            classification,
            output_step,
            C_init="lecun_normal",
            conj_sym=True,
            clip_eigs=True,
            discretisation="zoh",
            dt_min=0.001,
            dt_max=0.1,
            step_rescale=1.0,
        )
        
    elif model_name == "LinOSS":
        if num_blocks is None:
            raise ValueError("Must specify num_blocks for LinOSS.")
        if ssm_dim is None:
            raise ValueError("Must specify ssm_dim for LinOSS.")
        
        return LinOSS(
            num_blocks,
            data_dim,
            ssm_dim,
            hidden_dim,
            label_dim,
            classification,
            output_step,
            linoss_discretization,
        )
        
    elif model_name == "rnn_linear":
        cell = LinearCell(data_dim, hidden_dim)
        return RNN(cell, hidden_dim, label_dim, classification, output_step)
        
    elif model_name == "rnn_gru":
        cell = GRUCell(data_dim, hidden_dim)
        return RNN(cell, hidden_dim, label_dim, classification, output_step)
        
    elif model_name == "rnn_lstm":
        cell = LSTMCell(data_dim, hidden_dim)
        return RNN(cell, hidden_dim, label_dim, classification, output_step)
        
    elif model_name == "rnn_mlp":
        if vf_width is None or vf_depth is None:
            raise ValueError("Must specify vf_width and vf_depth for MLP cell.")
        cell = MLPCell(data_dim, hidden_dim, vf_depth, vf_width)
        return RNN(cell, hidden_dim, label_dim, classification, output_step)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# Convenience functions for creating specific model types
def create_log_ncde(data_dim, hidden_dim, label_dim, logsig_depth, intervals, 
                   vf_width, vf_depth, classification=True, **kwargs):
    """Create a Log Neural CDE model."""
    return create_model(
        "log_ncde", data_dim=data_dim, hidden_dim=hidden_dim, label_dim=label_dim,
        logsig_depth=logsig_depth, intervals=intervals, vf_width=vf_width, 
        vf_depth=vf_depth, classification=classification, **kwargs
    )


def create_ncde(data_dim, hidden_dim, label_dim, vf_width, vf_depth, 
               classification=True, **kwargs):
    """Create a Neural CDE model."""
    return create_model(
        "ncde", data_dim=data_dim, hidden_dim=hidden_dim, label_dim=label_dim,
        vf_width=vf_width, vf_depth=vf_depth, classification=classification, **kwargs
    )


def create_lru(num_blocks, data_dim, ssm_dim, hidden_dim, label_dim, 
              classification=True, **kwargs):
    """Create an LRU model."""
    return create_model(
        "lru", num_blocks=num_blocks, data_dim=data_dim, ssm_dim=ssm_dim,
        hidden_dim=hidden_dim, label_dim=label_dim, classification=classification, **kwargs
    )


def create_s5(num_blocks, data_dim, ssm_dim, ssm_blocks, hidden_dim, label_dim,
             classification=True, **kwargs):
    """Create an S5 model."""
    return create_model(
        "S5", num_blocks=num_blocks, data_dim=data_dim, ssm_dim=ssm_dim,
        ssm_blocks=ssm_blocks, hidden_dim=hidden_dim, label_dim=label_dim,
        classification=classification, **kwargs
    )


def create_linoss(num_blocks, data_dim, ssm_dim, hidden_dim, label_dim,
                 classification=True, discretization='IM', **kwargs):
    """Create a LinOSS model."""
    return create_model(
        "LinOSS", num_blocks=num_blocks, data_dim=data_dim, ssm_dim=ssm_dim,
        hidden_dim=hidden_dim, label_dim=label_dim, classification=classification,
        linoss_discretization=discretization, **kwargs
    )


def create_rnn(cell_type, data_dim, hidden_dim, label_dim, classification=True, 
              vf_width=None, vf_depth=None, **kwargs):
    """Create an RNN model with specified cell type."""
    model_name = f"rnn_{cell_type}"
    return create_model(
        model_name, data_dim=data_dim, hidden_dim=hidden_dim, label_dim=label_dim,
        classification=classification, vf_width=vf_width, vf_depth=vf_depth, **kwargs
    )
