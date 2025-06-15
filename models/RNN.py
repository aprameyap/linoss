"""
This module implements the `RNN` class and various RNN cell classes using PyTorch. The `RNN`
class is designed to handle both classification and regression tasks, and can be configured with different
types of RNN cells.

Attributes of the `RNN` class:
- `cell`: The RNN cell used within the RNN, which can be one of several types (e.g., `LinearCell`, `GRUCell`,
          `LSTMCell`, `MLPCell`).
- `output_layer`: The linear layer applied to the hidden state to produce the model's output.
- `hidden_dim`: The dimension of the hidden state $h_t$.
- `classification`: A boolean indicating whether the model is used for classification tasks.
- `output_step`: For regression tasks, specifies how many steps to skip before outputting a prediction.

RNN Cell Classes:
- `_AbstractRNNCell`: An abstract base class for all RNN cells, defining the interface for custom RNN cells.
- `LinearCell`: A simple RNN cell that applies a linear transformation to the concatenated input and hidden state.
- `GRUCell`: An implementation of the Gated Recurrent Unit (GRU) cell.
- `LSTMCell`: An implementation of the Long Short-Term Memory (LSTM) cell.
- `MLPCell`: An RNN cell that applies a multi-layer perceptron (MLP) to the concatenated input and hidden state.

Each RNN cell class implements the following methods:
- `__init__`: Initialises the RNN cell with the specified input dimensions and hidden state size.
- `forward`: Applies the RNN cell to the input and hidden state, returning the updated hidden state.

The `RNN` class also includes:
- A `forward` method that processes a sequence of inputs, returning either the final output for classification or a
sequence of outputs for regression.
"""

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional


class MLP(nn.Module):
    """Multi-layer perceptron with customizable activation functions."""
    
    def __init__(self, in_size, out_size, width_size, depth, activation=F.relu, final_activation=None):
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


class _AbstractRNNCell(nn.Module, abc.ABC):
    """Abstract RNN Cell class."""

    def __init__(self):
        super().__init__()
        self.hidden_size = None

    @abc.abstractmethod
    def forward(self, state, input_tensor):
        """
        Apply the RNN cell to input and hidden state.
        
        Args:
            state: Hidden state (or tuple of states for LSTM)
            input_tensor: Input tensor
            
        Returns:
            Updated hidden state (or tuple of states for LSTM)
        """
        raise NotImplementedError


class LinearCell(_AbstractRNNCell):    
    def __init__(self, data_dim, hidden_dim):
        super().__init__()
        self.cell = nn.Linear(data_dim + hidden_dim, hidden_dim)
        self.hidden_size = hidden_dim

    def forward(self, state, input_tensor):
        """
        Args:
            state: Hidden state tensor of shape (hidden_dim,)
            input_tensor: Input tensor of shape (data_dim,)
            
        Returns:
            Updated hidden state of shape (hidden_dim,)
        """
        concatenated = torch.cat([state, input_tensor], dim=-1)
        return self.cell(concatenated)


class GRUCell(_AbstractRNNCell):
    """Gated Recurrent Unit cell."""
    
    def __init__(self, data_dim, hidden_dim):
        super().__init__()
        self.cell = nn.GRUCell(data_dim, hidden_dim)
        self.hidden_size = hidden_dim

    def forward(self, state, input_tensor):
        """
        Args:
            state: Hidden state tensor of shape (batch_size, hidden_dim) or (hidden_dim,)
            input_tensor: Input tensor of shape (batch_size, data_dim) or (data_dim,)
            
        Returns:
            Updated hidden state
        """
        # Ensure tensors have batch dimension for GRUCell
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        new_state = self.cell(input_tensor, state)
        
        # Remove batch dimension if it was added
        if new_state.shape[0] == 1:
            new_state = new_state.squeeze(0)
            
        return new_state


class LSTMCell(_AbstractRNNCell):
    """Long Short-Term Memory cell."""
    
    def __init__(self, data_dim, hidden_dim):
        super().__init__()
        self.cell = nn.LSTMCell(data_dim, hidden_dim)
        self.hidden_size = hidden_dim

    def forward(self, state, input_tensor):
        """
        Args:
            state: Tuple of (hidden_state, cell_state)
            input_tensor: Input tensor
            
        Returns:
            Tuple of (new_hidden_state, new_cell_state)
        """
        hidden_state, cell_state = state
        
        # Ensure tensors have batch dimension for LSTMCell
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
            cell_state = cell_state.unsqueeze(0)
            
        new_hidden, new_cell = self.cell(input_tensor, (hidden_state, cell_state))
        
        # Remove batch dimension if it was added
        if new_hidden.shape[0] == 1:
            new_hidden = new_hidden.squeeze(0)
            new_cell = new_cell.squeeze(0)
            
        return (new_hidden, new_cell)


class MLPCell(_AbstractRNNCell):
    """Multi-layer perceptron RNN cell."""
    
    def __init__(self, data_dim, hidden_dim, depth, width):
        super().__init__()
        self.cell = MLP(data_dim + hidden_dim, hidden_dim, width, depth)
        self.hidden_size = hidden_dim

    def forward(self, state, input_tensor):
        """
        Args:
            state: Hidden state tensor of shape (hidden_dim,)
            input_tensor: Input tensor of shape (data_dim,)
            
        Returns:
            Updated hidden state of shape (hidden_dim,)
        """
        concatenated = torch.cat([state, input_tensor], dim=-1)
        return self.cell(concatenated)


class RNN(nn.Module):
    def __init__(
        self, 
        cell: _AbstractRNNCell, 
        hidden_dim: int, 
        label_dim: int, 
        classification: bool = True, 
        output_step: int = 1
    ):
        super().__init__()
        
        self.cell = cell
        self.output_layer = nn.Linear(hidden_dim, label_dim, bias=False)
        self.hidden_dim = cell.hidden_size
        self.classification = classification
        self.output_step = output_step
        
        # Stateful attributes to match original interface
        self.stateful = False
        self.nondeterministic = False
        self.lip2 = False

    def forward(self, x):
        """
        Forward pass of RNN.
        
        Args:
            x: Input sequence tensor of shape (seq_len, data_dim)
            
        Returns:
            Output predictions
        """
        seq_len = x.shape[0]
        
        # Initialize hidden state
        if isinstance(self.cell, LSTMCell):
            hidden = (
                torch.zeros(self.hidden_dim, dtype=x.dtype, device=x.device),
                torch.zeros(self.hidden_dim, dtype=x.dtype, device=x.device)
            )
        else:
            hidden = torch.zeros(self.hidden_dim, dtype=x.dtype, device=x.device)
        
        # Store all hidden states for regression tasks
        all_states = []
        
        # Process sequence step by step
        for t in range(seq_len):
            hidden = self.cell(hidden, x[t])
            
            # Store hidden state (extract hidden part for LSTM)
            if isinstance(self.cell, LSTMCell):
                all_states.append(hidden[0])  # hidden state (not cell state)
            else:
                all_states.append(hidden)
        
        # Convert list to tensor
        all_states = torch.stack(all_states)  # (seq_len, hidden_dim)
        
        if self.classification:
            # Use final state for classification
            final_state = all_states[-1]
            return F.softmax(self.output_layer(final_state), dim=0)
        else:
            # Subsample states for regression
            sampled_states = all_states[self.output_step - 1::self.output_step]
            outputs = torch.tanh(self.output_layer(sampled_states))
            return outputs