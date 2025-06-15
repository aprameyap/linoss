"""
This module defines functions for creating datasets, building models, and training them using PyTorch.
The main function, `create_dataset_model_and_train`, is designed to initialise the
dataset, construct the model, and execute the training process.

The function `create_dataset_model_and_train` takes the following arguments:

- `seed`: A random seed for reproducibility.
- `data_dir`: The directory where the dataset is stored.
- `use_presplit`: A boolean indicating whether to use a pre-split dataset.
- `dataset_name`: The name of the dataset to load and use for training.
- `output_step`: For regression tasks, the number of steps to skip before outputting a prediction.
- `metric`: The metric to use for evaluation. Supported values are `'mse'` for regression and `'accuracy'` for
            classification.
- `include_time`: A boolean indicating whether to include time as a channel in the time series data.
- `T`: The maximum time value to scale time data to [0, T].
- `model_name`: The name of the model architecture to use.
- `stepsize`: The size of the intervals for the Log-ODE method.
- `logsig_depth`: The depth of the Log-ODE method. Currently implemented for depths 1 and 2.
- `model_args`: A dictionary of additional arguments to customise the model.
- `num_steps`: The number of steps to train the model.
- `print_steps`: How often to print the loss during training.
- `lr`: The learning rate for the optimiser.
- `lr_scheduler`: The learning rate scheduler function.
- `batch_size`: The number of samples per batch during training.
- `output_parent_dir`: The parent directory where the training outputs will be saved.

The module also includes the following key functions:

- `calc_output`: Computes the model output for batched data.
- `classification_loss`: Computes the loss for classification tasks, including optional regularisation.
- `regression_loss`: Computes the loss for regression tasks, including optional regularisation.
- `make_step`: Performs a single optimisation step, updating model parameters based on the computed gradients.
- `train_model`: Handles the training loop, managing metrics, early stopping, and saving progress at regular intervals.
"""

import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data_dir.datasets import create_dataset
from models.generate_model import create_model


def calc_output(model, X):
    """
    Compute model output for batched data.
    
    Args:
        model: PyTorch model
        X: Input batch
        
    Returns:
        Model output
    """
    model.eval()
    with torch.no_grad():
        if isinstance(X, tuple):
            # Handle tuple inputs (for NCDEs, NRDEs, etc.)
            output = model(X)
        else:
            # Handle regular tensor inputs
            output = model(X)
    return output


def classification_loss_fn(model, X, y):
    """
    Compute classification loss with optional regularization.
    
    Args:
        model: PyTorch model
        X: Input batch
        y: Target labels
        
    Returns:
        Loss value
    """
    pred_y = model(X)
    
    # Cross-entropy loss
    if y.dim() > 1 and y.shape[1] > 1:  # One-hot encoded
        loss = -torch.sum(y * torch.log(pred_y + 1e-8), dim=1).mean()
    else:  # Class indices
        loss = nn.CrossEntropyLoss()(pred_y, y.long())
    
    # Add L2 regularization for Lip2 models
    norm = 0
    if hasattr(model, 'lip2') and model.lip2:
        if hasattr(model, 'vf') and hasattr(model.vf, 'mlp'):
            for layer in model.vf.mlp.layers:
                if hasattr(layer, 'weight'):
                    norm += torch.norm(layer.weight).mean()
                if hasattr(layer, 'bias') and layer.bias is not None:
                    norm += torch.norm(layer.bias).mean()
            if hasattr(model, 'lambd'):
                norm *= model.lambd
    
    return loss + norm


def regression_loss_fn(model, X, y):
    """
    Compute regression loss with optional regularization.
    
    Args:
        model: PyTorch model
        X: Input batch
        y: Target values
        
    Returns:
        Loss value
    """
    pred_y = model(X)
    
    # Handle different output shapes
    if pred_y.dim() > 2:
        pred_y = pred_y[:, :, 0]  # Take first channel if multi-dimensional
    
    # MSE loss
    loss = torch.mean((pred_y - y) ** 2)
    
    # Add L2 regularization for Lip2 models
    norm = 0
    if hasattr(model, 'lip2') and model.lip2:
        if hasattr(model, 'vf') and hasattr(model.vf, 'mlp'):
            for layer in model.vf.mlp.layers:
                if hasattr(layer, 'weight'):
                    norm += torch.norm(layer.weight).mean()
                if hasattr(layer, 'bias') and layer.bias is not None:
                    norm += torch.norm(layer.bias).mean()
            if hasattr(model, 'lambd'):
                norm *= model.lambd
    
    return loss + norm


def make_step(model, X, y, loss_fn, optimizer):
    """
    Perform a single optimization step.
    
    Args:
        model: PyTorch model
        X: Input batch
        y: Target batch
        loss_fn: Loss function
        optimizer: PyTorch optimizer
        
    Returns:
        Loss value
    """
    model.train()
    optimizer.zero_grad()
    
    loss = loss_fn(model, X, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate_model(model, dataloader, batch_size, classification, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        batch_size: Batch size
        classification: Whether it's a classification task
        device: Device to run on
        
    Returns:
        Metric value (accuracy for classification, MSE for regression)
    """
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for data in dataloader.loop_epoch(batch_size):
            X, y = data
            if isinstance(X, tuple):
                X = tuple(x.to(device) for x in X)
            else:
                X = X.to(device)
            y = y.to(device)
            
            prediction = model(X)
            predictions.append(prediction.cpu())
            labels.append(y.cpu())
    
    if len(predictions) == 0:
        return 0.0
        
    prediction = torch.cat(predictions, dim=0)
    y = torch.cat(labels, dim=0)
    
    if classification:
        # Handle one-hot encoded labels
        if y.dim() > 1 and y.shape[1] > 1:
            y_pred = torch.argmax(prediction, dim=1)
            y_true = torch.argmax(y, dim=1)
        else:
            y_pred = torch.argmax(prediction, dim=1)
            y_true = y.long()
        metric = torch.mean((y_pred == y_true).float()).item()
    else:
        if prediction.dim() > 2:
            prediction = prediction[:, :, 0]
        metric = torch.mean((prediction - y) ** 2).item()
    
    return metric


def train_model(
    dataset_name,
    model,
    metric,
    dataloaders,
    num_steps,
    print_steps,
    lr,
    lr_scheduler,
    batch_size,
    output_dir,
    device,
    id=None,
):
    """
    Train the model.
    
    Args:
        dataset_name: Name of the dataset
        model: PyTorch model
        metric: Evaluation metric ('accuracy' or 'mse')
        dataloaders: Dictionary of dataloaders
        num_steps: Number of training steps
        print_steps: How often to print progress
        lr: Learning rate
        lr_scheduler: Learning rate scheduler function
        batch_size: Batch size
        output_dir: Output directory
        device: Device to train on
        id: Optional run ID
        
    Returns:
        Trained model
    """
    
    if metric == "accuracy":
        best_val = max
        operator_improv = lambda x, y: x >= y
        operator_no_improv = lambda x, y: x <= y
        classification = True
        loss_fn = classification_loss_fn
    elif metric == "mse":
        best_val = min
        operator_improv = lambda x, y: x <= y
        operator_no_improv = lambda x, y: x >= y
        classification = False
        loss_fn = regression_loss_fn
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Handle output directory
    if os.path.isdir(output_dir):
        user_input = input(
            f"Warning: Output directory {output_dir} already exists. Do you want to delete it? (yes/no): "
        )
        if user_input.lower() == "yes":
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            print(f"Directory {output_dir} has been deleted and recreated.")
        else:
            raise ValueError(f"Directory {output_dir} already exists. Exiting.")
    else:
        os.makedirs(output_dir)
        print(f"Directory {output_dir} has been created.")

    # Move model to device
    model = model.to(device)
    
    # Setup optimizer
    if callable(lr_scheduler):
        optimizer = optim.Adam(model.parameters(), lr=lr_scheduler(lr))
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize tracking variables
    running_loss = 0.0
    if metric == "accuracy":
        all_val_metric = [0.0]
        all_train_metric = [0.0]
        val_metric_for_best_model = [0.0]
    elif metric == "mse":
        all_val_metric = [100.0]
        all_train_metric = [100.0]
        val_metric_for_best_model = [100.0]
    
    no_val_improvement = 0
    all_time = []
    start = time.time()
    
    # Training loop
    step = 0
    data_iter = iter(dataloaders["train"].loop(batch_size))
    
    while step < num_steps:
        try:
            X, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloaders["train"].loop(batch_size))
            X, y = next(data_iter)
        
        # Move data to device
        if isinstance(X, tuple):
            X = tuple(x.to(device) for x in X)
        else:
            X = X.to(device)
        y = y.to(device)
        
        # Training step
        value = make_step(model, X, y, loss_fn, optimizer)
        running_loss += value
        
        if (step + 1) % print_steps == 0:
            # Evaluate on training set
            train_metric = evaluate_model(
                model, dataloaders["train"], batch_size, classification, device
            )
            
            # Evaluate on validation set
            val_metric = evaluate_model(
                model, dataloaders["val"], batch_size, classification, device
            )
            
            end = time.time()
            total_time = end - start
            print(
                f"Step: {step + 1}, Loss: {running_loss / print_steps:.6f}, "
                f"Train metric: {train_metric:.6f}, "
                f"Validation metric: {val_metric:.6f}, Time: {total_time:.2f}s"
            )
            start = time.time()
            
            # Early stopping and best model tracking
            if step > 0:
                if operator_no_improv(val_metric, best_val(val_metric_for_best_model)):
                    no_val_improvement += 1
                    if no_val_improvement > 10:
                        break
                else:
                    no_val_improvement = 0
                
                if operator_improv(val_metric, best_val(val_metric_for_best_model)):
                    val_metric_for_best_model.append(val_metric)
                    
                    # Evaluate on test set
                    if dataloaders["test"] is not None:
                        test_metric = evaluate_model(
                            model, dataloaders["test"], batch_size, classification, device
                        )
                        print(f"Test metric: {test_metric:.6f}")
                    else:
                        test_metric = 0.0
            else:
                test_metric = 0.0
            
            # Save progress
            running_loss = 0.0
            all_train_metric.append(train_metric)
            all_val_metric.append(val_metric)
            all_time.append(total_time)
            
            # Save metrics
            steps = np.arange(0, step + 1, print_steps)
            np.save(output_dir + "/steps.npy", steps)
            np.save(output_dir + "/all_train_metric.npy", np.array(all_train_metric))
            np.save(output_dir + "/all_val_metric.npy", np.array(all_val_metric))
            np.save(output_dir + "/all_time.npy", np.array(all_time))
            np.save(output_dir + "/test_metric.npy", np.array(test_metric))
        
        step += 1

    print(f"Final test metric: {test_metric:.6f}")

    # Final save
    steps = np.arange(0, num_steps + 1, print_steps)
    np.save(output_dir + "/steps.npy", steps)
    np.save(output_dir + "/all_train_metric.npy", np.array(all_train_metric))
    np.save(output_dir + "/all_val_metric.npy", np.array(all_val_metric))
    np.save(output_dir + "/all_time.npy", np.array(all_time))
    np.save(output_dir + "/test_metric.npy", np.array(test_metric))
    
    # Save final model
    torch.save(model.state_dict(), output_dir + "/final_model.pt")

    return model


def create_dataset_model_and_train(
    seed,
    data_dir,
    use_presplit,
    dataset_name,
    output_step,
    metric,
    include_time,
    T,
    model_name,
    stepsize,
    logsig_depth,
    linoss_discretization,
    model_args,
    num_steps,
    print_steps,
    lr,
    lr_scheduler,
    batch_size,
    output_parent_dir="",
    device=None,
    id=None,
):
    """
    Create dataset, model, and train them.
    
    Args:
        seed: Random seed
        data_dir: Data directory
        use_presplit: Whether to use pre-split data
        dataset_name: Name of dataset
        output_step: Output step for regression
        metric: Evaluation metric
        include_time: Whether to include time
        T: Time scale
        model_name: Model architecture name
        stepsize: Step size for log-signature
        logsig_depth: Log-signature depth
        linoss_discretization: LinOSS discretization method
        model_args: Additional model arguments
        num_steps: Number of training steps
        print_steps: Print frequency
        lr: Learning rate
        lr_scheduler: Learning rate scheduler
        batch_size: Batch size
        output_parent_dir: Output parent directory
        device: Device to use
        id: Run ID
        
    Returns:
        Trained model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory name
    if model_name == 'LinOSS':
        model_name_directory = model_name + '_' + linoss_discretization
    else:
        model_name_directory = model_name
        
    output_parent_dir += "outputs/" + model_name_directory + "/" + dataset_name
    output_dir = f"T_{T:.2f}_time_{include_time}_nsteps_{num_steps}_lr_{lr}"
    
    if model_name == "log_ncde" or model_name == "nrde":
        output_dir += f"_stepsize_{stepsize:.2f}_depth_{logsig_depth}"
    
    for k, v in model_args.items():
        name = str(v)
        if "(" in name:
            name = name.split("(", 1)[0]
        if name == "dt0":
            output_dir += f"_{k}_" + f"{v:.2f}"
        else:
            output_dir += f"_{k}_" + name
    
    output_dir += f"_seed_{seed}"

    print(f"Creating dataset {dataset_name}")
    
    # Create dataset
    dataset = create_dataset(
        data_dir,
        dataset_name,
        stepsize=stepsize,
        depth=logsig_depth,
        include_time=include_time,
        T=T,
        use_idxs=False,
        use_presplit=use_presplit,
        device=device,
    )

    print(f"Creating model {model_name}")
    
    # Create model
    classification = metric == "accuracy"
    model = create_model(
        model_name,
        dataset.data_dim,
        dataset.logsig_dim,
        logsig_depth,
        dataset.intervals,
        dataset.label_dim,
        classification=classification,
        output_step=output_step,
        linoss_discretization=linoss_discretization,
        **model_args,
    )
    
    # Select appropriate dataloaders
    if model_name == "nrde" or model_name == "log_ncde":
        dataloaders = dataset.path_dataloaders
    elif model_name == "ncde":
        dataloaders = dataset.coeff_dataloaders
    else:
        dataloaders = dataset.raw_dataloaders

    return train_model(
        dataset_name,
        model,
        metric,
        dataloaders,
        num_steps,
        print_steps,
        lr,
        lr_scheduler,
        batch_size,
        output_parent_dir + "/" + output_dir,
        device,
        id,
    )
