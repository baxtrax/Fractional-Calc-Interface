import numpy as np
import torch

FUNCTIONS = {'x^2': lambda x: x**2,}

def generate_function_data(func, num_samples, sample_step):
    """
    Generates sample data for a given function.

    Parameters:
        func: A function that takes a single argument.
        num_samples: The number of samples to generate.
        sample_step: The step size between each sample.

    Returns:
        A tuple containing two lists: the x values and the corresponding function values.
    """
    x_values = np.arange(0, num_samples * sample_step, sample_step)
    f_values = func(x_values)
    return x_values, f_values

def get_deriv_approx(x_values, func, n_deriv):
    """
    Computes the nth derivative of a given function at specified x values using
    automatic differentiation. The derivative is computed using the autograd.grad() 
    function, which computes the partial gradient of a function with respect to 
    its inputs. By specifying grad_outputs interally to be ones_like(y), we are
    calculating the partial gradient across ALL inputs, effectively outputting 
    a full derivative.

    Parameters:
        x_values (list or numpy array): The x values at which to compute the derivative.
        func (callable): The function to differentiate.
        n_deriv (int): The order of the derivative to compute.

    Returns:
        numpy array: The values of the nth derivative of the function at the specified x values.
    """
    # Ensure x is a tensor
    x = torch.tensor(x_values, requires_grad=True, dtype=torch.float32)
    y = func(x)


    # Compute the first derivative for each element in y
    derivatives = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    
    # Calculate higher-order derivatives iteratively
    for _ in range(1, n_deriv):
        if derivatives.grad_fn is None:
            raise ValueError(f"Derivative order {n_deriv} invalid for selected function.")
        derivatives = torch.autograd.grad(derivatives, x, torch.ones_like(y), create_graph=True)[0]
    
    # Extract the derivative values as a numpy array
    derivative_values = derivatives.detach().numpy()
    
    return derivative_values