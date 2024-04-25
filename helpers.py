import sys
import numpy as np
import torch
import math
import tqdm

FUNCTIONS = {'x^(1.2)': lambda x: x**1.2,
             'x^2': lambda x: x**2,
             'x^3-6x^2+8x': lambda x: x**3 - 6*x**2 + 8*x,}

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
    # TODO convert to linspace, and changei nputs around
    x_values = np.arange(0, num_samples * sample_step, sample_step)
    # x_values = np.linspace(0, num_samples * sample_step, num_samples)
    f_values = func(x_values)
    return x_values, f_values

def get_deriv_approx(x_values, func, n_deriv):
    """
    Computes the nth derivative of a given function at specified x values using
    automatic differentiation. The derivative is computed using the autograd.grad() 
    function, which computes the partial gradient of a function with respect to 
    its inputs. By specifying grad_outputs interally to be ones_like(y), we are
    calculating the partial gradient across ALL inputs, effectively outputting 
    a full derivative. Technically uses finite differences.

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

########## GL method https://www.youtube.com/watch?v=opJVMMPb-EI ##########

def get_frac_deriv_approx_GL(x_values, func, n_deriv, h):
    """
    Computes the nth derivative of a function using the generalized GL formula.

    Args:
        x_values (list): List of x values to compute the derivative at.
        func (function): Function to compute the derivative of.
        n_deriv (int): Order of the derivative to compute. Also known as alpha
        h (float): Step size for the approximation.

    Returns:
        list: List of nth derivative values for each x value in x_values.

    Formula: 
        ![image](https://quicklatex.com/cache3/89/ql_52404e9c550962bb8b518396b5d24589_l3.png)

    where:
        - alpha = n_deriv (All Real numbers)
        - gamma is the gamma function
        - j! is the factorial of j
        - h is the step size
    """

    summation = 0.0
    y_values = []
    x_values = np.array(x_values)
    for i in tqdm.tqdm(range(len(x_values)), desc="Iterating through x values"):
            summation = 0
            x = x_values[i]
            N =  int(x/h)
            for j in range(N): # Start of summation

                # Computing the Gamma function fraction part of GL formula
                gamma_top = (math.gamma(n_deriv + 1))
                gamma_bot = (math.gamma(n_deriv - j + 1))
                if gamma_bot != 0.0: # Invalid values of gamma func will be 0.0
                    try:
                        alpha_j = (gamma_top) / (math.factorial(j) * gamma_bot)
                    except OverflowError:
                        # Assume if factorial is too large, it is infinity  
                        alpha_j = (gamma_top) / (math.inf * gamma_bot)
                
                summation += (((-1) ** j) * float(alpha_j) * func(x - (h * j)))
            
            y_values.append(((1 / h) ** n_deriv) * summation)
    return y_values

def C(n, alpha):
    C_ = (((((-1) ** (n - 1)) * alpha * math.gamma(n - alpha))/((math.gamma(1 - alpha) * math.gamma(n + 1)))) * (1 / math.gamma(n + 1 - alpha)))
    return C_

########## RL Method from https://arxiv.org/pdf/1208.2588.pdf Equation (4)##########

def FracGradApproxRL(inp_, alpha, h , sample_step = 0.01, f_sympy = None):
    # N = math.ceil(alpha) + 3 # The approximation gets more accuracte as N -> inf
    N = 10
    if f_sympy == None:
        f_sympy = f
    x_ = symbols('x')
    f_ = f_sympy(x_)
    attr = torch.tensor(inp_).clone().detach()
    attr = np.array(attr)
    inp = torch.tensor(inp_).clone().detach()
    inp = np.array(inp)
    for i in range(len(inp)):
            a = -h#inp_[i] - 0.00001
            summation = 0.0
            t = i * sample_step#inp[i] 
#            print("inp_[", i, "] = ",inp_[i])
            for n in range(N):
#                print(n)
                C_ = C(n, alpha)
                dx = diff(f_, x_, n)
                x_deriv = dx.subs(x_,t)# * 0.000001
                
#                print("n = ", n, "  x_deriv = ", x_deriv)
                
                summation += C_ * ((t - a) ** (n - alpha)) * (x_deriv)
            
            attr[i] = summation
    return attr.tolist()

