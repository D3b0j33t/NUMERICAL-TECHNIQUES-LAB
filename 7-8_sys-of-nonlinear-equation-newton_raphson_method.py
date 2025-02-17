import sympy as sp
import numpy as np
import random

def newton_raphson_method_system(funcs, jacobian, x0, tolerance=1e-7, max_iter=100):
    iteration = 0
    x0 = np.array(x0, dtype=float)
    
    while iteration < max_iter:
        # Evaluate the function values F(x0) and the Jacobian matrix J(x0)
        F_values = np.array([float(f(*x0)) for f in funcs], dtype=float)  # Evaluate functions numerically
        J_matrix = np.array(jacobian(*x0), dtype=float)  # Evaluate the Jacobian numerically
        
        try:
            # Solve the system J(x0) * delta = F(x0) for delta (the correction)
            delta = np.linalg.solve(J_matrix, -F_values)
        except np.linalg.LinAlgError:
            print("Error: Jacobian matrix is singular. The method failed.")
            return None, iteration
        
        # Update the guess
        x1 = x0 + delta
        
        # Check for convergence based on the residual (F_values) instead of just delta
        if np.linalg.norm(F_values) < tolerance:  # Residual check
            return x1, iteration
        
        # Debug: Print the current guess values for each variable
        # print(f"Iteration {iteration + 1}: x1 = {x1[0]}, x2 = {x1[1]}, x3 = {x1[2]}")

        x0 = x1
        iteration += 1
    
    print("Maximum iterations reached.")
    return x1, iteration

def parse_system_of_equations(equations, vars):
    funcs = []
    for eq in equations:
        funcs.append(sp.lambdify(vars, sp.sympify(eq), "numpy"))
    return funcs

def parse_jacobian(equations, vars):
    jacobian = []
    for eq in equations:
        jacobian_row = []
        for var in vars:
            jacobian_row.append(sp.diff(sp.sympify(eq), var))
        jacobian.append(jacobian_row)
    jacobian_func = sp.lambdify(vars, jacobian, "numpy")
    return jacobian_func

def generate_initial_guess(num_vars):
    # Automatically generate a random initial guess in the range [-10, 10] for each variable
    return [random.uniform(-10, 10) for _ in range(num_vars)]

if __name__ == "__main__":
    # User inputs the system of equations
    num_equations = int(input("Enter the number of equations in the system [DEFAULT=2]: ") or 2)
    
    equations = []
    for i in range(num_equations):
        eq = input(f"Enter equation {i+1} (use variables x1, x2, ..., xn): ")
        equations.append(eq)
    
    # Define the variables dynamically
    vars = sp.symbols(f'x1:{num_equations + 1}')  # This generates x1, x2, ..., xn for n variables
    
    # Parse the system of equations and the Jacobian
    funcs = parse_system_of_equations(equations, vars)
    jacobian = parse_jacobian(equations, vars)
    
    # Initial guess
    x0_input = input(f"Enter the initial guess (e.g., 'x1, x2, ..., xn'): ")
    if x0_input.strip():
        x0 = [float(i) for i in x0_input.split(',')]
    else:
        # Generate random initial guess if the user leaves it blank
        x0 = generate_initial_guess(num_equations)
        print(f"Auto-generated initial guess: {x0}")
    
    tolerance = float(input("Enter the desired tolerance (e.g., 0.1e-3): ") or 0.1e-3)
    
    # Apply Newton-Raphson Method to solve the system
    solution, iterations = newton_raphson_method_system(funcs, jacobian, x0, tolerance)
    
    if solution is not None:
        # Explicitly formatting each solution variable to ensure full visibility
        formatted_solution = [f"{sol:.6f}" for sol in solution]
        print(f"Solution found using Newton-Raphson Method: {', '.join(formatted_solution)}")
        print(f"Number of iterations: {iterations}")
