import sympy as sp
import numpy as np

# Secant Method implementation
def secant_method(func, x0, x1, tolerance=1e-7, max_iter=100):
    iteration = 0
    while abs(x1 - x0) > tolerance and iteration < max_iter:
        try:
            x_new = x1 - func(x1) * (x1 - x0) / (func(x1) - func(x0))
        except ZeroDivisionError:
            print("Error: Division by zero. The method failed.")
            return None, iteration

        x0, x1 = x1, x_new
        iteration += 1
    return x1, iteration

# Regula Falsi Method implementation
def regula_falsi_method(func, a, b, tolerance=1e-7, max_iter=100):
    iteration = 0
    while (b - a) / 2 > tolerance and iteration < max_iter:
        c = b - func(b) * (b - a) / (func(b) - func(a))
        if func(c) == 0:
            return c, iteration
        a, b = (a, c) if func(c) * func(a) < 0 else (c, b)
        iteration += 1
    return (a + b) / 2, iteration

# Function to parse the equation into a sympy function
def parse_function(expression):
    x = sp.symbols('x')
    expr = sp.sympify(expression)
    return sp.lambdify(x, expr, "numpy")

# Function to find a suitable interval for the method
def find_interval_for_method(func, x_start=-10, x_end=10, step_size=0.1):
    a, b = x_start, x_start + step_size
    while b <= x_end:
        if func(a) * func(b) < 0:
            return a, b
        a, b = a + step_size, b + step_size
    print("No sign change found within the specified range.")
    return None, None

# Function to get initial guesses for methods
def smart_initial_guesses(func, x_start=-10, x_end=10, step_size=0.1):
    x0, x1 = None, None
    for x in np.arange(x_start, x_end, step_size):
        # Sign change detected using ternary-like logic
        if func(x) * func(x + step_size) < 0:  
            x0, x1 = x, x + step_size
            break
    return (x0, x1) if x0 is not None and x1 is not None else (None, None)

# Display method options
def display_menu():
    print("\nChoose a method for root finding:")
    print("1. Secant Method")
    print("2. Regula Falsi Method")
    print("3. Exit")

if __name__ == "__main__":
    equation_str = input("Enter the equation function f(x): (e.g., 'x**3 + x**2 + x + 7'): ")
    
    func = parse_function(equation_str)
    
    # Find interval using ternary logic to check if it's found
    a, b = find_interval_for_method(func)
    if a is not None and b is not None:
        print(f"Found suitable interval: [{a}, {b}]")
        
        tolerance = float(input("Enter the desired tolerance (default is 0.1e-3): ") or 0.1e-3)
        
        while True:
            display_menu()
            method_choice = input("\nEnter the number corresponding to your choice: ").strip()

            if method_choice == '1':
                # Get initial guesses for Secant method using ternary check
                x0, x1 = smart_initial_guesses(func)
                if x0 is not None and x1 is not None:
                    root, iterations = secant_method(func, x0, x1, tolerance)
                    print(f"\nRoot found using Secant Method: {root:.4f}") if root is not None else print("Root not found.")
                    print(f"Number of iterations: {iterations}")
                
            elif method_choice == '2':
                root, iterations = regula_falsi_method(func, a, b, tolerance)
                print(f"\nRoot found using Regula Falsi Method: {root:.4f}") if root is not None else print("Root not found.")
                print(f"Number of iterations: {iterations}")
                
            elif method_choice == '3':
                print("Program successfully Terminated.")
                break
            
            else:
                print("Invalid choice, please try again.")
    
    else:
        print("Couldn't find a suitable interval for the method.")
