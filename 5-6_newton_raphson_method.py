import sympy as sp

def newton_raphson_method(func, func_prime, x0, tolerance=1e-7, max_iter=100):
    iteration = 0
    while iteration < max_iter:
        # Calculate the next approximation using Newton-Raphson formula
        try:
            x1 = x0 - func(x0) / func_prime(x0)
        except ZeroDivisionError:
            print("Error: Division by zero. The method failed.")
            return None, iteration
        
        if abs(x1 - x0) < tolerance:
            return x1, iteration
        
        x0 = x1
        iteration += 1
    
    print("Maximum iterations reached.")
    return x1, iteration

def parse_function(expression):
    x = sp.symbols('x')
    expr = sp.sympify(expression)
    func = sp.lambdify(x, expr, "numpy")
    return func

def find_initial_guess(func):
    # Sample points to evaluate the function
    sample_points = [-10, -5, 0, 5, 10]
    
    prev_value = func(sample_points[0])
    
    # Loop through points and find a sign change
    for point in sample_points[1:]:
        current_value = func(point)
        
        if prev_value * current_value < 0:  # Sign change detected
            return (point + sample_points[sample_points.index(point) - 1]) / 2  # Choose middle point
            
        prev_value = current_value
    
    # If no sign change is detected, use 0 as the initial guess
    return 0

if __name__ == "__main__":
    # User inputs the equation for f(x)
    equation_str = input("Enter the equation function f(x): (e.g., 'x**3 + x**2 + x + 7'): ")
    
    # Parse the function f(x) from the input string
    func = parse_function(equation_str)
    
    # Compute the derivative of the function
    x = sp.symbols('x')
    expr = sp.sympify(equation_str)
    func_prime = sp.lambdify(x, sp.diff(expr, x), "numpy")
    
    tolerance = float(input("Enter the desired tolerance (default is 0.1e-3): ") or 0.1e-3)
    
    # Ask the user for the initial guess
    x0_input = input("Enter the initial guess x0 (leave empty for automated guess): ")
    
    if x0_input.strip():  # If the user enters a value
        x0 = float(x0_input)
    else:
        # Automatically determine the initial guess x0
        x0 = find_initial_guess(func)
        print(f"\nSmartly Automated initial guess x0: {x0}")
    
    # Apply Newton-Raphson Method
    root, iterations = newton_raphson_method(func, func_prime, x0, tolerance)
    
    if root is not None:
        print(f"\nRoot found using Newton-Raphson Method: {root:.4f}")
        print(f"Number of iterations: {iterations}")
