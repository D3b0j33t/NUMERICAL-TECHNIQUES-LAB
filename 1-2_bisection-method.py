import sympy as sp

# Bisection method implementation
def bisection_method(func, a, b, tolerance=1e-7, max_iter=100):
    if func(a) * func(b) >= 0:
        print("The bisection method cannot be applied because the signs of func(a) and func(b) are not opposite.")
        return None, 0, 0

    iteration = 0
    while (b - a) / 2 > tolerance and iteration < max_iter:
        c = (a + b) / 2
        func_c = func(c)

        # Ternary operations to update bounds and check for root
        if func_c == 0: 
            return c, iteration, (b - a) / 2
        a, b = (a, c) if func_c * func(a) < 0 else (c, b)
        iteration += 1

    error_bound = (b - a) / 2
    return (a + b) / 2, iteration, error_bound

# Function to parse user input into a sympy function
def parse_function(expression):
    x = sp.symbols('x')
    expr = sp.sympify(expression)
    return sp.lambdify(x, expr, "numpy")

# Function to find a suitable interval for the Bisection method
def find_interval_for_bisection(func, x_start=-10, x_end=10, step_size=0.1):
    a, b = x_start, x_start + step_size
    while b <= x_end:
        if func(a) * func(b) < 0:
            return a, b
        a, b = a + step_size, b + step_size

    print("No sign change found within the specified range.")
    return None, None

# Main program execution
if __name__ == "__main__":
    equation_str = input("Enter the equation function f(x): (e.g., 'x**3 + x**2 + x + 7'): ")
    func = parse_function(equation_str)
    
    a, b = find_interval_for_bisection(func)
    
    # Ternary operation to check if suitable interval is found
    if a is not None and b is not None:
        print(f"Found suitable interval for bisection method: [{a}, {b}]")
        
        tolerance = float(input("Enter the desired tolerance (e.g., 0.1e-3): ") or 0.1e-3)
        
        root, iterations, error_bound = bisection_method(func, a, b, tolerance)
        
        # Displaying the results
        if root is not None:
            print(f"Root found using Bisection Method: {root:.4f}")
            print(f"Number of iterations: {iterations}")
            print(f"Error bound: {error_bound:.6f}")
    else:
        print("Couldn't find a suitable interval for the Bisection Method.")
