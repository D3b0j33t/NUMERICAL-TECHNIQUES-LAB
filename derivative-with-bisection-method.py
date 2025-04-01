import sympy as sp

def bisection_method(func, a, b, tolerance=1e-7, max_iter=100):
    if func(a) * func(b) >= 0:
        print("The bisection method cannot be applied because the signs of func(a) and func(b) are not opposite.")
        return None, 0, 0

    iteration = 0
    while (b - a) / 2 > tolerance and iteration < max_iter:
        c = (a + b) / 2
        if func(c) == 0:
            return c, iteration, (b - a) / 2
        (b, a) = (c, a) if func(c) * func(a) < 0 else (b, c)
        iteration += 1

    return (a + b) / 2, iteration, (b - a) / 2

def central_difference_derivative(func, x, h=1e-5):
    return (func(x + h) - func(x - h)) / (2 * h)

def parse_function(expression):
    x = sp.symbols('x')
    expr = sp.sympify(expression)
    return sp.lambdify(x, expr, "numpy")

def find_interval_for_bisection(func, x_start=-10, x_end=10, step_size=0.1):
    a, b = x_start, x_start + step_size
    while b <= x_end:
        if func(a) * func(b) < 0:
            return a, b
        a, b = a + step_size, b + step_size

    print("No sign change found within the specified range.")
    return None, None

def main():
    equation_str = input("Enter the equation (in terms of x, e.g., 'x**3 + x**2 + x + 7', 'sin(x)', 'log(x)'):\n> ")
    func = parse_function(equation_str)

    a, b = find_interval_for_bisection(func)

    if a is not None and b is not None:
        print(f"Found suitable interval for bisection method: [{a}, {b}]")
        tolerance = float(input("Enter the desired tolerance (e.g., 0.1e-3): ") or 0.1e-3)
        root, iterations, error_bound = bisection_method(func, a, b, tolerance)

        if root is not None:
            print(f"Root found using Bisection Method: {root:.6f}")
            print(f"Number of iterations: {iterations}")
            print(f"Error bound: {error_bound:.6f}")

        x_point = float(input("Enter the point at which to calculate the derivative: "))
        derivative = central_difference_derivative(func, x_point)
        print(f"The numerical derivative of the function at x = {x_point} is approximately: {derivative:.6f}")
    else:
        print("Couldn't find a suitable interval for the Bisection Method.")

if __name__ == "__main__":
main()
