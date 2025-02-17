import random
from sympy import symbols, sympify, lambdify
from scipy.integrate import quad

# Function to get input or generate random values
def get_input_or_random(prompt, use_random=False, min_val=-10, max_val=10):
    return random.uniform(min_val, max_val) if use_random else get_user_input(prompt)

def get_user_input(prompt):
    while True:
        try:
            value = input(prompt).strip()
            if value == '': raise ValueError("Input cannot be empty.")
            return float(value)
        except ValueError as e:
            print(f"Invalid input. Please enter a valid number. Error: {e}")

# Function to parse the equation function
def parse_function(function_str):
    try:
        x = symbols('x')
        func = sympify(function_str)
        return lambdify(x, func, "numpy")
    except Exception as e:
        print(f"Error parsing the function: {e}")
        return None

if __name__ == "__main__":
    print("1. Manually input the lower and upper limits.")
    print("2. Randomly generate the suitable lower and upper limits.\n")

    choice = input("Choose an option (1 or 2): ").strip()
    use_random = True if choice == '2' else False

    # Get the equation for f(x)
    equation_str = input("Enter the equation function f(x): (e.g., 'x**3 + x**2 + x + 7'): ")

    # Parse the function f(x)
    func = parse_function(equation_str)
    if not func:
        print("Exiting due to invalid function.")
        exit()

    # Get limits of integration
    a = get_input_or_random("Enter the lower limit of integration (a): ", use_random)
    b = get_input_or_random("Enter the upper limit of integration (b): ", use_random)

    # Ensure valid limits
    if use_random:
        print(f"Randomized limits used: a = {a:.3f}, b = {b:.3f}")
    elif a >= b:
        print("Invalid limits: The lower limit must be less than the upper limit. Exiting.")
        exit()

    while True:
        print("\nGaussian Quadrature and Romberg Integration")
        print("------------------------------------------------")
        print("1. Perform Gaussian Quadrature Integration")
        print("2. Perform Romberg Integration")
        print("3. Exit")
        
        # Method selection using ternary operator
        method = input("Enter your choice: ").strip()
        
        if method == '1':
            print("\nUsing Gaussian Quadrature Method...")
            try:
                result, _ = quad(func, a, b)
                print(f"Result using Gaussian Quadrature: {result:.3f}\n")
            except Exception as e:
                print(f"Error with integration: {e}")
        elif method == '2':
            print("\nUsing Romberg Integration Method (via quad with higher accuracy)...")
            try:
                result, _ = quad(func, a, b, epsrel=1e-8)
                print(f"Result using Romberg Integration: {result:.3f}\n")
            except Exception as e:
                print(f"Error with integration: {e}")
        elif method == '3':
            print("\nGoodBye.")
            print("------------------------------------------------")
            break
        else:
            print("\n------------------------------------------------")
            print("\tInvalid method selected! Try again...")
            print("------------------------------------------------")
            continue
