import random
import numpy as np
import matplotlib.pyplot as plt

def get_input_or_random(prompt, use_random=False, min_val=-10, max_val=10):
    return (random.uniform(-1, 1) if 'boundary' in prompt.lower() else
            random.randint(10, 100) if 'grid points' in prompt.lower() else
            random.uniform(min_val, max_val)) if use_random else get_user_input(prompt)
def get_user_input(prompt):
    while True:
        try:
            value = input(prompt).strip()
            if value == '': raise ValueError("Input cannot be empty.")
            return float(value)
        except ValueError as e:
            print(f"Invalid input. Please enter a valid number. Error: {e}")
def solve_bvp(N, a=0, b=1, alpha=0, beta=0):
    h = (b - a) / (N + 1)
    x = np.linspace(a, b, N + 2)
    f = -np.pi**2 * np.sin(np.pi * x)
    f[0], f[-1] = alpha, beta
    A = np.diag(-2 * np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)
    A /= h**2
    b = f[1:-1]
    y = np.linalg.solve(A, b)
    full_y = np.concatenate(([alpha], y, [beta]))
    return x, full_y

if __name__ == "__main__":
    print("1. Manually input the number of grid points (N) and boundary conditions.")
    print("2. Randomly generate the suitable number of grid points (N) and boundary conditions.\n")
    choice = int(input("Choose an option (1 or 2): "))
    if choice == 1:
    	use_random = False
    elif choice == 2:
    	use_random = True
    else:
    	print("\nInvalid Choice Made! Try Again... \n")
    a = get_input_or_random("Enter the lower boundary (a): ", use_random)
    b = get_input_or_random("Enter the upper boundary (b): ", use_random)
    if a >= b:
        print("Invalid limits: The lower limit must be less than the upper limit. Exiting.")
        exit()
    alpha = get_input_or_random("Enter the value for the lower boundary condition (alpha): ", use_random)
    beta = get_input_or_random("Enter the value for the upper boundary condition (beta): ", use_random)
    N = int(get_input_or_random("Enter the number of interior grid points (N): ", use_random, 10, 100))
    print(f"\nValues used:")
    print(f"a (lower boundary): {a}")
    print(f"b (upper boundary): {b}")
    print(f"alpha (lower boundary condition): {alpha}")
    print(f"beta (upper boundary condition): {beta}")
    print(f"N (number of interior grid points): {N}")
    if use_random:
        print(f"Randomized values used: \na = {a:.3f}, b = {b:.3f}, alpha = {alpha:.3f}, beta = {beta:.3f}, N = {N}")
    while True:
        print("\nFinite Difference Method for Boundary Value Problem")
        print("---------------------------------------------------")
        print("1. Solve the Boundary Value Problem using FDM")
        print("2. Exit")
        method = input("Enter your choice: ").strip()
        if method == '1':
            print("\nSolving the Boundary Value Problem using Finite Difference Method...")
            x, y = solve_bvp(N, a=a, b=b, alpha=alpha, beta=beta)
            print(f"Solution at grid points: {list(zip(x, y))}\n")
            plt.plot(x, y, label="FDM Solution")
            plt.xlabel("x")
            plt.ylabel("y(x)")
            plt.title("Solution of Boundary Value Problem using Finite Difference Method (FDM)")
            plt.grid(True)
            plt.legend()
            plt.show()
        elif method == '2':
            print("\nGoodBye.")
            print("---------------------------------------------------")
            break
        else:
            print("\n---------------------------------------------------")
            print("\tInvalid method selected! Try again...")
            print("---------------------------------------------------")
            continue
