import numpy as np
import sympy as sp
from tabulate import tabulate

def is_diagonally_dominant(A):
    return all(abs(A[i, i]) >= sum(abs(A[i, j]) for j in range(len(A)) if j != i) for i in range(len(A)))
def generate_random_system(n, min_val=1, max_val=10):
    while True:
        A = np.random.uniform(min_val, max_val, (n, n))
        b = np.random.uniform(min_val, max_val, n)
        for i in range(n):
            A[i, i] = sum(abs(A[i, j]) for j in range(n) if j != i) + np.random.uniform(1, max_val)
        if is_diagonally_dominant(A):
            return A, b
def format_matrix(A, b=None):
    table = [list(np.round(A[i], 5)) + ([round(b[i], 5)] if b is not None else []) for i in range(len(A))]
    headers = [f"x{i+1}" for i in range(len(A))] + (["b"] if b is not None else [])
    return tabulate(table, headers=headers, tablefmt="fancy_grid")
def get_float_input(prompt, default):
    val = input(prompt)
    return float(val) if val.strip() != "" else default
def get_int_input(prompt, default):
    val = input(prompt)
    return int(val) if val.strip() != "" else default
def solve_system(A, b, method, tol=1e-6, max_iter=100):
    # Solve Ax = b using Jacobi or Gauss-Seidel method with detailed output.
    n, x = len(b), np.zeros(len(b))
    print(f"\nSolving using {method} Method...")
    headers = ["Iteration"] + [f"x{i+1}" for i in range(n)] + ["Error"]
    iterations = []
    if method == "Jacobi":
        D = np.diag(A)
        R = A - np.diagflat(D)
        for k in range(max_iter):
            x_new = (b - np.dot(R, x)) / D
            error = np.linalg.norm(x_new - x, ord=np.inf)
            iterations.append([k+1] + list(np.round(x_new, 5)) + [f"{error:.6e}"])
            if error < tol:
                break
            x = x_new
    elif method == "Gauss-Seidel":
        for k in range(max_iter):
            x_new = np.copy(x)
            for i in range(n):
                s1, s2 = np.dot(A[i, :i], x_new[:i]), np.dot(A[i, i+1:], x[i+1:])
                x_new[i] = (b[i] - s1 - s2) / A[i, i]
            error = np.linalg.norm(x_new - x, ord=np.inf)
            iterations.append([k+1] + list(np.round(x_new, 5)) + [f"{error:.6e}"])
            if error < tol:
                break
            x = x_new
    print("\nIteration Process:")
    print(tabulate(iterations, headers=headers, tablefmt="fancy_grid"))
    if error >= tol:
        print("\nIterative method did not converge within the max iterations.")
    return x_new
def solve_linear_system():
    print("\nSystem of Linear Equations Solver")
    print("1. Manually input the coefficient matrix and vectors.")
    print("2. Randomly generate a diagonally dominant system.")

    choice = input("Choose an option (1 or 2): ").strip()
    if choice not in {"1", "2"}:
        print("Invalid choice! Exiting.")
        return
    n = get_int_input("\nEnter the number of variables: ", 3)
    if n <= 0:
        print("Error: Number of variables must be positive!")
        return
    if choice == "1":
        print("\nEnter the coefficient matrix row by row (space-separated):")
        A = np.array([list(map(float, input(f"Row {i+1}: ").split())) for i in range(n)])
        b = np.array(list(map(float, input("\nEnter the right-hand side vector (space-separated): ").split())))
    else:
        A, b = generate_random_system(n)
        print("\nGenerated System of Equations:")
        print(format_matrix(A, b))
    if np.any(np.diag(A) == 0):
        print("Error: Matrix contains zero diagonal elements. Cannot proceed.")
        return
    tol = get_float_input("\nEnter tolerance (default 1e-6): ", 1e-6)
    max_iter = get_int_input("Enter max iterations (default 100): ", 100)
    while True:
        print("\nChoose a method to solve:")
        print("1. Jacobi Method")
        print("2. Gauss-Seidel Method")
        method_choice = input("Enter choice (1 or 2): ").strip()
        method = {"1": "Jacobi", "2": "Gauss-Seidel"}.get(method_choice)
        if not method:
            print("Invalid method choice! Exiting.")
            return
        solution = solve_system(A, b, method, tol, max_iter)
        print("\n Final Approximate Solution (Iterative Method):")
        print(tabulate([solution], headers=[f"x{i+1}" for i in range(n)], tablefmt="fancy_grid"))
        print("\n Direct Method Solution (Verification):")
        try:
            x_direct = np.linalg.solve(A, b)
            print(tabulate([x_direct], headers=[f"x{i+1}" for i in range(n)], tablefmt="fancy_grid"))
        except np.linalg.LinAlgError:
            print("Direct solution failed due to a singular or ill-conditioned matrix.")
if __name__ == "__main__":
    solve_linear_system()
