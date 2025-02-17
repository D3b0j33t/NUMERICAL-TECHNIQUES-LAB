import numpy as np

# Check if a matrix is diagonally dominant
def is_diagonally_dominant(A):
    return all(abs(A[i, i]) >= sum(abs(A[i, j]) for j in range(len(A)) if j != i) for i in range(len(A)))

# Generate a random diagonally dominant matrix and vector
def generate_random_system(n, min_val=1, max_val=10):
    while True:
        A = np.random.uniform(min_val, max_val, (n, n))  # Random floats in [min_val, max_val)
        b = np.random.uniform(min_val, max_val, n)
        # Make the matrix diagonally dominant
        for i in range(n):
            A[i, i] = sum(abs(A[i, j]) for j in range(n) if j != i) + np.random.uniform(1, max_val)
        if is_diagonally_dominant(A):
            return A, b

# Solver function
def solve_linear_system():
    print("System of Linear Equations Solver\n")
    print("1. Manually input the coefficient matrix and vectors.")
    print("2. Randomly generate the coefficient matrix and vectors.\n")

    choice = int(input("Choose an option (1 or 2): "))
    if choice not in [1, 2]:
        print("Invalid choice! Exiting.")
        return

    n = int(input("\nEnter the number of variables: "))
    if n <= 0:
        print("Error: Number of variables must be positive!")
        return

    if choice == 1:  # Manual input
        print("\nEnter the coefficient matrix row by row (space-separated):")
        A = np.array([list(map(float, input(f"Row {i + 1}: ").split())) for i in range(n)])
        print("\nEnter the right-hand side vector (space-separated):")
        b = np.array(list(map(float, input().split())))
        x = np.array(list(map(float, input("\nEnter initial guesses (space-separated): ").split())))
    else:  # Random generation
        A, b = generate_random_system(n, 1, 10)
        x = np.zeros(n)  # Default initial guess
        print("\nRandomly generated coefficient matrix (diagonally dominant):")
        print(A)
        print("\nRandomly generated right-hand side vector:")
        print(b)
        print("\nInitial guesses (default: zeros):")
        print(x)

    # Check for valid matrix dimensions
    if A.shape[0] != A.shape[1] or A.shape[0] != len(b):
        print("Error: The coefficient matrix must be square and compatible with the right-hand side vector.")
        return

    # Check for zero diagonal elements
    if np.any(np.diag(A) == 0):
        print("Error: Matrix contains zero diagonal elements. Cannot proceed.")
        return

    # Get tolerance and max iterations
    tol = float(input("\nEnter the tolerance (default is 1e-6): ") or 1e-6)
    max_iter = int(input("Enter the maximum number of iterations (default is 100): ") or 100)

    print("\nChoose a method to solve the system of equations:")
    print("1. Jacobi Method")
    print("2. Gauss-Seidel Method")
    method = int(input("Enter your choice (1 or 2): "))

    if method not in [1, 2]:
        print("Invalid method choice! Exiting.")
        return

    if method == 1:
        print("\nUsing Jacobi Method...")
        D = np.diag(A)
        R = A - np.diagflat(D)

        for iteration in range(max_iter):
            x_new = (b - np.dot(R, x)) / D
            error = np.linalg.norm(x_new - x, ord=np.inf)
            print(f"Iteration {iteration + 1}: x = {x_new}, Error = {error:.6e}")
            if error < tol:
                print("\nSolution found using Jacobi Method:")
                print(f"x = {x_new}")
                return x_new
            x = x_new

        print("\nJacobi Method did not converge within the maximum number of iterations.")
        print("Best approximation so far:")
        print(f"x = {x_new}")
        return x_new

    elif method == 2:
        print("\nUsing Gauss-Seidel Method...")
        for iteration in range(max_iter):
            x_new = np.copy(x)
            for i in range(n):
                s1 = np.dot(A[i, :i], x_new[:i])  # Sum of known values
                s2 = np.dot(A[i, i + 1:], x[i + 1:])  # Sum of unknown values
                x_new[i] = (b[i] - s1 - s2) / A[i, i]
            error = np.linalg.norm(x_new - x, ord=np.inf)
            print(f"Iteration {iteration + 1}: x = {x_new}, Error = {error:.6e}")
            if error < tol:
                print("\nSolution found using Gauss-Seidel Method:")
                print(f"x = {x_new}")
                return x_new
            x = x_new

        print("\nGauss-Seidel Method did not converge within the maximum number of iterations.")
        print("Best approximation so far:")
        print(f"x = {x_new}")
        return x_new

    # Direct Method fallback
    print("\nSolving using Direct Method:")
    try:
        x_direct = np.linalg.solve(A, b)
        print("Solution x (Direct Method):")
        print(x_direct)
    except np.linalg.LinAlgError:
        print("Error: Direct solution failed due to a singular or ill-conditioned matrix.")


# Run the solver
if __name__ == "__main__":
    solve_linear_system()
