import numpy as np
from sympy import symbols, Matrix
from tabulate import tabulate

def get_input(n, mode="manual"):
    """Get user input or generate random coefficients for a tridiagonal system."""
    if mode == "manual":
        print("\nEnter the coefficients for the tridiagonal matrix:")
        a = np.array([float(input(f"  a[{i+1}] (below main diagonal in row {i+2}): ")) for i in range(n-1)])
        b = np.array([float(input(f"  b[{i+1}] (main diagonal in row {i+1}): ")) for i in range(n)])
        c = np.array([float(input(f"  c[{i+1}] (above main diagonal in row {i+1}): ")) for i in range(n-1)])
        d = np.array([float(input(f"  d[{i+1}] (right-hand side for row {i+1}): ")) for i in range(n)])
    else:
        a, c = np.random.randint(1, 10, n-1), np.random.randint(1, 10, n-1)
        b, d = np.random.randint(10, 20, n), np.random.randint(10, 50, n)
        print("Generated random coefficients successfully!")
    
    return a, b, c, d

def display_matrix(n, a, b, c, d):
    """Display the system as a tridiagonal matrix."""
    matrix = [[b[i] if i == j else (a[j] if i == j+1 else (c[i] if i+1 == j else 0)) for j in range(n)] for i in range(n)]
    table = [row + [d[i]] for i, row in enumerate(matrix)]
    headers = ["a", "b", "c"][:n] + ["d"]
    print("\nTridiagonal System Representation:")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

def symbolic_representation(n, a, b, c, d):
    """Show symbolic representation of the equations."""
    x = symbols(f'x1:{n+1}')
    equations = [f"{a[i-1]}*{x[i-1]} + {b[i]}*{x[i]} + {c[i]}*{x[i+1]} = {d[i]}" if 0 < i < n-1 else 
                 (f"{b[i]}*{x[i]} + {c[i]}*{x[i+1]} = {d[i]}" if i == 0 else f"{a[i-1]}*{x[i-1]} + {b[i]}*{x[i]} = {d[i]}") 
                 for i in range(n)]
    print("\nSystem of Equations:\n" + "\n".join([f"  {eq}" for eq in equations]))

def gauss_thomas(n, a, b, c, d):
    """Solve the tridiagonal system using the Gauss-Thomas method."""
    # Step 1: Forward Elimination
    c_star, d_star = np.zeros(n-1), np.zeros(n)
    c_star[0], d_star[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n-1):
        denominator = b[i] - a[i-1] * c_star[i-1]
        c_star[i] = c[i] / denominator
        d_star[i] = (d[i] - a[i-1] * d_star[i-1]) / denominator
    d_star[n-1] = (d[n-1] - a[n-2] * d_star[n-2]) / (b[n-1] - a[n-2] * c_star[n-2])
    print("\nForward Elimination Steps:")
    for i in range(n-1):
        print(f"  c*[{i+1}] = {c_star[i]:.6f},  d*[{i+1}] = {d_star[i]:.6f}")
    print(f"  d*[{n}] = {d_star[n-1]:.6f}")

    # Step 2: Backward Substitution
    x = np.zeros(n)
    x[n-1] = d_star[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i+1]
    print("\nBackward Substitution Steps:")
    for i in range(n-1, -1, -1):
        print(f"  x[{i+1}] = {x[i]:.6f}")
    return x

if __name__ == "__main__":
    # print("Solving a tridiagonal system using the Gauss-Thomas (Thomas) method.\n")
    n = int(input("Enter the number of equations (n): "))
    choice = input("\nChoose input method (1: Manual, 2: Random): ")
    a, b, c, d = get_input(n, mode="manual" if choice == "1" else "random")
    display_matrix(n, a, b, c, d)
    symbolic_representation(n, a, b, c, d)
    solution = gauss_thomas(n, a, b, c, d)
    print("\nFinal Solution:")
    print(tabulate([[f"x[{i+1}]", f"{solution[i]:.6f}"] for i in range(n)], tablefmt="fancy_grid"))
