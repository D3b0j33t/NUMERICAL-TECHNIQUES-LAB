import numpy as np

def gauss_thomas():
    print("Gauss-Thomas Method for Solving Tridiagonal Systems\n")
    
    # Step 1: Input the size of the system
    n = int(input("Enter the number of equations (n): "))

    # Step 2: Choose input method
    print("\nChoose an input method:")
    print("1. Manually input the matrix coefficients and RHS vector")
    print("2. Generate random coefficients and RHS vector")
    choice = int(input("Enter your choice (1 or 2): "))

    # Initialize the coefficients
    a = np.zeros(n - 1)  # Sub-diagonal elements
    b = np.zeros(n)      # Main diagonal elements
    c = np.zeros(n - 1)  # Super-diagonal elements
    d = np.zeros(n)      # Right-hand side elements

    # Handle input method choice using ternary operations
    if choice == 1:
        # Manual Input
        print("\nEnter the coefficients of the tridiagonal matrix:")
        
        # Sub-diagonal input
        for i in range(n - 1):
            a[i] = float(input(f"Enter sub-diagonal element a[{i + 1}] (below main diagonal): "))
        
        # Main diagonal input
        for i in range(n):
            b[i] = float(input(f"Enter main diagonal element b[{i + 1}] (on main diagonal): "))
        
        # Super-diagonal input
        for i in range(n - 1):
            c[i] = float(input(f"Enter super-diagonal element c[{i + 1}] (above main diagonal): "))
        
        # Right-hand side input
        for i in range(n):
            d[i] = float(input(f"Enter right-hand side element d[{i + 1}]: "))

    elif choice == 2:
        # Randomized Input
        print("\nGenerating random coefficients for the tridiagonal matrix...")
        
        # Randomize sub-diagonal
        a = np.random.randint(1, 10, size=n - 1).astype(float)
        
        # Randomize main diagonal
        b = np.random.randint(10, 20, size=n).astype(float)
        
        # Randomize super-diagonal
        c = np.random.randint(1, 10, size=n - 1).astype(float)
        
        # Randomize right-hand side
        d = np.random.randint(10, 50, size=n).astype(float)
        
        print("\nRandomized matrix generated successfully!")

    else:
        print("Invalid choice! Exiting program.")
        return

    # Display the matrix visually
    print("\nTridiagonal Matrix Representation:")
    for i in range(n):
        row = []
        for j in range(n):
            row.append(b[i] if i == j else (a[j] if i == j + 1 else (c[i] if i + 1 == j else 0)))
        print(row, " = ", d[i])

    # Step 3: Forward elimination
    c_star = np.zeros(n - 1)
    d_star = np.zeros(n)

    # Initial calculations for the first row
    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]

    for i in range(1, n):
        temp = b[i] - a[i - 1] * c_star[i - 1]
        c_star[i] = c[i] / temp if i < n - 1 else 0
        d_star[i] = (d[i] - a[i - 1] * d_star[i - 1]) / temp

    # Step 4: Back substitution
    x = np.zeros(n)
    x[-1] = d_star[-1]

    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]

    # Step 5: Output the solution
    print("\nSolution:")
    for i in range(n):
        print(f"x{i + 1} = {x[i]:.6f}")

    return x

# Call the function
if __name__ == "__main__":
    print("Solving a tridiagonal system using the Gauss-Thomas method.")
    solution = gauss_thomas()
