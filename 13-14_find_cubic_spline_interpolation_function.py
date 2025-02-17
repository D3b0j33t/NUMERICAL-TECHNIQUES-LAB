import numpy as np
import matplotlib.pyplot as plt

def cubic_spline(x, y, query_points):
    n = len(x) - 1  # Number of intervals
    h = np.diff(x)  # Interval widths

    # Step 1: Solve for second derivatives (M)
    A = np.zeros((n + 1, n + 1))  # Coefficient matrix
    b = np.zeros(n + 1)           # Right-hand side vector

    # Natural spline boundary conditions
    A[0, 0] = 1
    A[n, n] = 1

    # Fill the tridiagonal matrix for interior points
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Solve for second derivatives (M)
    M = np.linalg.solve(A, b)

    # Step 2: Compute spline coefficients for each interval
    splines = []
    for i in range(n):
        a = y[i]
        b = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 3
        c = M[i] / 2
        d = (M[i + 1] - M[i]) / (3 * h[i])
        splines.append((a, b, c, d))

    # Step 3: Evaluate the spline at query points
    def evaluate_spline(xi):
        for i in range(n):
            if x[i] <= xi <= x[i + 1]:
                a, b, c, d = splines[i]
                dx = xi - x[i]
                return a + b * dx + c * dx**2 + d * dx**3
        raise ValueError("Query point out of bounds.")

    interpolated_values = np.array([evaluate_spline(xi) for xi in query_points])

    return splines, interpolated_values

def get_input():
    while True:
        try:
            print("\nChoose how to provide input:")
            print("1. Enter data points manually")
            print("2. Use randomized data points (optimal range)")
            choice = int(input("Enter your choice (1 or 2): "))

            if choice == 1:
                n = int(input("Enter the number of data points (>= 3): "))
                if n < 3:
                    print("You must have at least 3 data points for spline interpolation!")
                    continue

                x = []
                y = []
                print("Enter the x and y coordinates:")
                for i in range(n):
                    xi = float(input(f"x[{i}]: "))
                    yi = float(input(f"y[{i}]: "))
                    x.append(xi)
                    y.append(yi)
                x = np.array(x)
                y = np.array(y)

            elif choice == 2:
                n = int(input("Enter the number of data points (>= 3): "))
                if n < 3:
                    print("You must have at least 3 data points for spline interpolation!")
                    continue

                x = np.sort(np.random.uniform(0, 10, n))  # Random x in range [0, 10]
                y = np.random.uniform(-10, 10, n)         # Random y in range [-10, 10]
                print("\nGenerated random data points:")
                for i in range(n):
                    print(f"x[{i}] = {x[i]:.2f}, y[{i}] = {y[i]:.2f}")

            else:
                print("Invalid choice! Please enter 1 or 2.")
                continue

            return x, y

        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

def main():
    while True:
        print("\n=== Cubic Spline Interpolation ===")
        x, y = get_input()

        # Points to interpolate
        print("\nEnter the number of query points for interpolation (leave empty for default): ")
        user_input = input()
        if user_input.strip() == "":
            # Default to an optimal number of query points
            num_query_points = max(100, len(x) * 10)
            print(f"Using default number of query points: {num_query_points}")
        else:
            try:
                num_query_points = int(user_input)
                if num_query_points <= 0:
                    print("Number of query points must be greater than 0. Using default value.")
                    num_query_points = max(100, len(x) * 10)
            except ValueError:
                print("Invalid input! Using default number of query points.")
                num_query_points = max(100, len(x) * 10)

        query_points = np.linspace(min(x), max(x), num_query_points)

        # Compute the cubic spline and evaluate it
        splines, interpolated_values = cubic_spline(x, y, query_points)

        # Display results
        print("\nCubic Spline Coefficients (for each interval):")
        for i, (a, b, c, d) in enumerate(splines):
            print(f"Interval [{x[i]:.2f}, {x[i+1]:.2f}]: a = {a:.4f}, b = {b:.4f}, c = {c:.4f}, d = {d:.4f}")

        # Plot the results
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'o', label="Data Points", markersize=6, color='red')
        plt.plot(query_points, interpolated_values, '-', label="Cubic Spline", color='blue')
        plt.title("Cubic Spline Interpolation", fontsize=14)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.show()

        # Option to run again or exit
        repeat = input("\nDo you want to run the program again? (yes/no): ").strip().lower()
        if repeat not in ('yes', 'y'):
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()
