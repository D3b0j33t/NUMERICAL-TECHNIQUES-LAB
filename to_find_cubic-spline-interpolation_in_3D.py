import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cubic_spline_3d(x, y, z, query_points_x, query_points_y):
    X, Y = np.meshgrid(query_points_x, query_points_y)
    def cubic_spline_1d(x_vals, y_vals, query_points):
        n = len(x_vals) - 1
        h = np.diff(x_vals)
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)
        A[0, 0] = 1
        A[n, n] = 1
        for i in range(1, n):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            b[i] = 3 * ((y_vals[i + 1] - y_vals[i]) / h[i] - (y_vals[i] - y_vals[i - 1]) / h[i - 1])
        M = np.linalg.solve(A, b)
        def evaluate_spline(xi):
            for i in range(n):
                if x_vals[i] <= xi <= x_vals[i + 1]:
                    a = y_vals[i]
                    b = (y_vals[i + 1] - y_vals[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 3
                    c = M[i] / 2
                    d = (M[i + 1] - M[i]) / (3 * h[i])
                    dx = xi - x_vals[i]
                    return a + b * dx + c * dx**2 + d * dx**3
            raise ValueError("Query point out of bounds.")

        return np.array([evaluate_spline(xi) for xi in query_points])
    interpolated_z = np.array([cubic_spline_1d(x, z_row, query_points_x) for z_row in z])
    interpolated_z_final = np.array([cubic_spline_1d(y, interpolated_z[:, i], query_points_y) for i in range(len(query_points_x))])
    return X, Y, interpolated_z_final
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
                z = []
                print("Enter the x, y and z coordinates:")
                for i in range(n):
                    xi = float(input(f"x[{i}]: "))
                    yi = float(input(f"y[{i}]: "))
                    zi = float(input(f"z[{i}]: "))
                    x.append(xi)
                    y.append(yi)
                    z.append(zi)
                x = np.array(x)
                y = np.array(y)
                z = np.array(z)
            elif choice == 2:
                n = int(input("Enter the number of data points (>= 3): "))
                if n < 3:
                    print("You must have at least 3 data points for spline interpolation!")
                    continue
                x = np.sort(np.random.uniform(0, 10, n))  # Random x in range [0, 10]
                y = np.sort(np.random.uniform(0, 10, n))  # Random y in range [0, 10]
                z = np.random.uniform(-10, 10, (n, n))    # Random 2D z values in range [-10, 10]
                print("\nGenerated random data points:")
                for i in range(n):
                    for j in range(n):
                        print(f"x[{i}] = {x[i]:.2f}, y[{j}] = {y[j]:.2f}, z[{i},{j}] = {z[i,j]:.2f}")
            else:
                print("Invalid choice! Please enter 1 or 2.")
                continue
            return x, y, z
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")
def main():
    while True:
        print("\n=== 3D Cubic Spline Interpolation ===")
        x, y, z = get_input()
        print("\nEnter the number of query points for interpolation (leave empty for default): ")
        user_input = input()
        if user_input.strip() == "":
            num_query_points = max(10, len(x) * 3)
            print(f"Using default number of query points: {num_query_points}")
        else:
            try:
                num_query_points = int(user_input)
                if num_query_points <= 0:
                    print("Number of query points must be greater than 0. Using default value.")
                    num_query_points = max(10, len(x) * 3)
            except ValueError:
                print("Invalid input! Using default number of query points.")
                num_query_points = max(10, len(x) * 3)

        query_points_x = np.linspace(min(x), max(x), num_query_points)
        query_points_y = np.linspace(min(y), max(y), num_query_points)
        X, Y, interpolated_z = cubic_spline_3d(x, y, z, query_points_x, query_points_y)
        print("\nInterpolation complete. Plotting 3D surface...")
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, interpolated_z, cmap='viridis', edgecolor='none')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Cubic Spline Interpolation')
        plt.show()
        repeat = input("\nDo you want to run the program again? (yes/no): ").strip().lower()
        if repeat not in ('yes', 'y'):
            print("Goodbye!")
            break
if __name__ == "__main__":
    main()
