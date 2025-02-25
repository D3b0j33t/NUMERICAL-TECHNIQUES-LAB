import numpy as np

def print_matrix(matrix, step=""):
    """Helper function to print the matrix in a more readable format."""
    print(f"{step}")
    for row in matrix:
        print("  ".join(f"{elem:8.2f}" for elem in row))
    print("\n" + "-"*50)

def gaussian_elimination(A, B):
    n = len(B)
    augmented_matrix = np.hstack((A, B.reshape(-1, 1)))
    
    # Initial augmented matrix
    print_matrix(augmented_matrix, "Initial Augmented Matrix:")
    
    for i in range(n):
        # Pivoting (row swapping)
        max_row = np.argmax(np.abs(augmented_matrix[i:n, i])) + i
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]
        
        print_matrix(augmented_matrix, f"After swapping row {i+1} with row {max_row+1}:")
        
        # Normalize the pivot row
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]
        
        print_matrix(augmented_matrix, f"After normalizing row {i+1}:")
        
        # Eliminate the entries below the pivot
        for j in range(i + 1, n):
            augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j, i]
        
        print_matrix(augmented_matrix, f"After eliminating column {i+1}:")
    
    # Back substitution to solve for the variables
    solution = np.zeros(n)
    for i in range(n-1, -1, -1):
        solution[i] = augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i+1:n], solution[i+1:]) if i < n-1 else augmented_matrix[i, -1]
    
    return solution

A = np.array([[2, 1, 1], [3, 2, 3], [1, 4, 9]], dtype=float)
B = np.array([10, 18, 16], dtype=float)

solution = gaussian_elimination(A, B)

# Final Solution Output
print("\nFinal Solution:")
print(f"x = {solution[0]:.3f}")
print(f"y = {solution[1]:.3f}")
print(f"z = {solution[2]:.3f}")