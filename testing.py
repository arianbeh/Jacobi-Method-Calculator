import numpy as np
import project 

def generate_invertible_array(n):
    while True:
        # Create a random square matrix of size n x n
        matrix = np.random.rand(n, n)
        
        # Check if the matrix is invertible by ensuring its determinant is non-zero
        if np.linalg.det(matrix) != 0:
            return matrix
n = 3
A = generate_invertible_array(3)
b = np.random.rand(n)
expected = np.linalg.solve(A, b)
actual  = project.Jacobi(A, n, b, 1)
print("Actual:", actual)
print("Expected:", expected)

