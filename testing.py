import numpy as np
import sys

def get_D(A):
    n = A.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = A[i, i]
    return D

def get_A0(A, D):
    n = A.shape[0]
    A0 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A0[i, j] = A[i, j] if i != j else 0
    return A0

def get_D_inverse(D):
    n = D.shape[0]
    D_inverse = np.zeros((n, n))
    for i in range(n):
        if D[i, i] != 0:
            D_inverse[i, i] = 1 / D[i, i]
        else:
            D_inverse[i, i] = float('inf')  # Avoid division by zero
    return D_inverse

def multiply(A, x):
    n = len(x)
    result = np.zeros(n)
    for i in range(n):
        for j in range(n):
            result[i] += A[i, j] * x[j]
    return result

def max_entry(v):
    return np.max(np.abs(v))

def Jacobi(A, b, epsilon, max_iterations=1000):
    n = A.shape[0]
    x = np.zeros(n)
    D = get_D(A)
    A0 = get_A0(A, D)
    D_inv = get_D_inverse(D)
    relaxation_factor = 0.5  # Often set between 0.5 and 1

    for iteration in range(max_iterations):
        r = b - multiply(A0, x)
        delta_x = multiply(D_inv, r)
        x_new = x + relaxation_factor * delta_x  # Applying relaxation factor
        if max_entry(x_new - x) < epsilon:
            return x_new
        x = x_new

    return x  # Return last computed x if no convergence within max_iterations

# Test the Jacobi method
def generate_invertible_array(n):
    while True:
        matrix = np.random.rand(n, n)
        if np.linalg.cond(matrix) < 1/sys.float_info.epsilon:  # Condition number check
            return matrix

n = 3
A = generate_invertible_array(n)
b = np.random.rand(n)
epsilon = 0.01

expected = np.linalg.solve(A, b)
actual = Jacobi(A, b, epsilon)

print("A:", A)
print("b:", b)
print("Actual:", actual)
print("Expected:", expected)
