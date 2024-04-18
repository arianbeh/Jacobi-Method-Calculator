import numpy as np

def jacobi_method(A, b, epsilon):
    n = A.shape[0]  # Assuming A is a square matrix
    D_inv = np.zeros((n, n))
    A0 = np.zeros((n, n))
    
    # Constructing D_inv and A0 explicitly
    for i in range(n):
        D_inv[i][i] = 1.0 / A[i][i]  # Inverse of diagonal elements of A
        for j in range(n):
            if i != j:
                A0[i][j] = A[i][j]  # A0 with zeros on diagonal

    x = np.zeros(n)  # Initial guess for x is a zero vector
    iteration_count = 0  # Track iteration count for convergence
    
    while True:
        # Calculate the next iteration of x
        x_new = np.zeros(n)
        for i in range(n):
            # Manually compute the product A0*x for the ith row
            sum_A0x = 0
            for j in range(n):
                sum_A0x += A0[i][j] * x[j]
            x_new[i] = D_inv[i][i] * (b[i] - sum_A0x)  # Jacobi update formula
        
        # Manually compute the residual r = Ax - b
        r = np.zeros(n)
        for i in range(n):
            # Compute the matrix-vector product Ax for the ith element
            Ax_i = 0
            for j in range(n):
                Ax_i += A[i][j] * x_new[j]
            r[i] = Ax_i - b[i]
        
        if np.max(np.abs(r)) < epsilon:
            break  # Convergence achieved if the maximum error is below epsilon
        x = x_new
        iteration_count += 1

    return x, iteration_count

# Example usage
A = np.array([[4, 1, 2],
              [1, 5, 1],
              [2, 1, 3]], dtype=float)

b = np.array([1, 2, 3], dtype=float)
epsilon = 1e-5

x, iterations = jacobi_method(A, b, epsilon)
print("Solution x:", x)
print("Iterations:", iterations)
