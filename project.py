import numpy as np

# def delta_N(N):
#     # Generate the matrix ΔN for the 2D Poisson problem
#     size = N**2
#     ΔN = np.zeros((size, size))
#     for i in range(size):
#         ΔN[i, i] = -4
#         if i % N != 0:     # not on the left edge
#             ΔN[i, i-1] = 1
#         if (i + 1) % N != 0: # not on the right edge
#             ΔN[i, i+1] = 1
#         if i >= N:         # not on the top edge
#             ΔN[i, i-N] = 1
#         if i < size - N:   # not on the bottom edge
#             ΔN[i, i+N] = 1
#     return ΔN

# def jacobi_method(A, b, x0, tol, max_iter=10000):
#     D = np.diag(np.diag(A))
#     D_inv = np.linalg.inv(np.diag(np.diag(A)))
#     R = A - np.diag(np.diag(A))
    
#     x = x0
#     for _ in range(max_iter):
#         x_new = D_inv @ (b - R @ x)
#         if np.max(np.abs(A @ x_new - b)) < tol:
#             break
#         x = x_new
#     return x

# # Example usage for Δ2
# N = 2
# Δ2 = delta_N(N)
# b = np.array([1, 0, 0, 0])  # Source function example
# x0 = np.zeros(N*N)
# result = jacobi_method(Δ2, b, x0, 1e-12)
# print("Result of Jacobi Method for N=2:", result)



def get_D(A, n):
    D = np.zeros(n, n)
    for i in range (n):
        D[i, i] = A[i, i]
    return D

def get_A0(A, D, n):
    A0 = np.zeros(n, n)
    for i in range (n):
        A0[:,i] = A[:,i] - D[:, i] #get the i-th column 
    return A0

def get_D_inverse(D, n):
    D_inverse = np.zeros(n, n)
    for i in range(n):
        D_inverse[i, i] = 1 / D[i, i]
    return D_inverse

def max_entry(v, n):
    max = 0
    for i in range (n):
        val = abs(v[i])
        if val > max:
            max = val
    return max

def multiply(A, x, n):
    result = np.zeros(n)
    for i in range (n):
        for j in range (n):
            result[i] += A[i, j] * n[j]
    return result 

# e is epsilon
def Jacobi(A, n, b, e):
    x = np.zeros(n)
    v = multiply(A, x)
    D = get_D(A, n)
    A0 = get_A0(A, D, n)
    D_inverse = get_D_inverse(D, n)
    while(max(v - b) >= e):
        temp = b - multiply(A0, x)
        x = multiply(D_inverse, temp)
        v = multiply(A, x)
    print(x)
        
