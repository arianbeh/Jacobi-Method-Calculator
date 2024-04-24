import numpy as np
import Onk

def lu_factorization(A):
    n = A.shape[0]
    U = A.copy()
    U = U.astype(float)
    L = np.zeros((n, n))
    N = int(np.sqrt(n))
    # Create identity
    for i in range(n):
        L[i, i] = 1
    operations = 0
    for i in range(n - 1):
        for j in range(i + 1, i + N + 1):
            if j < n:
                factor = U[j, i] / U[i, i]
                L[j, i] = factor
                U[j] -= factor * U[i] 
                operations += 1       
    return L, U, operations


def count_nonzero_entries(matrix):
    count = 0
    for row in matrix:
        for element in row:
            if element != 0:
                count += 1
    return count
def create_b(N):
    b = np.zeros((N * N))
    N = np.floor((N * N) / 2).astype(int)
    b[N] = 1
    return b

# Main program
N_list = []
iterations_list = []
for N in range(1, 16):
    # Generate ∆N matrix
    delta_N = np.zeros((N*N, N*N), dtype=float)
    for i in range(N*N):
        for j in range(N*N):
            p, q = divmod(i, N)
            r, s = divmod(j, N)
            if abs(p - r) + abs(q - s) == 1:
                delta_N[i][j] = 1
            if i == j:
                delta_N[i][j] = -4

    # Perform LU-factorization
    A_1 = np.copy(delta_N).astype(np.float64)
    L, U, operations = lu_factorization(A_1)
    result = L @ U
    result = np.round(result).astype(int)
    print("LU worked? ", np.array_equal(result, delta_N))
    print("LU = ")
    print(result)

    # Count non-zero entries in matrices L and U
    nonzero_L = count_nonzero_entries(L)
    nonzero_U = count_nonzero_entries(U)
    # b = np.random.randint(-10, 10, size=(N * N,))
    b = create_b(N)
    epsilon = 10**-12

    actual, iterations, operations = Onk.jacobi_method(delta_N, b, epsilon)
    N_list.append(N)
    iterations_list.append(iterations)

