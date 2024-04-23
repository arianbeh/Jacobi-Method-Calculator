import numpy as np
import testing
import time

def lu_factorization(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    operations = 0

    for i in range(n):
        # Upper Triangular Matrix
        for k in range(i, n):
            sum_ = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_
            operations += 1

        # Lower Triangular Matrix
        for k in range(i, n):
            if i == k:
                L[i][i] = 1.0
            else:
                sum_ = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - sum_) / U[i][i]
            operations += 1

    return L, U, operations

def count_nonzero_entries(matrix):
    count = 0
    for row in matrix:
        for element in row:
            if element != 0:
                count += 1
    return count

# Main program
for N in range(1, 8):
    # Generate ∆N matrix
    delta_N = [[0] * (N*N) for _ in range(N*N)]
    for i in range(N*N):
        for j in range(N*N):
            p, q = divmod(i, N)
            r, s = divmod(j, N)
            if abs(p - r) + abs(q - s) == 1:
                delta_N[i][j] = 1
            if i == j:
                delta_N[i][j] = -4
    print(np.matrix(delta_N))

    # Perform LU-factorization
    L, U, operations = lu_factorization(delta_N)

    # Count non-zero entries in matrices L and U
    nonzero_L = count_nonzero_entries(L)
    nonzero_U = count_nonzero_entries(U)
    b = np.random.randint(-10, 10, size=(N,))
    epsilon = 1 * 10**-14

    print("A:", delta_N)
    print("b:", b)
    print("size of A: ", N * N)
    start_time = time.time()
    expected = np.linalg.solve(delta_N, b)
    end_time = time.time()
    print("Elapsed time for numpy: ", end_time - start_time)
    start_time = time.time()
    actual, iterations = Onk.jacobi_method(delta_N, b, epsilon)
    end_time = time.time()
    print("Elapsed time for Jacobi: ", end_time - start_time)

    # Print results
    print(f"For N = {N}:")
    print("Number  of Operations: ", operations)
    print("Number of operations in terms of N: ", operations/N, "N")
    print(f"Non-zero entries in L: {nonzero_L}")
    print(f"Non-zero entries in U: {nonzero_U}")
