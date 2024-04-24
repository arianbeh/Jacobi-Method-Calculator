import numpy as np
import time

def jacobi_method(A, b, epsilon):
    n = A.shape[0] 
    entries, jump = sparse_matrix(A, n)
    print("Size of new A:", len(entries) * 2)
    x = np.zeros(n, dtype=np.float64)
    iteration_count = 0 
    operations = 0
    while True:
        x_new = np.zeros(n, dtype=np.float64)
        x_count = 0
        i = 0
        while i < len(entries):
            sum = 0
            while jump[i] != -1:
                sum += -(entries[i] * x[jump[i]])
                i += 1
                operations += 1
            x_new[x_count] = (sum + b[x_count]) / entries[i]
            operations += 1
            x_count += 1
            i += 1
        # print("Current iteration solved: ", x_new)
        result = x_new - x
        operations += 1
        norm = np.float64(0)
        for i in result:
            norm += i ** 2
            operations += 1
        if norm < epsilon:
            return x_new, iteration_count, operations
        x = x_new
        iteration_count += 1
def sparse_matrix(A, n):
    entries = []
    directions = []
    for i in range(n):
        for j in range(n):
            if i != j:
                if A[i, j] != 0:
                    entries.append(A[i, j])
                    directions.append(j)
        entries.append(A[i, i])
        directions.append(-1)
    return entries, directions

def max_nonzeroes_in_row(A):
    max = 0
    for row in A:
        new_max = 0
        for j in row:
            if j != 0:
                new_max += 1
        if new_max > max:
            max = new_max
    return max
def generate_diag_dom(n):
    sparsity = 0.7
    while True:
        # Generate a dense matrix
        dense_matrix = np.random.randint(0, 10, size=(n, n))       
        # Create a mask for sparse entries
        mask = np.random.choice([0, 1], size=(n, n), p=[sparsity, 1 - sparsity])       
        # Apply the mask to the dense matrix
        sparse_matrix = dense_matrix * mask       
        # Check if the condition number is less than the threshold
        for i in range(len(sparse_matrix)):
            sum = 0
            for j in range(len(sparse_matrix[i])):
                if i != j:
                    sum += abs(sparse_matrix[i, j])
            sparse_matrix[i, i] += sum
        if (np.linalg.det(sparse_matrix) != 0):
            return sparse_matrix

# n = 160
# A = generate_diag_dom(n)
# b = np.random.randint(-10, 10, size=(n,))
# epsilon = 1 * 10**-14

# print("A:", A)
# print("b:", b)
# print("Target size: ", max_nonzeroes_in_row(A) * n)
# print("size of A: ", n * n)
# start_time = time.time()
# expected = np.linalg.solve(A, b)
# end_time = time.time()
# print("Elapsed time for numpy: ", end_time - start_time)
# start_time = time.time()
# actual, iterations = jacobi_method(A, b, epsilon)
# end_time = time.time()
# print("Elapsed time for Jacobi: ", end_time - start_time)

# print("Iterations: ", iterations)
# print("Actual:", actual)
# print("Expected:", expected)