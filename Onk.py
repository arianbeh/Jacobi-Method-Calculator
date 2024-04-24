import numpy as np
import time

def jacobi_method(A, b, epsilon):
    n = A.shape[0] 
    entries, jump = sparse_matrix(A, n)
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
        sparse_matrix = dense_matrix * mask       
        for i in range(len(sparse_matrix)):
            sum = 0
            for j in range(len(sparse_matrix[i])):
                if i != j:
                    sum += abs(sparse_matrix[i, j])
            sparse_matrix[i, i] += sum
        if (np.linalg.det(sparse_matrix) != 0):
            return sparse_matrix

