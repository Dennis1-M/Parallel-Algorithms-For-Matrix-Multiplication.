import numpy as np
from numba import cuda
import math

@cuda.jit
def gpu_matrix_multiply(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

def parallel_matrix_multiply_gpu(A, B):
    # Transfer the data to the GPU
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    C_global_mem = cuda.device_array((A.shape[0], B.shape[1]))
    
    # Define the size of the grid and blocks
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(A.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(B.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Launch the kernel
    gpu_matrix_multiply[blocks_per_grid, threads_per_block](A_global_mem, B_global_mem, C_global_mem)
    
    # Copy the result back to the host
    C = C_global_mem.copy_to_host()

    return C