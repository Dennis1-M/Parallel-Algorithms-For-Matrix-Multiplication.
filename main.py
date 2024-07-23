import numpy as np
import time
from cuda_kernel import parallel_matrix_multiply_gpu

def cpu_matrix_multiply(A, B):
    return np.dot(A, B)

if __name__=='__main__':
    # Create two random matrices
    A = np.random.rand(1024, 1024).astype(np.float32)
    B = np.random.rand(1024, 1024).astype(np.float32)
    
    # Measure time for CPU multiplication
    start_time = time.time()
    C_cpu = cpu_matrix_multiply(A, B)
    cpu_time = time.time() - start_time
    print(f"CPU Time: {cpu_time} seconds")
    
    # Measure time for GPU multiplication
    start_time = time.time()
    C_gpu = parallel_matrix_multiply_gpu(A, B)
    gpu_time = time.time() - start_time
    print(f"GPU Time: {gpu_time} seconds")
    
    # Verify the result
    assert np.allclose(C_cpu, C_gpu), "Results do not match!"
    print("Results match!")