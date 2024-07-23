# Parallel-Algorithms-For-Matrix-Multiplication.
# Parallel Matrix Multiplication using CUDA

This project implements parallel matrix multiplication using CUDA in Python. It leverages the power of GPU computing to significantly speed up the computation compared to traditional CPU-based methods. The project includes the CUDA kernel for matrix multiplication, a function to launch the kernel, performance analysis, and a comparison with CPU-based multiplication.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Analysis](#performance-analysis)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Matrix multiplication is a fundamental operation in many scientific and engineering applications. Traditional sequential algorithms are often slow and inefficient for large matrices. This project demonstrates the use of parallel algorithms to improve performance by distributing the computation across multiple processing units using CUDA.

## Installation

### Prerequisites

- Python 3.6 or later
- Numba
- CUDA Toolkit (appropriate version for your GPU)

### Install Required Libraries

```bash
pip install numpy numba
```

## Usage

### Running the Code

1. Clone the repository:

```bash
git clone https://github.com/yourusername/parallel-matrix-multiplication-cuda.git
cd parallel-matrix-multiplication-cuda
```

2. Run the matrix multiplication example:

```bash
python main.py
```

### Code Structure

- `main.py`: Contains the main function to perform matrix multiplication and measure performance.
- `cuda_kernel.py`: Contains the CUDA kernel for matrix multiplication.

### Example Code

Here's how to perform matrix multiplication using the provided functions:

```python
import numpy as np
from cuda_kernel import parallel_matrix_multiply_gpu

if __name__ == "__main__":
    A = np.random.rand(1024, 1024).astype(np.float32)
    B = np.random.rand(1024, 1024).astype(np.float32)
    
    C = parallel_matrix_multiply_gpu(A, B)
    print(C)
```

## Performance Analysis

To compare the performance of GPU-based and CPU-based matrix multiplication, you can use the provided functions:

```python
import time
import numpy as np
from cuda_kernel import parallel_matrix_multiply_gpu

def cpu_matrix_multiply(A, B):
    return np.dot(A, B)

if __name__ == "__main__":
    A = np.random.rand(1024, 1024).astype(np.float32)
    B = np.random.rand(1024, 1024).astype(np.float32)
    
    start_time = time.time()
    C_cpu = cpu_matrix_multiply(A, B)
    cpu_time = time.time() - start_time
    print(f"CPU Time: {cpu_time} seconds")
    
    start_time = time.time()
    C_gpu = parallel_matrix_multiply_gpu(A, B)
    gpu_time = time.time() - start_time
    print(f"GPU Time: {gpu_time} seconds")
    
    assert np.allclose(C_cpu, C_gpu), "Results do not match!"
    print("Results match!")
```

### Results

The GPU implementation significantly outperforms the CPU implementation, achieving faster computation times for large matrices. Detailed performance results and comparisons can be found in the `report.pdf` file.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to modify and expand this README as needed to fit your project's specific requirements and to provide additional details as necessary.
