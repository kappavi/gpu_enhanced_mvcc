# GPU-Enhanced MVCC (Multi-Version Concurrency Control)

This project implements a Multi-Version Concurrency Control (MVCC) system that uses GPU acceleration for certain operations to improve performance.

## Overview

MVCC is a concurrency control method commonly used in database systems. It maintains multiple versions of data to allow concurrent read and write operations without blocking. This implementation explores the performance benefits of offloading batch operations to the GPU.

Key features:
- CPU-based MVCC implementation for traditional operations
- GPU-accelerated batch operations for improved performance
- Support for testing with random data and CSV datasets
- Performance benchmarking between CPU and GPU implementations

## Requirements
- GT's PACE cluster is really good for all of the following:
- CUDA Toolkit
- C++ compiler with C++11 support
- Python 3 (for generating test data)

## How to Run

1. Compile the MVCC implementation:
   ```bash
   nvcc -o mvcc mvcc.cu
   ```

2. Run the program:
   ```bash
   ./mvcc
   ```

## Tests

The program runs several tests:
1. Basic functionality test
2. Performance benchmarks with varying dataset sizes, which are auto generated data.
   - Small (1,000 operations)
   - Medium (10,000 operations)
   - Large (100,000 operations)
3. CSV file tests, which are based on hardcoded CSV data.
   - Small CSV dataset (test_small.csv)
   - Larger CSV dataset (test_large.csv)
   - Very large CSV dataset (test_very_large.csv with 100,000 operations)

For small datasets, the CPU implementation may be faster due to GPU initialization overhead. As dataset size increases, the GPU implementation should demonstrate better performance.

## Project Structure

- `mvcc.cu`: Main implementation file with MVCC store and benchmarking code
- `test_small.csv`: Small dataset for testing
- `test_large.csv`: Medium-sized dataset for testing
- `test_very_large.csv`: Large dataset (100,000 operations) for testing
- `generate_large_csv.py`: Python script to generate the very large CSV file 
