# PyTorch-101-Tensor-Operations-and-Utilities
A collection of utility functions and exercises for basic tensor operations in PyTorch, designed to help beginners get started with PyTorch.

## Table of Contents

- [Getting Started](#getting-started)
- [Functions](#functions)
  - [hello](#hello)
  - [create_sample_tensor](#create_sample_tensor)
  - [mutate_tensor](#mutate_tensor)
  - [count_tensor_elements](#count_tensor_elements)
  - [create_tensor_of_pi](#create_tensor_of_pi)
  - [multiples_of_ten](#multiples_of_ten)
  - [slice_indexing_practice](#slice_indexing_practice)
  - [slice_assignment_practice](#slice_assignment_practice)
  - [shuffle_cols](#shuffle_cols)
  - [reverse_rows](#reverse_rows)
  - [take_one_elem_per_col](#take_one_elem_per_col)
  - [make_one_hot](#make_one_hot)
  - [sum_positive_entries](#sum_positive_entries)
  - [reshape_practice](#reshape_practice)
  - [zero_row_min](#zero_row_min)
  - [batched_matrix_multiply](#batched_matrix_multiply)
  - [normalize_columns](#normalize_columns)
  - [mm_on_cpu](#mm_on_cpu)
  - [mm_on_gpu](#mm_on_gpu)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

To get started with this project, clone the repository and install the necessary dependencies. This project requires PyTorch, which you can install via pip:

```sh
pip install torch
```

You can then import and use the functions as needed in your own projects.

## Functions

### hello

```python
def hello():
```

Prints a simple greeting message to verify that the environment is correctly set up.

### create_sample_tensor

```python
def create_sample_tensor() -> Tensor:
```

Returns a torch Tensor of shape (3, 2) filled with random values.

### mutate_tensor

```python
def mutate_tensor(x: Tensor, indices: List[Tuple[int, int]], values: List[float]) -> Tensor:
```

Mutates the tensor `x` at specified `indices` with the provided `values`.

### count_tensor_elements

```python
def count_tensor_elements(x: Tensor) -> int:
```

Counts the number of scalar elements in a tensor `x` without using built-in functions like `torch.numel` or `x.numel`.

### create_tensor_of_pi

```python
def create_tensor_of_pi(M: int, N: int) -> Tensor:
```

Returns a Tensor of shape (M, N) filled entirely with the value 3.14.

### multiples_of_ten

```python
def multiples_of_ten(start: int, stop: int) -> Tensor:
```

Returns a Tensor of dtype `torch.float64` containing all multiples of ten in the range `[start, stop]`.

### slice_indexing_practice

```python
def slice_indexing_practice(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
```

Extracts and returns several subtensors from `x` using slice indexing.

### slice_assignment_practice

```python
def slice_assignment_practice(x: Tensor) -> Tensor:
```

Mutates the first 4 rows and 6 columns of `x` according to a specific pattern.

### shuffle_cols

```python
def shuffle_cols(x: Tensor) -> Tensor:
```

Re-orders the columns of `x` according to a specified pattern.

### reverse_rows

```python
def reverse_rows(x: Tensor) -> Tensor:
```

Reverses the rows of the input tensor `x`.

### take_one_elem_per_col

```python
def take_one_elem_per_col(x: Tensor) -> Tensor:
```

Constructs a new tensor by picking one element from each column of the input tensor `x`.

### make_one_hot

```python
def make_one_hot(x: List[int]) -> Tensor:
```

Constructs a tensor of one-hot-vectors from a list of integers.

### sum_positive_entries

```python
def sum_positive_entries(x: Tensor) -> Tensor:
```

Returns the sum of all positive values in the input tensor `x`.

### reshape_practice

```python
def reshape_practice(x: Tensor) -> Tensor:
```

Returns a reshaped tensor of `x` from shape (24,) to (3, 8) according to a specific pattern.

### zero_row_min

```python
def zero_row_min(x: Tensor) -> Tensor:
```

Returns a copy of the input tensor `x` with the minimum value along each row set to 0.

### batched_matrix_multiply

```python
def batched_matrix_multiply(x: Tensor, y: Tensor, use_loop: bool = True) -> Tensor:
```

Performs batched matrix multiplication between tensors `x` and `y`.

### normalize_columns

```python
def normalize_columns(x: Tensor) -> Tensor:
```

Normalizes the columns of the matrix `x` by subtracting the mean and dividing by the standard deviation of each column.

### mm_on_cpu

```python
def mm_on_cpu(x: Tensor, w: Tensor) -> Tensor:
```

Performs matrix multiplication on CPU.

### mm_on_gpu

```python
def mm_on_gpu(x: Tensor, w: Tensor) -> Tensor:
```

Performs matrix multiplication on GPU, then moves the result back to CPU.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
