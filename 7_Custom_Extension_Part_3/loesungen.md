# Serie 7 Custom Extension Part 3

## Aufgabenstellung

## From PyTorch to Machine Code

## TPP-based Extensions

1. Write a TPP-enabled constructor for our new layer eml.linear_tpp.Layer, i.e.:

    - Allocate memory for internal data (C++);  

    - Generate matrix kernels through LIBXSMM and store respective function pointers in the internal memory section (C++); and

    - Store a pointer to the internal data in a member variable of the layer (Python).

2. Write a destructor for eml.linear_tpp.Layer which frees all allocated memory.

3. Make the pointer to the internal data available in the forward and backward pass when subclassing torch.autograd.Function in the new function eml.linear_tpp.Function.

4. Implement the forward and backward pass in C++ by calling the TPP kernels.