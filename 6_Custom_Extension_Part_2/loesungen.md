# Serie 6 Custom Extension Part 2

## Aufgabenstellung

## From PyTorch to Machine Code

## C++ Extensions

1. Implement the C++ function hello() which simply prints “Hello World!”. Use pybind11 and setuptools to make your function available in Python. Call it!

2. Implement the two Python classes eml.linear_cpp.Layer and eml.linear_cpp.Function. Call the two C++ functions forward and backward from eml.linear_cpp.Function for the forward and backward pass. Implement the two functions in the file eml/linear_cpp/FunctionCpp.cpp and use the following declarations:

        torch::Tensor forward( torch::Tensor i_input,  
                            torch::Tensor i_weight );  

        std::vector< torch::Tensor > backward( torch::Tensor i_grad_output,  
                                            torch::Tensor i_input,  
                                            torch::Tensor i_weights );  

Use the ATen tensor library for the actual matrix-matrix multiplication.

3. Adjust the implementation of the functions forward and backward by replacing the ATen matrix-matrix multiplications with your own C++-only implementation.