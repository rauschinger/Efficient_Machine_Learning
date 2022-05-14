# Serie 5 Custom Extensions

## Aufgabenstellung

## From PyTorch to Machine Code


![Alt-Text](https://github.com/rauschinger/Efficient_Machine_Learning/blob/main/5_Custom%20Extensions/aufgabenstellung.png)

## Python Extensions

1. Implement the class eml.ext.linear_python.Layer. The constructor should have the following signature:  

        def __init__( self,  
                    i_n_features_input,  
                    i_n_features_output )  

For the time being, implement the forward pass exclusively in the member function forward of the class.

2. Add the class eml.ext.linear_python.Function. Move the details of the forward pass to eml.ext.linear_python.Function. For the time being, skip the backward pass.

3. Add the backward pass to eml.ext.linear_python.Function. Test your implementation!

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

## TPP-based Extensions

1. Write a TPP-enabled constructor for our new layer eml.linear_tpp.Layer, i.e.:

    - Allocate memory for internal data (C++);  

    - Generate matrix kernels through LIBXSMM and store respective function pointers in the internal memory section (C++); and

    - Store a pointer to the internal data in a member variable of the layer (Python).

2. Write a destructor for eml.linear_tpp.Layer which frees all allocated memory.

3. Make the pointer to the internal data available in the forward and backward pass when subclassing torch.autograd.Function in the new function eml.linear_tpp.Function.

4. Implement the forward and backward pass in C++ by calling the TPP kernels.