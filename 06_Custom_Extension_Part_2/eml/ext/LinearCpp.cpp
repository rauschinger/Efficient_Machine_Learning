#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <iostream>


torch::Tensor forward(  torch::Tensor i_inputs,
                        torch::Tensor i_weights) {

    std::cout << "Hello World from forward" << std::endl;

    torch::Tensor l_result = torch::matmul( i_inputs,
                                            i_weights);               
    
    return l_result;
}

std::vector< torch::Tensor > backward( torch::Tensor i_grad_output,
                                       torch::Tensor i_input,
                                       torch::Tensor i_weights ){

    std::cout << "Hello World from backward" << std::endl;

    std::vector< torch::Tensor > l_result = torch::matmul( i_grad_output, l_weights.transpose(0, 1) ),
                                            torch::matmul( l_input.transpose(0, 1), i_grad_output );

    return l_result;
}

PYBIND11_MODULE( TORCH_EXTENSION_NAME,
                 io_module  ) {
    io_module.def( "forward",
                    &forward,
                    "Forward Pass of our C++ extension");
    io_module.def( "backward",
                    &backward,
                    "Backward Pass of our C++ extension");
}