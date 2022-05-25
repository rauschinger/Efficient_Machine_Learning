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

PYBIND11_MODULE( TORCH_EXTENSION_NAME,
                 io_module  ) {
    io_module.def( "forward",
                    &forward,
                    "Forward Pass of our C++ extension");
}