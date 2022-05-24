#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <iostream>


void hello()    {
    std::cout << "Hello World!" << std::endl;
}


PYBIND11_MODULE( TORCH_EXTENSION_NAME,
                 io_module  ) {
    io_module.def( "hello",
                    &hello,
                    "simple example which prints hello world");
}