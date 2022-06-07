#ifndef MINI_DNN_BACKEND_RELU_HPP
#define MINI_DNN_BACKEND_RELU_HPP

#include "../io/Logging.hpp"
#include <ATen/ATen.h>

namespace mini_dnn {
  namespace backend {
    class Relu;
  }
}

class mini_dnn::backend::Relu {
  private:

  public:
    /**
     * Perform the forward pass.
     * Virtual member, implemented by the actual backends.
     *
     * @param i_input input tensor.
     * @param i_weight weight tensor.
     * @return output of the ReLU.
     **/
    virtual at::Tensor forward( at::Tensor i_input ) = 0;
};

#endif