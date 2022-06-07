#ifndef MINI_DNN_BACKEND_RELU_LIBXSMM_H
#define MINI_DNN_BACKEND_RELU_LIBXSMM_H

#include <ATen/ATen.h>
#include "Relu.hpp"

namespace mini_dnn {
  namespace backend {
    class ReluLibxsmm;
  }
}

/**
 * ReLU backend using the Libxsmm library.
 **/
class mini_dnn::backend::ReluLibxsmm: public Relu {
  private:
  public:
    /**
     * Perform the forward pass.
     *
     * @param i_input input.
     * @return output of the ReLU.
     **/
    at::Tensor forward( at::Tensor i_input );
};

#endif