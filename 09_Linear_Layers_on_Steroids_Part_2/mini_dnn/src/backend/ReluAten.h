#ifndef MINI_DNN_BACKEND_RELU_ATEN_H
#define MINI_DNN_BACKEND_RELU_ATEN_H

#include <ATen/ATen.h>
#include "Relu.hpp"

namespace mini_dnn {
  namespace backend {
    class ReluAten;
  }
}

/**
 * ReLU backend using the Aten library.
 **/
class mini_dnn::backend::ReluAten: public Relu {
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