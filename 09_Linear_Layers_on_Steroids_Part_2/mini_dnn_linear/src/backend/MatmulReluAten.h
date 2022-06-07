#ifndef MINI_DNN_BACKEND_MATMUL_RELU_ATEN_H
#define MINI_DNN_BACKEND_MATMUL_RELU_ATEN_H

#include <ATen/ATen.h>
#include "Matmul.hpp"

namespace mini_dnn {
  namespace backend {
    class MatmulReluAten;
  }
}

/**
 * Matmul+ReLU backend using the Aten calls.
 **/
class mini_dnn::backend::MatmulReluAten: public Matmul {
  private:
  public:
    /**
     * Perform the forward pass, i.e., Y = XW.
     * Note: Both matrices are expected column-major
     *       which is different from PyTorch's default.
     *
     * @param i_x column-major matrix X.
     * @param i_w column-major matrix W.
     * @return output of the matmul, i.e., Y.
     **/
    at::Tensor forward( at::Tensor i_x,
                        at::Tensor i_w );
};

#endif