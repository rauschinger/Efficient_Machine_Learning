#ifndef MINI_DNN_BACKEND_MATMUL_LIBXSMM_H
#define MINI_DNN_BACKEND_MATMUL_LIBXSMM_H

#include "Matmul.hpp"
#include <ATen/ATen.h>

namespace mini_dnn {
  namespace backend {
    class MatmulLibxsmm;
  }
}

/**
 * Matmul backend using LIBXSMM.
 **/
class mini_dnn::backend::MatmulLibxsmm: public Matmul {
  private:
  public:
    /**
     * Perform the forward pass, i.e., Y = XW.
     *
     * @param i_x matrix X.
     * @param i_w matrix W.
     * @return output of the matmul, i.e., Y.
     **/
    at::Tensor forward( at::Tensor i_x,
                        at::Tensor i_w );
};

#endif