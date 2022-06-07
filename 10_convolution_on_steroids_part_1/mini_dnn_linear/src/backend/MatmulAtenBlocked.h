#ifndef MINI_DNN_BACKEND_MATMUL_ATEN_BLOCKED_H
#define MINI_DNN_BACKEND_MATMUL_ATEN_BLOCKED_H

#include <ATen/ATen.h>
#include "Matmul.hpp"

namespace mini_dnn {
  namespace backend {
    class MatmulAtenBlocked;
  }
}

/**
 * Matmul backend using the blocked Aten calls.
 **/
class mini_dnn::backend::MatmulAtenBlocked: public Matmul {
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