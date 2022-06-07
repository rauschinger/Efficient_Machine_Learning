#ifndef MINI_DNN_BACKEND_CONV2D_ATEN_H
#define MINI_DNN_BACKEND_CONV2D_ATEN_H

#include <ATen/ATen.h>
#include "Conv2d.hpp"

namespace mini_dnn {
  namespace backend {
    class Conv2dAten;
  }
}

/**
 * Conv2d backend using the Aten library.
 **/
class mini_dnn::backend::Conv2dAten: public Conv2d {
  private:
  public:
    /**
     * Perform the forward pass.
     *
     * @param i_input input.
     * @param i_weight weights.
     * @return output of the convolution.
     **/
    at::Tensor forward( at::Tensor i_input,
                        at::Tensor i_weight );
};

#endif