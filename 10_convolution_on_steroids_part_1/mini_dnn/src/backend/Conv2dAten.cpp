#include "Conv2dAten.h"

at::Tensor mini_dnn::backend::Conv2dAten::forward( at::Tensor i_input,
                                                   at::Tensor i_weight ) {
  at::Tensor l_output = at::conv2d( i_input,
                                    i_weight );

  return l_output;
}