#include "ReluAten.h"

at::Tensor mini_dnn::backend::ReluAten::forward( at::Tensor i_input ) {
  at::Tensor l_output = at::relu( i_input );

  return l_output;
}