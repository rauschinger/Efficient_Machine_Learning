#include "MatmulReluAten.h"

#include "../io/Logging.hpp"

at::Tensor mini_dnn::backend::MatmulReluAten::forward( at::Tensor i_x,
                                                       at::Tensor i_w ) {
  // call Aten
  // note: A and B are column major, thus we compute
  //       (AB)^T = B^T A^T
  at::Tensor l_y = at::matmul( i_w, i_x );
  l_y = at::relu( l_y );

  return l_y;
}