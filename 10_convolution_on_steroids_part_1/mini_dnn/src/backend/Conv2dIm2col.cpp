#include "Conv2dIm2col.h"

at::Tensor mini_dnn::backend::Conv2dIm2col::forward( at::Tensor i_input,
                                                     at::Tensor i_weight ) {
  // get involved sizes
  Conv2d::Sizes l_sizes = Conv2d::getSizes( i_input,
                                            i_weight );

  // check that we are not having batched data
  MINI_DNN_CHECK_EQ( l_sizes.bc, 1 );
  MINI_DNN_CHECK_EQ( l_sizes.bk, 1 );


  // TODO: finish implementation

  // TODO: remove dummy
  at::Tensor l_output = at::zeros( {l_sizes.n, l_sizes.kb, l_sizes.p, l_sizes.q} );

  return l_output;
}
