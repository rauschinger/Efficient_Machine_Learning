#include "MatmulAtenBlocked.h"

at::Tensor mini_dnn::backend::MatmulAtenBlocked::forward( at::Tensor i_x,
                                                          at::Tensor i_w ) {
  // get involved sizes
  Matmul::Sizes l_sizes = Matmul::getSizes( i_x,
                                            i_w );

  // prepare data for blocked Aten calls
  at::Tensor l_output = at::zeros( {l_sizes.kb, l_sizes.nb, l_sizes.bk, l_sizes.bn} );

  // TODO: finished blocked ATen implementation

  return l_output;
}