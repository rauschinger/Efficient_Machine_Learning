#include "MatmulAtenBlocked.h"

at::Tensor mini_dnn::backend::MatmulAtenBlocked::forward( at::Tensor i_x,
                                                          at::Tensor i_w ) {
  // get involved sizes
  Matmul::Sizes l_sizes = Matmul::getSizes( i_x,
                                            i_w );

  // prepare data for blocked Aten calls
  at::Tensor l_output = at::zeros( {l_sizes.kb, l_sizes.nb, l_sizes.bk, l_sizes.bn} );

  for( int64_t l_kb = 0; l_kb < l_sizes.kb; l_kb++ ) {
      for( int64_t l_nb = 0; l_nb < l_sizes.nb; l_nb++ ) {
        for( int64_t l_cb = 0; l_cb < l_sizes.cb; l_cb++ ) {
          l_output[l_kb][l_nb] += at::matmul( i_w[l_kb][l_cb],
                                              i_x[l_nb][l_cb] );
        }
      }
  }

  return l_output;
}