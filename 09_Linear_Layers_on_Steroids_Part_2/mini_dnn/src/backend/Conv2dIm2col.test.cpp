#include <catch2/catch.hpp>
#include "Conv2dIm2col.h"

TEST_CASE( "Tests the convolution operator going through im2col + sgemm.",
           "[conv2d][im2col][forward]" ) {
  int64_t l_size_n = 3;
  int64_t l_size_h = 8;
  int64_t l_size_w = 12;
  int64_t l_size_c = 5;

  int64_t l_size_k = 4;
  int64_t l_size_r = 3;
  int64_t l_size_s = 3;

  // construct input and weight tensors
  at::Tensor l_input = at::rand( {l_size_n, l_size_c, l_size_h , l_size_w} );
  at::Tensor l_weight = at::rand( {l_size_k, l_size_c, l_size_r, l_size_s} );

  // compute solution
  mini_dnn::backend::Conv2dIm2col l_conv2d;

  at::Tensor l_output = l_conv2d.forward( l_input,
                                          l_weight );

  // compute reference
  at::Tensor l_reference = at::conv2d( l_input,
                                       l_weight );

  // check solution
  REQUIRE( at::allclose( l_output, l_reference ) );
}