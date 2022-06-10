#include <catch2/catch.hpp>
#include "ReluLibxsmm.h"

TEST_CASE( "Tests the ReLU operator through generated LIBXSMM code.",
           "[relu][libxsmm][forward]" ) {
  // sizes of the input
  int64_t l_size_n = 4;
  int64_t l_size_h = 16;
  int64_t l_size_w = 16;
  int64_t l_size_c = 16;

  // construct input tensor
  at::Tensor l_input = at::rand( {l_size_n, l_size_c, l_size_h , l_size_w} );
  l_input -= 0.5f;

  // compute solution
  mini_dnn::backend::ReluLibxsmm l_relu;
  at::Tensor l_output = l_relu.forward( l_input );

  // compute reference
  at::Tensor l_reference = at::max( l_input, at::zeros( l_input.sizes() ) );

  // check solution
  REQUIRE( at::allclose( l_output, l_reference ) );
}