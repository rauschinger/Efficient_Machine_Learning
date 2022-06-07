#include <catch2/catch.hpp>
#include "ReluAten.h"

TEST_CASE( "Tests the ReLU forward operator through Aten.",
           "[relu][aten][forward]" ) {
  int64_t l_size_h = 4;
  int64_t l_size_w = 8;

  // construct input and weight tensors
  at::Tensor l_input = at::rand( {l_size_h , l_size_w} );
  l_input -= 0.5f;

  // compute solution
  mini_dnn::backend::ReluAten l_relu;
  at::Tensor l_output = l_relu.forward( l_input );

  // compute reference
  at::Tensor l_reference = at::max( l_input, at::zeros( l_input.sizes() ) );

  // check solution
  REQUIRE( at::allclose( l_output, l_reference ) );
}