#include <catch2/catch.hpp>
#include "MatmulReluAten.h"

TEST_CASE( "Tests the fused Matmul+ReLU forward operator through Aten calls.",
           "[matmul_relu][aten][forward]" ) {
  // sizes of the input
  int64_t l_size_n = 32;
  int64_t l_size_k = 56;
  int64_t l_size_c = 54;

  // construct row-major input tensors
  at::Tensor l_x = at::rand( { l_size_n, l_size_c } ) - 0.5;
  at::Tensor l_w = at::rand( { l_size_c, l_size_k } ) - 0.5;

  // convert to column major
  at::Tensor l_x_col_major = l_x.transpose(0, 1).contiguous();
  at::Tensor l_w_col_major = l_w.transpose(0, 1).contiguous();

  // compute solution
  mini_dnn::backend::MatmulReluAten l_matmul;
  at::Tensor l_output = l_matmul.forward( l_x_col_major,
                                          l_w_col_major );

  // convert back to row major
  l_output = l_output.transpose(0, 1).contiguous();

  // compute reference
  at::Tensor l_reference = at::matmul( l_x, l_w );
  l_reference = at::relu( l_reference );

  // check solution
  REQUIRE( at::allclose( l_output,    // self
                         l_reference, // other
                         1E-5,        // rtol
                         1E-5 ) );    // atol
}