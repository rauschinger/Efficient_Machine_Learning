#include <catch2/catch.hpp>
#include "MatmulLibxsmm.h"

TEST_CASE( "Tests the Matmul forward operator through LIBXSMM calls.",
           "[matmul][libxsmm][forward]" ) {
  // BLAS -> Deep Learning:
  // M: N (batch size)
  // K: C (in features)
  // N: K (out features)

  // sizes of the input
  int64_t l_size_m = 128;
  int64_t l_size_n = 256;
  int64_t l_size_k = 512;

  int64_t l_size_bm =  64;
  int64_t l_size_bn =  32;
  int64_t l_size_bk = 128;

  // construct input tensors
  at::Tensor l_a = at::rand( { l_size_m, l_size_k } );
  at::Tensor l_b = at::rand( { l_size_k, l_size_n } );

  // blocking
  // A: m x k x bk x bm
  // B: n x k x bn x bk
  // C: n x m x bn x bm

  //                                                     0          1                     2           3
  at::Tensor l_a_blocked = l_a.view( { l_size_m / l_size_bm, l_size_bm, l_size_k / l_size_bk, l_size_bk } );
  l_a_blocked = l_a_blocked.permute( { 0, 2, 3, 1 } ).contiguous();

  //                                                      0          1                     2          3
  at::Tensor l_b_blocked = l_b.view( { l_size_k / l_size_bk, l_size_bk, l_size_n / l_size_bn, l_size_bn } );
  l_b_blocked = l_b_blocked.permute( { 2, 0, 3, 1 } ).contiguous();

  // compute solution
  mini_dnn::backend::MatmulLibxsmm l_matmul;
  at::Tensor l_output_blocked = l_matmul.forward( l_a_blocked, l_b_blocked );

  // reverse blocking
  at::Tensor l_output = l_output_blocked.permute( {1, 3, 0, 2} ).contiguous();
  l_output = l_output.view( { l_size_m, l_size_n } );

  // compute reference
  at::Tensor l_reference = at::matmul( l_a, l_b );

  // check solution
  REQUIRE( at::allclose( l_output, l_reference ) );
}