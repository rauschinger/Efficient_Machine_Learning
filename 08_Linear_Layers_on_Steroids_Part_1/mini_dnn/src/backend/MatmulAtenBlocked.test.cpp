#include <catch2/catch.hpp>
#include "MatmulAtenBlocked.h"

TEST_CASE( "Tests the Matmul forward operator through blocked Aten calls.",
           "[matmul][aten_blocked][forward]" ) {
  // BLAS -> Deep Learning:
  // M: N (batch size)
  // K: C (in features)
  // N: K (out features)

  // sizes of the input
  int64_t l_size_n = 128;
  int64_t l_size_k = 256;
  int64_t l_size_c = 512;

  int64_t l_size_bn =  64;
  int64_t l_size_bk =  32;
  int64_t l_size_bc = 128;

  int64_t l_size_nb = l_size_n / l_size_bn;
  int64_t l_size_kb = l_size_k / l_size_bk;
  int64_t l_size_cb = l_size_c / l_size_bc;

  // construct input tensors
  // X and W are column major matrices, therefore here exists no blocking
  at::Tensor l_x = at::rand( { l_size_n, l_size_c } );
  at::Tensor l_w = at::rand( { l_size_c, l_size_k } );


  //                                           0          1          2          3
  at::Tensor l_x_blocked = l_x.view( {l_sized_nb, l_size_bn, l_size_cb, l_size_bc} );
  l_x_blocked = l_x_blocked.permute( 0, 2, 3, 1 ).contiguous(); // x_blocked hat nun gewünschtes internes datenformat. Mit beispielsweise l_x_blocked[0,0] spingt man über die blöcke (dies muss noch in MatmulAtenBlocked.cpp implementiert werden)

  //mit Y muss das gleiche rückwarts gemacht werden, d.h erst permutieren, neue view definieren und contiguous aufrufen


  l_x_blocked(0,0)

  // TODO:
  //   1) derive blocked X and W
  //   2) compute blocked solution through MatmulAtenBlocked.forward
  //   3) reverse blocking and verify

  // X: nb x cb x bc x bn
  // W: kb x cb x bk x bc
  // Y: kb x nb x bk x bn

  // compute reference
  //at::Tensor l_reference = at::matmul( l_x, l_w );

  // check solution
  //REQUIRE( at::allclose( l_y, l_reference ) );
}