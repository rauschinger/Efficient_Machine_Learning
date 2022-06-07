#include <catch2/catch.hpp>
#include "Conv2d.hpp"

TEST_CASE( "Tests sizes derivation for the default conv2d layout.",
           "[conv2d][sizes_default]" ) {
  int64_t l_size_n = 4;
  int64_t l_size_h = 16 + 2; // zero-padding
  int64_t l_size_w = 16 + 2; // zero-padding
  int64_t l_size_c = 16;

  int64_t l_size_k = 32;
  int64_t l_size_r = 3;
  int64_t l_size_s = 3;
  
  int64_t l_size_p = l_size_h - (l_size_r / 2)*2;
  int64_t l_size_q = l_size_w - (l_size_s / 2)*2;

  at::Tensor l_input = at::rand( {l_size_n, l_size_c, l_size_h , l_size_w} );
  at::Tensor l_weight = at::rand( {l_size_k, l_size_c, l_size_r, l_size_s} );

  mini_dnn::backend::Conv2d::Sizes l_sizes = mini_dnn::backend::Conv2d::getSizes( l_input,
                                                                                  l_weight ); 

  REQUIRE( l_sizes.bc == 1 );
  REQUIRE( l_sizes.bk == 1 );

  REQUIRE( l_sizes.n  == l_size_n );
  REQUIRE( l_sizes.h  == l_size_h );
  REQUIRE( l_sizes.cb == l_size_c );
  REQUIRE( l_sizes.w  == l_size_w );

  REQUIRE( l_sizes.kb == l_size_k );
  REQUIRE( l_sizes.r  == l_size_r );
  REQUIRE( l_sizes.s  == l_size_s );

  REQUIRE( l_sizes.p  == l_size_p );
  REQUIRE( l_sizes.q  == l_size_q );
}

TEST_CASE( "Tests sizes derivation for the blocked conv2d layout.",
           "[conv2d][sizes_blocked]" ) {
  int64_t l_size_bc = 8;
  int64_t l_size_bk = 4;

  int64_t l_size_n = 4;
  int64_t l_size_h = 16 + 2; // zero-padding
  int64_t l_size_w = 16 + 2; // zero-padding
  int64_t l_size_cb = 16 / l_size_bc;

  int64_t l_size_kb = 32 / l_size_bk;
  int64_t l_size_r = 3;
  int64_t l_size_s = 3;

  int64_t l_size_p = l_size_h - (l_size_r / 2)*2;
  int64_t l_size_q = l_size_w - (l_size_s / 2)*2;

  at::Tensor l_input = at::rand( {l_size_n, l_size_cb, l_size_h , l_size_w, l_size_bc } );
  at::Tensor l_weight = at::rand( {l_size_kb, l_size_cb, l_size_r, l_size_s, l_size_bc, l_size_bk} );

  mini_dnn::backend::Conv2d::Sizes l_sizes = mini_dnn::backend::Conv2d::getSizes( l_input,
                                                                                  l_weight ); 


  REQUIRE( l_sizes.bc == l_size_bc );
  REQUIRE( l_sizes.bk == l_size_bk );

  REQUIRE( l_sizes.n == l_size_n );
  REQUIRE( l_sizes.h == l_size_h );
  REQUIRE( l_sizes.cb == l_size_cb );
  REQUIRE( l_sizes.w == l_size_w );

  REQUIRE( l_sizes.kb == l_size_kb );
  REQUIRE( l_sizes.r == l_size_r );
  REQUIRE( l_sizes.s == l_size_s );

  REQUIRE( l_sizes.p == l_size_p );
  REQUIRE( l_sizes.q == l_size_q );
}
