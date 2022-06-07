#ifndef MINI_DNN_BACKEND_MATMUL_HPP
#define MINI_DNN_BACKEND_MATMUL_HPP

#include "../io/Logging.hpp"
#include <ATen/ATen.h>

namespace mini_dnn {
  namespace backend {
    class Matmul;
  }
}

class mini_dnn::backend::Matmul {
  private:

  public:
    typedef struct {
      // number of blocks
      int64_t nb;
      int64_t kb;
      int64_t cb;

      // block sizes
      int64_t bn;
      int64_t bk;
      int64_t bc;
    } Sizes;

    /**
     * Gets the matmul's sizes based on the input matrices A and B.
     *
     * @param i_x tensor representing X.
     * @param i_w tensor representing W.
     * @return sizes occuring in the matmul.
     **/
    static Sizes getSizes( at::Tensor i_x,
                           at::Tensor i_w ) {
      int64_t l_n_dims_x = i_x.ndimension();
      int64_t l_n_dims_w = i_w.ndimension();

      MINI_DNN_CHECK_GE( l_n_dims_x, 2 );
      MINI_DNN_CHECK_GE( l_n_dims_w, 2 );

      // get raw sizes from Aten
      c10::IntArrayRef l_sizes_x = i_x.sizes();
      c10::IntArrayRef l_sizes_w = i_w.sizes();

      // interpret common sizes
      Sizes l_sizes;

      if( l_n_dims_x == 4 &&
          l_n_dims_w == 4 ) {
        l_sizes.nb  = l_sizes_x[0];
        l_sizes.kb  = l_sizes_w[0];
        l_sizes.cb  = l_sizes_x[1];
        MINI_DNN_CHECK_EQ( l_sizes_w[1], l_sizes.cb );

        l_sizes.bn = l_sizes_x[3];
        l_sizes.bk = l_sizes_w[2];
        l_sizes.bc = l_sizes_x[2];
        MINI_DNN_CHECK_EQ( l_sizes_w[3], l_sizes.bc );
      }
      else if( l_n_dims_x == 2 &&
               l_n_dims_w == 2 ) {
        l_sizes.nb  = l_sizes_x[1];
        l_sizes.kb  = l_sizes_w[0];
        l_sizes.cb  = l_sizes_x[0];
        MINI_DNN_CHECK_EQ( l_sizes_w[1], l_sizes.cb );

        l_sizes.bn = 1;
        l_sizes.bk = 1;
        l_sizes.bc = 1;
      }
      else{ MINI_DNN_CHECK( false ); }

      return l_sizes;
    }

    /**
     * Derives the number of required operations to convolve the input tensor with the weight tensor.
     *
     * @param i_x input tensor X.
     * @param i_w weight tensor W.
     * @return number of required operations.
     **/
    static uint64_t nOps( at::Tensor i_x,
                          at::Tensor i_w ) {
      Sizes l_sizes = getSizes( i_x,
                                i_w );

      uint64_t l_n_flops = 1;
      l_n_flops *= l_sizes.nb * l_sizes.bn;
      l_n_flops *= l_sizes.kb * l_sizes.bk;
      l_n_flops *= l_sizes.cb * l_sizes.bc;
      l_n_flops *= 2;

      return l_n_flops;
    }

    /**
     * Perform the forward pass.
     * Virtual member, implemented by the actual backends.
     *
     * @param i_x input tensor X.
     * @param i_w weight tensor W.
     * @return output of the convolution.
     **/
    virtual at::Tensor forward( at::Tensor i_x,
                                at::Tensor i_w ) = 0;
};

#endif