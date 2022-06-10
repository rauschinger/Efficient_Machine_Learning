#ifndef MINI_DNN_BACKEND_CONV2D_HPP
#define MINI_DNN_BACKEND_CONV2D_HPP

#include "../io/Logging.hpp"
#include <ATen/ATen.h>

namespace mini_dnn {
  namespace backend {
    class Conv2d;
  }
}

class mini_dnn::backend::Conv2d {
  private:

  public:
    typedef struct {
      int64_t n;
      int64_t cb;
      int64_t h;
      int64_t w;
      int64_t bc;

      int64_t kb;
      int64_t r;
      int64_t s;
      int64_t bk;

      int64_t p;
      int64_t q;
    } Sizes;

    /**
     * Gets the convolution's sizes based on the input and weight tensors.
     *
     * @param i_input input tensor.
     * @param i_weight weight tensor.
     * @return sizes occuring in the convolution.
     **/
    static Sizes getSizes( at::Tensor i_input,
                           at::Tensor i_weight ) {
      int64_t l_n_dims_input = i_input.ndimension();
      int64_t l_n_dims_weight = i_weight.ndimension();

      // get raw sizes from Aten
      c10::IntArrayRef l_sizes_input = i_input.sizes();
      c10::IntArrayRef l_sizes_weight = i_weight.sizes();

      // interpret common sizes
      Sizes l_sizes;

      MINI_DNN_CHECK_GE( l_n_dims_input,  4 );
      MINI_DNN_CHECK_GE( l_n_dims_weight, 4 );

      l_sizes.n  = l_sizes_input[0];
      l_sizes.cb = l_sizes_input[1];
      l_sizes.h  = l_sizes_input[2];
      l_sizes.w  = l_sizes_input[3];

      l_sizes.kb = l_sizes_weight[0];
      MINI_DNN_CHECK_EQ( l_sizes_weight[1], l_sizes.cb );
      l_sizes.r  = l_sizes_weight[2];
      l_sizes.s  = l_sizes_weight[3];

      // derive p and q through filter size
      l_sizes.p = l_sizes.h - l_sizes.r + 1;
      l_sizes.q = l_sizes.w - l_sizes.s + 1;

      // set blocking sizes
      if(     l_n_dims_input == 5
           && l_n_dims_weight == 6 ) {
        l_sizes.bc = l_sizes_input[4]; 
        MINI_DNN_CHECK_EQ( l_sizes_weight[4], l_sizes.bc );
        l_sizes.bk = l_sizes_weight[5];
      }
      else if(    l_n_dims_input  == 4
               && l_n_dims_weight == 4 ) {
        l_sizes.bc = 1;
        l_sizes.bk = 1;
      }
      else{ MINI_DNN_CHECK( false ); }

      return l_sizes;
    }

    /**
     * Derives the number of required operations to convolve the input tensor with the weight tensor.
     *
     * @param i_input input tensor.
     * @param i_weight weight tensor.
     * @return number of required operations.
     **/
    static uint64_t nOps( at::Tensor i_input,
                          at::Tensor i_weight ) {
      Sizes l_sizes = getSizes( i_input,
                                i_weight );

      uint64_t l_n_flops = 1;
      l_n_flops *= l_sizes.n;
      l_n_flops *= l_sizes.kb * l_sizes.bk;
      l_n_flops *= l_sizes.p;
      l_n_flops *= l_sizes.q;
      l_n_flops *= l_sizes.cb * l_sizes.bc;
      l_n_flops *= l_sizes.r;
      l_n_flops *= l_sizes.s;
      l_n_flops *= 2;

      return l_n_flops;
    }

    /**
     * Perform the forward pass.
     * Virtual member, implemented by the actual backends.
     *
     * @param i_input input tensor.
     * @param i_weight weight tensor.
     * @return output of the convolution.
     **/
    virtual at::Tensor forward( at::Tensor i_input,
                                at::Tensor i_weight ) = 0;
};

#endif