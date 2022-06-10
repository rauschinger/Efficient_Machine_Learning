#include "Conv2dIm2col.h"

at::Tensor mini_dnn::backend::Conv2dIm2col::forward( at::Tensor i_input,
                                                     at::Tensor i_weight ) {
  // get involved sizes
  Conv2d::Sizes l_sizes = Conv2d::getSizes( i_input,
                                            i_weight );

  // check that we are not having batched data
  MINI_DNN_CHECK_EQ( l_sizes.bc, 1 );
  MINI_DNN_CHECK_EQ( l_sizes.bk, 1 );


  // assign kernel sizes
  std::vector< int64_t > l_kernel_sizes = { l_sizes.r, 
                                            l_sizes.s };

  // assign dilation
  std::vector< int64_t > l_dilations = { 1, 1 };
  
  // assign padding
  std::vector< int64_t > l_paddings = { 0, 0 };
  
  // assign stride
  std::vector< int64_t > l_strides = { 1,1 };
  
  std::cout << "i_input: " << i_input.sizes() << std::endl;
  
  // cal im2col
  at::Tensor l_input = at::im2col( l_input,
                                   l_kernel_sizes,
                                   l_dilations,
                                   l_paddings,
                                   l_strides );
                                   
  std::cout << "l_input: " << l_input.sizes() << std::endl;
  
  
  // multiply tensor with weights
  std::cout << "i_weight: " << i_weight.sizes() << std::endl;
  
  at::Tensor l_weight = at::flatten( i_weight, 1 );
  
  std::cout << "l_weight: " << l_weight.sizes()  << std::endl;
  
  at::Tensor l_output = at::matmul( l_weight,
                                    l_input );
  
  std::cout << "l_output [n, k, p*q]: " << l_output.sizes() << std::endl;

  l_output = l_output.view( { l_sizes.n, l_sizes.kb, l_sizes.p, l_sizes.q } ); 
  
  return l_output;
}
