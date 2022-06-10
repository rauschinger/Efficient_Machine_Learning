#include "Conv2dAten.h"

at::Tensor mini_dnn::backend::Conv2dAten::forward( at::Tensor i_input,
                                                   at::Tensor i_weight ) {
  at::Tensor l_output = at::conv2d( i_input,
                                    i_weight );



for( int64_t l_n = 0; l_n < l_sizes.n; l_n++ ) {
  for( int64_t l_kb = 0; l_kb < l_sizes.kb; l_k++ ) {
    for( int64_t l_p = 0; l_p < l_sizes.p; l_p++ ) {
       for( int64_t l_cb = 0; l_cb < l_sizes.cb; l_cb++ ) {
         for( int64_t l_r = 0; l_r < l_sizes.r; l_r++ ) {
           for( int64_t l_s = 0; l_s < l_sizes.s; l_s++ ) {
             // TODO: execute small matrix kernel
           }
         }
       }
     }
   }
 }



  return l_output;
}