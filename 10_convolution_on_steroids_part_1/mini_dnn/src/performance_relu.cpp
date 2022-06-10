#include <cstdlib>
#include <ATen/ATen.h>
#include "io/Logging.hpp"
#include "backend/Relu.hpp"
#include "backend/ReluAten.h"
#include "backend/ReluLibxsmm.h"

/**
 * Measures the performance (time) of the given ReLU implementation.
 *
 * @param i_n_repetitions number of performed repetitions.
 * @param i_input input tensor.
 * @param io_relu benchmarked relu implementation.
 * @return duration required for the performed repetitions.
 **/
double timeRelu( uint64_t                    i_n_repetitions,
                   at::Tensor              & i_input,
                   mini_dnn::backend::Relu & io_relu ) {
  std::chrono::high_resolution_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;

  // warmup
  io_relu.forward( i_input );

  // measure runtime
  l_tp0 = std::chrono::high_resolution_clock::now();
  for( uint64_t l_re = 0; l_re < i_n_repetitions; l_re++ ) {
    io_relu.forward( i_input );
  }
  l_tp1 = std::chrono::high_resolution_clock::now();

  // derive duration
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );

  return l_dur.count();
}

/**
 * Benchmarks the performance (repetitions, time) of the given ReLU implementation.
 *
 * @param i_input input tensor.
 * @param io_relu benchmarked relu implementation.
 * @param i_time_target targeted total execution time; the number of actual repetitions are adjusted accordingly.
 * @param i_n_repetitions_initial initial number of performed repetitions.
 * @return (repetitions, time).
 **/
std::tuple< uint64_t,
            double > benchRelu( at::Tensor              & i_input,
                                mini_dnn::backend::Relu & io_relu,
                                double                    i_time_target = 1.0,
                                uint64_t                  i_n_repetitions_initial = 500 ) {
  double l_dur = timeRelu( i_n_repetitions_initial,
                           i_input,
                           io_relu );

  double l_scaling_time = i_time_target / l_dur;
  uint64_t l_n_repetitions_adj = i_n_repetitions_initial * l_scaling_time;
  if( l_n_repetitions_adj == 0 ) {
    l_n_repetitions_adj = 1;
  }

  l_dur = timeRelu( l_n_repetitions_adj,
                    i_input,
                    io_relu );

  return std::make_tuple( l_n_repetitions_adj,
                          l_dur );
}

int main() {
  MINI_DNN_LOG_INFO << "running performance tests" << std::endl;

  int64_t l_size_n = 48;
  int64_t l_size_h = 32 + 2; // zero-padding
  int64_t l_size_w = 32 + 2; // zero-padding
  int64_t l_size_c = 512;

  int64_t l_size_k = 512;
  int64_t l_size_r = 3;
  int64_t l_size_s = 3;

  int64_t l_size_p = l_size_h - l_size_r + 1;
  int64_t l_size_q = l_size_w - l_size_s + 1;

  // blocking of C and K
  int64_t l_size_bc = 128;
  int64_t l_size_bk = 64;

  int64_t l_size_cb = l_size_c / l_size_bc;
  int64_t l_size_kb = l_size_k / l_size_bk;

  MINI_DNN_LOG_INFO << "here are our dimensions:" << std::endl;
  MINI_DNN_LOG_INFO << "  n: " << l_size_n << std::endl;
  MINI_DNN_LOG_INFO << "  h: " << l_size_h << std::endl;
  MINI_DNN_LOG_INFO << "  w: " << l_size_w << std::endl;
  MINI_DNN_LOG_INFO << "  c: " << l_size_c << std::endl;
  MINI_DNN_LOG_INFO << "  k: " << l_size_k << std::endl;
  MINI_DNN_LOG_INFO << "  r: " << l_size_r << std::endl;
  MINI_DNN_LOG_INFO << "  s: " << l_size_s << std::endl;
  MINI_DNN_LOG_INFO << "  p: " << l_size_p << std::endl;
  MINI_DNN_LOG_INFO << "  q: " << l_size_q << std::endl;

  MINI_DNN_LOG_INFO << "  bc: " << l_size_bc << std::endl;
  MINI_DNN_LOG_INFO << "  bk: " << l_size_bk << std::endl;
  MINI_DNN_LOG_INFO << "  cb: " << l_size_cb << std::endl;
  MINI_DNN_LOG_INFO << "  kb: " << l_size_kb << std::endl;

  // construct input and weight tensors
  at::Tensor l_input = at::rand( {l_size_n, l_size_c, l_size_h , l_size_w} );
  at::Tensor l_input_blocked =  at::rand( { l_size_n, l_size_cb, l_size_h , l_size_w, l_size_bc, } );

  uint64_t l_n_repetitions = 0;
  double l_time = 0;

  /*
   * ReluAten
   */
  MINI_DNN_LOG_INFO << "benchmarking ReluAten.." << std::endl;
  mini_dnn::backend::ReluAten l_relu_aten;

  std::tie( l_n_repetitions,
            l_time ) = benchRelu( l_input_blocked,
                                  l_relu_aten );

  MINI_DNN_LOG_INFO << "  repetitions:         " << l_n_repetitions << std::endl;
  MINI_DNN_LOG_INFO << "  duration in seconds: " << l_time << std::endl;
  MINI_DNN_LOG_INFO << "  repetitions / second: " << l_n_repetitions / l_time << std::endl;

  /*
   * ReluLibxsmm
   */
  MINI_DNN_LOG_INFO << "benchmarking ReluLibxsmm.." << std::endl;
  mini_dnn::backend::ReluLibxsmm l_relu_libxsmm;

  std::tie( l_n_repetitions,
            l_time ) = benchRelu( l_input_blocked,
                                  l_relu_libxsmm );

  MINI_DNN_LOG_INFO << "  repetitions:          " << l_n_repetitions << std::endl;
  MINI_DNN_LOG_INFO << "  duration in seconds:  " << l_time << std::endl;
  MINI_DNN_LOG_INFO << "  repetitions / second: " << l_n_repetitions / l_time << std::endl;

  return EXIT_SUCCESS;
}
