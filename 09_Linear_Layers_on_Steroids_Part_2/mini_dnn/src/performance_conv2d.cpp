#include <cstdlib>
#include <tuple>
#include <ATen/ATen.h>
#include "io/Logging.hpp"
#include "backend/Conv2d.hpp"
#include "backend/Conv2dAten.h"
#include "backend/Conv2dIm2col.h"
#include "backend/Relu.hpp"
#include "backend/ReluAten.h"
#include "backend/ReluLibxsmm.h"

/**
 * Measures the performance (time) of the given Conv2d implementation.
 *
 * @param i_n_repetitions number of performed repetitions.
 * @param i_input input tensor.
 * @param i_weight weight tensor.
 * @param io_conv2d benchmarked conv2d implementation.
 * @return duration required for the performed repetitions.
 **/
double timeConv2d( uint64_t                    i_n_repetitions,
                   at::Tensor                & i_input,
                   at::Tensor                & i_weight,
                   mini_dnn::backend::Conv2d & io_conv2d ) {
  std::chrono::high_resolution_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;

  // warmup
  io_conv2d.forward( i_input,
                     i_weight );

  // measure runtime
  l_tp0 = std::chrono::high_resolution_clock::now();
  for( uint64_t l_re = 0; l_re < i_n_repetitions; l_re++ ) {
    io_conv2d.forward( i_input,
                       i_weight );
  }
  l_tp1 = std::chrono::high_resolution_clock::now();

  // derive duration
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );

  return l_dur.count();
}

/**
 * Benchmarks the performance (repetitions, time, gflops) of the given conv2d implementation.
 *
 * @param i_input input tensor.
 * @param i_weight weight tensor.
 * @param io_conv2d benchmarked conv2d implementation.
 * @param i_time_target targeted total execution time; the number of actual repetitions are adjusted accordingly.
 * @param i_n_repetitions_initial initial number of performed repetitions.
 * @return (repetitions, time, gflops).
 **/
std::tuple< uint64_t,
            double,
            double > benchConv2d( at::Tensor                & i_input,
                                  at::Tensor                & i_weight,
                                  mini_dnn::backend::Conv2d & io_conv2d,
                                  double                      i_time_target = 1.0,
                                  uint64_t                    i_n_repetitions_initial = 10 ) {
  double l_dur = timeConv2d( i_n_repetitions_initial,
                             i_input,
                             i_weight,
                             io_conv2d );

  double l_scaling_time = i_time_target / l_dur;
  uint64_t l_n_repetitions_adj = i_n_repetitions_initial * l_scaling_time;
  if( l_n_repetitions_adj == 0 ) {
    l_n_repetitions_adj = 1;
  }

  l_dur = timeConv2d( l_n_repetitions_adj,
                      i_input,
                      i_weight,
                      io_conv2d );

  uint64_t l_n_flops = mini_dnn::backend::Conv2d::nOps( i_input,
                                                        i_weight );

  double l_gflops = l_n_repetitions_adj;
  l_gflops *= l_n_flops / l_dur;
  l_gflops *= 1.0E-9;

  return std::make_tuple( l_n_repetitions_adj,
                          l_dur,
                          l_gflops );
}

/**
 * Measures the performance (time) of the given Conv2d and ReLu implementation when executing separately.
 *
 * @param i_n_repetitions number of performed repetitions.
 * @param i_input input tensor.
 * @param i_weight weight tensor.
 * @param io_conv2d benchmarked conv2d implementation.
 * @param io_relu benchmarked relu implementation.
 * @return duration required for the performed repetitions.
 **/
double timeConv2dRelu( uint64_t                    i_n_repetitions,
                       at::Tensor                & i_input,
                       at::Tensor                & i_weight,
                       mini_dnn::backend::Conv2d & io_conv2d,
                       mini_dnn::backend::Relu   & io_relu ) {
  std::chrono::high_resolution_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;

  // warmup
  at::Tensor l_output = io_conv2d.forward( i_input,
                                           i_weight );
  l_output = io_relu.forward( l_output );

  // measure runtime
  l_tp0 = std::chrono::high_resolution_clock::now();
  for( uint64_t l_re = 0; l_re < i_n_repetitions; l_re++ ) {
    l_output = io_conv2d.forward( i_input,
                                  i_weight );
    l_output = io_relu.forward( l_output );
  }
  l_tp1 = std::chrono::high_resolution_clock::now();

  // derive duration
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );

  return l_dur.count();
}

/**
 * Benchmarks the performance (repetitions, time, gflops) of the given conv2d implementation.
 *
 * @param i_input input tensor.
 * @param i_weight weight tensor.
 * @param io_conv2d benchmarked conv2d implementation.
 * @param io_relu benchmarked relu implementation.
 * @param i_time_target targeted total execution time; the number of actual repetitions are adjusted accordingly.
 * @param i_n_repetitions_initial initial number of performed repetitions.
 * @return (repetitions, time, gflops).
 **/
std::tuple< uint64_t,
            double,
            double > benchConv2dRelu( at::Tensor                & i_input,
                                      at::Tensor                & i_weight,
                                      mini_dnn::backend::Conv2d & io_conv2d,
                                      mini_dnn::backend::Relu   & io_relu,
                                      double                      i_time_target = 1.0,
                                      uint64_t                    i_n_repetitions_initial = 10 ) {
  double l_dur = timeConv2dRelu( i_n_repetitions_initial,
                                 i_input,
                                 i_weight,
                                 io_conv2d,
                                 io_relu );

  double l_scaling_time = i_time_target / l_dur;
  uint64_t l_n_repetitions_adj = i_n_repetitions_initial * l_scaling_time;
  if( l_n_repetitions_adj == 0 ) {
    l_n_repetitions_adj = 1;
  }

  l_dur = timeConv2dRelu( l_n_repetitions_adj,
                          i_input,
                          i_weight,
                          io_conv2d,
                          io_relu );

  uint64_t l_n_flops = mini_dnn::backend::Conv2d::nOps( i_input,
                                                        i_weight );

  double l_gflops = l_n_repetitions_adj;
  l_gflops *= l_n_flops / l_dur;
  l_gflops *= 1.0E-9;

  return std::make_tuple( l_n_repetitions_adj,
                          l_dur,
                          l_gflops );
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
  at::Tensor l_input = at::rand( {l_size_n, l_size_c, l_size_h , l_size_w} ) - 0.5f;
  at::Tensor l_weight = at::rand( {l_size_k, l_size_c, l_size_r, l_size_s} ) - 0.5f;

  at::Tensor l_input_blocked =  at::rand( { l_size_n, l_size_cb, l_size_h , l_size_w, l_size_bc } ) - 0.5f;
  at::Tensor l_weight_blocked =  at::rand( { l_size_kb, l_size_cb, l_size_r, l_size_s, l_size_bc, l_size_bk } ) - 0.5f;

  uint64_t l_n_repetitions = 0;
  double l_time = 0;
  double l_gflops = 0;

  /*
   * Conv2dAten
   */
  MINI_DNN_LOG_INFO << "benchmarking Conv2dAten.." << std::endl;
  mini_dnn::backend::Conv2dAten l_conv2d_aten;

  std::tie( l_n_repetitions,
            l_time,
            l_gflops ) = benchConv2d( l_input,
                                      l_weight,
                                      l_conv2d_aten );

  MINI_DNN_LOG_INFO << "  repetitions:         " << l_n_repetitions << std::endl;
  MINI_DNN_LOG_INFO << "  duration in seconds: " << l_time << std::endl;
  MINI_DNN_LOG_INFO << "  FP32 GFLOPS:         " << l_gflops << std::endl;

  /*
   * Conv2dIm2col
   */
  MINI_DNN_LOG_INFO << "benchmarking Conv2dIm2col.." << std::endl;
  mini_dnn::backend::Conv2dIm2col l_conv2d_im2col;

  std::tie( l_n_repetitions,
            l_time,
            l_gflops ) = benchConv2d( l_input,
                                      l_weight,
                                      l_conv2d_im2col );

  MINI_DNN_LOG_INFO << "  repetitions:         " << l_n_repetitions << std::endl;
  MINI_DNN_LOG_INFO << "  duration in seconds: " << l_time << std::endl;
  MINI_DNN_LOG_INFO << "  FP32 GFLOPS:         " << l_gflops << std::endl;

  /*
   * Conv2dReluAten
   */
  MINI_DNN_LOG_INFO << "benchmarking Conv2dAten+ReluAten.." << std::endl;
  mini_dnn::backend::ReluAten l_relu_aten;

  std::tie( l_n_repetitions,
            l_time,
            l_gflops ) = benchConv2dRelu( l_input,
                                          l_weight,
                                          l_conv2d_aten,
                                          l_relu_aten );

  MINI_DNN_LOG_INFO << "  repetitions:         " << l_n_repetitions << std::endl;
  MINI_DNN_LOG_INFO << "  duration in seconds: " << l_time << std::endl;
  MINI_DNN_LOG_INFO << "  FP32 GFLOPS:         " << l_gflops << std::endl;

  return EXIT_SUCCESS;
}
