#include <cstdlib>
#include <tuple>
#include <ATen/ATen.h>
#include "io/Logging.hpp"
#include "backend/MatmulReluAten.h"
#include "backend/MatmulLibxsmm.h"

/**
 * Measures the performance (time) of the given matmul implementation.
 *
 * @param i_n_repetitions number of performed repetitions.
 * @param i_input input tensor.
 * @param i_weight weight tensor.
 * @param io_matmul benchmarked matmul implementation.
 * @return duration required for the performed repetitions.
 **/
double timeMatmul( uint64_t                    i_n_repetitions,
                   at::Tensor                & i_input,
                   at::Tensor                & i_weight,
                   mini_dnn::backend::Matmul & io_matmul ) {
  std::chrono::high_resolution_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;

  // warmup
  io_matmul.forward( i_input,
                     i_weight );

  // measure runtime
  l_tp0 = std::chrono::high_resolution_clock::now();
  for( uint64_t l_re = 0; l_re < i_n_repetitions; l_re++ ) {
    io_matmul.forward( i_input,
                       i_weight );
  }
  l_tp1 = std::chrono::high_resolution_clock::now();

  // derive duration
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );

  return l_dur.count();
}

/**
 * Benchmarks the performance (repetitions, time, gflops) of the given matmul implementation.
 *
 * @param i_input input tensor.
 * @param i_weight weight tensor.
 * @param io_matmul benchmarked matmul implementation.
 * @param i_time_target targeted total execution time; the number of actual repetitions are adjusted accordingly.
 * @param i_n_repetitions_initial initial number of performed repetitions.
 * @return (repetitions, time, gflops).
 **/
std::tuple< uint64_t,
            double,
            double > benchMatmul( at::Tensor                & i_input,
                                  at::Tensor                & i_weight,
                                  mini_dnn::backend::Matmul & io_matmul,
                                  double                      i_time_target = 1.0,
                                  uint64_t                    i_n_repetitions_initial = 1000 ) {
  double l_dur = timeMatmul( i_n_repetitions_initial,
                             i_input,
                             i_weight,
                             io_matmul );

  double l_scaling_time = i_time_target / l_dur;
  uint64_t l_n_repetitions_adj = i_n_repetitions_initial * l_scaling_time;
  if( l_n_repetitions_adj == 0 ) {
    l_n_repetitions_adj = 1;
  }

  l_dur = timeMatmul( l_n_repetitions_adj,
                      i_input,
                      i_weight,
                      io_matmul );

  uint64_t l_n_flops = mini_dnn::backend::Matmul::nOps( i_input,
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

  // sizes of the input

  //diese größen müssen noch auf VGG angepasst werden (4096x4096)
  int64_t l_size_n = 128;
  int64_t l_size_k = 768;
  int64_t l_size_c = 512;

  int64_t l_size_bn =  64;
  int64_t l_size_bk =  16;
  int64_t l_size_bc = 128;

  int64_t l_size_nb = l_size_n / l_size_bn;
  int64_t l_size_kb = l_size_k / l_size_bk;
  int64_t l_size_cb = l_size_c / l_size_bc;

  MINI_DNN_LOG_INFO << "here are our dimensions:" << std::endl;
  MINI_DNN_LOG_INFO << "  n: " << l_size_n << std::endl;
  MINI_DNN_LOG_INFO << "  k: " << l_size_k << std::endl;
  MINI_DNN_LOG_INFO << "  c: " << l_size_c << std::endl;

  MINI_DNN_LOG_INFO << "  bn: " << l_size_bn << std::endl;
  MINI_DNN_LOG_INFO << "  bk: " << l_size_bk << std::endl;
  MINI_DNN_LOG_INFO << "  bc: " << l_size_bc << std::endl;

  MINI_DNN_LOG_INFO << "  nb: " << l_size_nb << std::endl;
  MINI_DNN_LOG_INFO << "  kb: " << l_size_kb << std::endl;
  MINI_DNN_LOG_INFO << "  cb: " << l_size_cb << std::endl;

  // construct input and weight tensors
  at::Tensor l_input = at::rand( {l_size_n, l_size_c} ) - 0.5f;
  at::Tensor l_weight = at::rand( {l_size_c, l_size_k} ) - 0.5f;

  at::Tensor l_input_col_major = l_input.transpose(0, 1).contiguous();
  at::Tensor l_weight_col_major = l_weight.transpose(0, 1).contiguous();

  at::Tensor l_input_blocked =  at::rand( { l_size_nb, l_size_cb, l_size_bc, l_size_bn } ) - 0.5f;
  at::Tensor l_weight_blocked =  at::rand( { l_size_kb, l_size_cb, l_size_bk, l_size_bc } ) - 0.5f;

  uint64_t l_n_repetitions = 0;
  double l_time = 0;
  double l_gflops = 0;

  /*
   * MatmulReluAten
   */
  MINI_DNN_LOG_INFO << "benchmarking MatmulReluAten.." << std::endl;
  mini_dnn::backend::MatmulReluAten l_matmul_relu_aten;


  //l_input_col_major = X, l_weight_col_major = W, l_matmul_relu_aten = Matmul Funktion
  std::tie( l_n_repetitions,
            l_time,
            l_gflops ) = benchMatmul( l_input_col_major,
                                      l_weight_col_major,
                                      l_matmul_relu_aten );

  MINI_DNN_LOG_INFO << "  repetitions:         " << l_n_repetitions << std::endl;
  MINI_DNN_LOG_INFO << "  duration in seconds: " << l_time << std::endl;
  MINI_DNN_LOG_INFO << "  FP32 GFLOPS:         " << l_gflops << std::endl;

  // TODO: benchmark the performance of your new implementations, i.e.,
  //       MatmulAtenBlocked nad MatmulLibxsmm
  //Dazu Zeile 132 - 148 kopieren und wieder einfügen mit anpassung dass die MatmulAtenBlocked MatmulLibxsmm aufgerufen werden


}