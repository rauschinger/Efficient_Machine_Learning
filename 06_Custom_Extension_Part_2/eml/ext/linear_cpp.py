import torch.autograd
import torch.nn
import eml_ext_linear_cpp

class Function( torch.autograd.Function ):
       @staticmethod
       def forward( io_ctx,
                    i_input,
                    i_weights ):
              io_ctx.save_for_backward( i_input,
                                        i_weights )


              l_input = i_input.contigous()
              l_weights = i_weights.contigous()
              l_output = eml_ext_linear_cpp.forward( i_input,
                                                     i_weights )

              

              return l_output

       @staticmethod
       def backward( io_ctx,
                     i_grad_output ):
              l_input, l_weights = io_ctx.saved_tensors

              l_grad_input = torch.matmul( i_grad_output,
                                           l_weights.transpose(0, 1) )
              
              l_grad_weights = torch.matmul( l_input.transpose(0, 1),
                                             i_grad_output )
              return l_grad_input, l_grad_weights

## @package eml.ext.linear_python.Layer
#  Custom implementation of a linear layer in python.
class Layer( torch.nn.Module ):
  ## Initialize the linear layer.
  #  @param i_n_features_input number of input features.
  #  @param i_n_features_output number of output features.
  def __init__( self,
                i_n_features_input,
                i_n_features_output ):
    super( Layer,
           self ).__init__()

    # store weight matrix A^T
    self.m_weights = torch.nn.Parameter( torch.Tensor( i_n_features_input,
                                                       i_n_features_output ) )

  ## Forward pass of the linear layer.
  #  @param i_input input data.
  #  @return input @ weights.
  def forward( self,
               i_input ):
    l_output = torch.matmul( i_input,
                             self.m_weights )

    return l_output
