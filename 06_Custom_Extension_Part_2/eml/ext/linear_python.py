import torch.autograd
import torch.nn

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
