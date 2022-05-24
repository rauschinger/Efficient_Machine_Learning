#!/usr/bin/python3
import torch
import torch.nn
import eml.ext.linear_python

print( '##########################' )
print( '## linear layer example ##' )
print( '##########################' )
l_w = torch.tensor( [ [1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0] ],
                    requires_grad = True )

l_x = torch.tensor( [ [7.0, 8.0, 9.0],
                      [10.0, 11.0, 12.0] ],
                    requires_grad = True )

l_linear_torch = torch.nn.Linear( 3,
                                  2,
                                  bias = False )
l_linear_torch.weight = torch.nn.Parameter( l_w )

l_result = l_linear_torch.forward( l_x )

l_grad = torch.tensor( [ [ 1.0, 2.0 ],
                         [ 3.0, 4.0] ] )
l_result.backward( l_grad )


print( 'result:' )
print( l_result )
print( 'dfdx:' )
print( l_x.grad )
print( 'dfdw:' )
print( l_linear_torch.weight.grad )

print( '#####################################' )
print( '## EML linear python layer example ##' )
print( '#####################################' )
l_w = torch.tensor( [ [1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0] ],
                    requires_grad = True )

l_x = torch.tensor( [ [7.0, 8.0, 9.0],
                      [10.0, 11.0, 12.0] ],
                    requires_grad = True )

l_linear_eml_python = eml.ext.linear_python.Layer( 3,
                                                   2 )
l_linear_eml_python.m_weights = torch.nn.Parameter( l_w.transpose( 0, 1 ) )

l_result = l_linear_eml_python.forward( l_x )

print( 'result:' )
print( l_result )
