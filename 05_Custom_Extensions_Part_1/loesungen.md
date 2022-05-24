# Serie 5 Custom Extensions Part 1

## Aufgabenstellung

## From PyTorch to Machine Code


![Alt-Text](https://github.com/rauschinger/Efficient_Machine_Learning/blob/main/5_Custom%20Extensions/aufgabenstellung.png)

## Python Extensions

1. Implement the class eml.ext.linear_python.Layer. The constructor should have the following signature:  

        def __init__( self,  
                    i_n_features_input,  
                    i_n_features_output )  

For the time being, implement the forward pass exclusively in the member function forward of the class.

2. Add the class eml.ext.linear_python.Function. Move the details of the forward pass to eml.ext.linear_python.Function. For the time being, skip the backward pass.

3. Add the backward pass to eml.ext.linear_python.Function. Test your implementation!
