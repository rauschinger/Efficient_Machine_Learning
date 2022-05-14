# Serie 5

## Aufgabenstellung

1. Assume the following exemplary data for the input batch X, the weights W, and the vector-Jacobian product batch of the loss function L w.r.t. the output batch Y:

 
  
 
 
 
 
Derive the output of the linear layer  manually. Manually derive the vector-Jacobian products 
 
 and 
 
.

Construct a torch.nn.Linear layer with the following arguments in_features = 3, out_features = 2 and bias = False. Initialize the weights  of the layer manually using the given exemplary input data.

Apply the layer’s forward function to the exemplary input batch  and print the result.

Run the backward pass and print the gradients w.r.t. the input  and the weights .