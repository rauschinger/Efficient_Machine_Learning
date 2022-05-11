Serie 3 Pytorch Tensoren

Aufgabenstellung:  

3.1 Tensors

Creation

1. Try different tensor-generating functions and illustrate their behavior. Include torch.zeros, torch.ones, torch.rand and torch.ones_like in your tests.  

2. Use a “list of lists of lists” data structure in Python to allocate memory for tensor T with shape (4, 2, 3) and initialize it to the values in Eq. (3.1.1). Use torch.tensor to convert your Python-native data structure to a PyTorch tensor and print it.  

3. Once again start with your Python-native representation of T. This time use numpy.convert to convert it to a NumPy array first. Then create a PyTorch tensor from the NumPy array and print both.

Opertaions

1. Generate the rank-2 tensors P and Q in PyTorch. Illustrate the behavior of element-wise operations on P and Q. Try at least torch.add and torch.mul. Show that you may also perform element-wise addition or multiplication through the overloaded binary operators + and *.

2. Compute the matrix-matrix product of P and Q<sub>T</sup> by using torch.matmul. Show that you may achieve the same through the overloaded @ operator.

3. Illustrate the behavior of reduction operations, e.g., torch.sum or torch.max.

4. Given two tensors l_tensor_0 and l_tensor_1, explain the difference of the following two code snippets:  

        1   l_tmp = l_tensor_0  
        2   l_tmp[:] = 0  
        
        
        1   l_tmp = l_tensor_1.clone().detach()  
        2   l_tmp[:] = 0  

Storage

1. Create a PyTorch tensor from the rank-3 tensor T given in Eq. (3.1.1). Print the tensor’s size and stride. Print the tensor’s attributes, i.e., its dtype, layout and device.

2. Create a new tensor l_tensor_float from T but use torch.float32 as its dtype.

3. Fix the second dimension of l_tensor_float, i.e., assign l_tensor_fixed to:

l_tensor_fixed = l_tensor_float[:,0,:]
Which metadata of the tensor (size, stride, dtype, layout, device) changed? Which stayed the same?

4. Create an even more complex view of l_tensor_float:

l_tensor_complex_view = l_tensor_float[::2,1,:]
Explain the changes in size and stride.

5. Apply the contiguous function to l_tensor_complex_view. Explain the changes in the stride.

6. Illustrate the internal storage of a tensor by printing corresponding internal data directly.

    Hint:
    The function data_ptr returns the memory address of the internal data. ctypes allows you to directly load data from memory. For example, the following code loads four bytes from address l_data_ptr, interprets the result as a 32-bit floating point value and writes the data to l_data_raw:  

    l_data_raw = (ctypes.c_float).from_address( l_data_ptr )