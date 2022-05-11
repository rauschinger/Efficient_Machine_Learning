Serie 4 Pytorch Multilayer Perceptron

Datasets and Data Loaders

1. Create a training and test dataset by calling torchvision.datasets.FashionMNIST. Use the transformation torchvision.transforms.ToTensor() to convert the data to PyTorch tensors.

2. Visualize a few images and show their labels. Use Matplotlib for your plots. Save your visualizations in a PDF file by using PdfPages. An example is given in the Multipage PDF demo.

3. Wrap the datasets into data loaders. Use torch.utils.data.DataLoader and illustrate the parameter bach_size.


Training and Validation

1. Implement the class Model in the module eml.mlp.model which contains the MultiLayer Perceptron (MLP).

2. Implement the training loop and print the total training loss after every epoch. For the time being implement the training loop directly in your main function.

3. Move the training loop to the module eml.mlp.trainer. Use the template in Listing 3.2.1 to guide your implementation.

4. Implement the module eml.mlp.tester. Use the template in Listing 3.2.2 to guide your implementation. The module’s only function test simply applies the MLP to the given data and returns the obtained total loss and number of correctly predicted samples.

    Hint:
    When testing your model, switch to evaluation through nn.Module.eval(). Don’t forget to switch back to training mode afterwards if needed. Further information is available from the article Autograd mechanics.


Visualization

1. Implement a Fashion MNIST visualization module in eml.vis.fashion_mnist. Use the template in Listing 3.2.3 to guide your implementation. The module’s function plot function takes the argument i_off for the offset of the first visualized image and the argument i_stride for the stride between images. For example, if i_off=5 and i_stride=17, the function would plot the images with ids 5, 21, 38, and so on.

2. Monitor your training process by visualizing the test data after every ten epochs. Use the stride feature of eml.vis.fashion_mnist.plot to keep the file sizes small.

Batch Jobs

1. Write a job script which powers the training of your MLP training.

2. Submit your job and maybe grab a coffee.