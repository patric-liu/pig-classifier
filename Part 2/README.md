# Pig Classification 

I chose to use a convolutional neural network implemented through tensorflow to train on the images.  Because the photos were hardly distinguishable, I found that it was necessary to use external data. My idea was to make a classifier that would determine if a picture was of a pig in an cage, or of an empty cage. 

### implementation 
data
I found 15 images of empty cages online, and randomly chose 15 of the provided pig photos and scaled them all down to a size (148 x 200 px) that my computer could handle. 

### network
The network contains two convolution layers with max-pooling, then a fully connected layer before the output. I chose two convolution layers in order to extract higher level features while still keeping the network size small. 

### training
Training was done with tensorflow’s AdamOptimizer with a fixed learning rate and with dropout to improve regularization

### results
The network quickly converges to an optimum due to the simplicity of the problem, arising from the limited and easy-to-learn data. 

## improvements 
data
If we really wanted a neural net that could tell whether or not a cage had a pig in it, a lot more data would be needed, and ideally data with plenty of variation (lighting, angle, type of pig etc). Using more than 15 of the provided pig photos wouldn’t make much of a difference for this application. 

### network
The network structure used should suffice for this simple binary classification problem. A deeper network would need to be employed for problems with more complexity and image classes. 

### training
If the dataset were larger and had more variety, a training schedule would be useful for maintaining a balance between training speed and network accuracy. 

### results
An evaluation dataset that the network was not trained on would be ideal
