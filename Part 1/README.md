# MNIST Trainer
I decided to train on the MNIST dataset because it’s a well understood dataset which can be quickly trained upon. This way, rapid iterations could be made on the model to improve training. The MNIST dataset consists of 60,000 images of B&W handwritten digits all scaled to 28x28 pixels. 

##the code
I implemented a basic deep perceptron network without the use of a deep learning library. Just a note, all this code was written a few months ago as a way to teach myself deep learning, and took me about two weeks.  All the functionality needed was written into network.py.  trainer.py loads training data, specifies the network shape and training hyperparameters, and trains the network. Once a network is trained, it can be loaded using front_end.py which takes and classifies a custom digit image called sample.jpg. 

##training results
The best performing network achieved 98.3% accuracy on an evaluation dataset of 10,000 images, which is quite high for a model which doesn’t use convolution. Its layers had the shape [input: 784, 150, 50, 100, output: 10]. It employed a learning schedule which halved the learning rate for every 3 epochs of no improvement and terminated training after 20 epochs of no improvement. 

##hyperparameters
In order to arrive upon a good set of hyper-parameters such as learning rate and amount of L2 regularization,  I first found good hyper parameters using less data on a much smaller network [784,30,10] - this allowed me to rapidly test the hyper-parameter space for values which worked. These values were then fine tuned on the full model [784, 150, 50, 100, 10]. The final shape of the network was chosen based off successful attempts from literature, which seemed to have a smaller layer in the middle. 

##discussion
Normally, convolution is used for image classification, but a perceptron is sufficient for MNIST because of its low dimensionality and that all the images are centered and resized quite uniformly. Rarely would this kind of data be available/relevant in typical applications of image classification. In cases such as pig classification, feature extraction becomes more valuable and I would use a convolutional neural network for that. 

##data
The MNIST data was prepared by [mnielsen (Michael Nielsen) · GitHub](https://github.com/mnielsen).  I downloaded mnist_pkl.gz and mnist_loader.py, which allows me to access the data as a list tuples of input-output pairs. 
