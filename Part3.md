# Auto Encoder
 An auto encoder is deep learning-based compression method that is data-specific and lossy. Two neural networks, an encoder and decoder, work in conjunction to reconstruct an image after it is passed through a low dimensional filter. The input and output pairs would be identical. 

The idea is that important features could be extracted from a piece of data such as an image. For example, the pig’s x-y position, body angle and leg positions. Unchanging features such as the cage will generally not be encoded, because the information to reconstruct those parts of the image would be contained within the weights of the decoder. The activation of each encoded neuron would be a numerical representation of some feature. 

This type of unsupervised learning would be a good fit for the data we have, since it is unlabeled and has very simple and minute variations.  In this situation, only a very small number of encoding neurons would be sufficient because of the lack of variability between photos. 
