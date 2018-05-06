import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage
import network
import pickle

# import user image
user_array = scipy.ndimage.imread('sample.jpg', flatten=True, mode=None)
user_array = (-user_array/255)+1
shape = np.shape(user_array)

# plot imported image
imgplot = plt.imshow(user_array, cmap = 'Greys')
plt.show()

# turns imported image (array) into a vector
user_vector = np.zeros( (784, 1) )
for i in range(shape[1]):
	for j in range(shape[0]):
		value = user_array[ i, j ]	
		index = i*shape[1] + j
		user_vector[ index ] = value

# chooses the network shape to use
net = network.Network( [ 784, 150, 50, 100, 10 ] )
	
# file name of the file containing best weights and biases for that shape
file_name = str(net.sizes).replace("[","").replace("]","").replace(" ","").replace(",","_")+'.pkl'
print(file_name)

digit2word = {
	0 : 'ZERO',
	1 : 'ONE',
	2 : 'TWO',
	3 : 'THREE',
	4 : 'FOUR',
	5 : 'FIVE',
	6 : 'SIX',
	7 : 'SEVEN',
	8 : 'EIGHT',
	9 : 'NINE'
}

# loads network and feeds vectorized image through the network
try:
	with open('best_networks/{0}'.format(file_name), 'rb') as f:
		net_properties = pickle.load(f)

	net.weights = net_properties[1]
	net.biases  = net_properties[2]
	result = np.argmax(net.feedforward(user_vector))

	print('my guess is: {0}! '.format(digit2word[result]))
	indices = [0,1,2,3,4,5,6,7,8,9]

	print('confidence:')
	for point,indice in zip(net.feedforward(user_vector),indices):
		print ("{0}".format(indice)+":{:10.4f}".format(float(point)))

# brings up error if no weights/biases exist for the desired shape 
except FileNotFoundError:
	print('{0} shape has not been trained yet'.format(net.sizes))