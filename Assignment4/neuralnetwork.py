from sklearn import preprocessing
import math
import random
import numpy as np
import struct
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random.uniform(-0.1,0.1) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random.uniform(-0.1,0.1) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	activation = np.clip(activation, -650, 650)
	return 1.0 / (1.0 + math.exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
			#original		
			#neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate, m_batch_size):
	for i in range(len(network)):
		#inputs = row[:-1] #original
		inputs = row
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				#neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j] # original
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j] * 1.0/m_batch_size
			neuron['weights'][-1] += l_rate * neuron['delta'] * 1.0/m_batch_size

# Train a network for a fixed number of epochs
def train_network(network, train_data, train_labels, test_data, test_labels, n_epoch, l_rate, momentum, m_batch_size):
	for epoch in range(n_epoch):
		for i in range(len(train_data)):
			row = train_data[i]
			label = train_labels[i]
			outputs = forward_propagate(network, row)
			expected = [1.0 if x == label else 0.0 for x in range(len(outputs))]
			backward_propagate_error(network, expected)
			if i % m_batch_size == 0 :
				update_weights(network, row, l_rate, m_batch_size)
			if i == len(train_data)/2 - 1 or i == len(train_data) - 1:
				mean_sqr_err, error_rate = compute_metrics(network, test_data, test_labels)
				print('>epoch=%d, lrate=%.5f, error=%.5f, error_rate=%.5f' % (epoch, l_rate, mean_sqr_err, error_rate))

def compute_metrics(network, examples, target):
	mean = 0.0
	negative = 0
	for i in range(len(examples)):
		example = examples[i]
		expected = target[i]
		outputs = forward_propagate(network, example)
		predicted = outputs.index(max(outputs))
		mean += sum([(1-outputs[x])**2 if x == expected else (outputs[x])**2 for x in range(len(outputs))])
		if expected != predicted:
			negative += 1
	return mean / (2.0 * len(examples)), negative * 100.0/len(examples)

def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

def test_backpropogation():
	# Test training backprop algorithm
	dataset = [[1,0,0,0,0,0,0,0],
		[0,1,0,0,0,0,0,0],
		[0,0,1,0,0,0,0,0],
		[0,0,0,1,0,0,0,0],
		[0,0,0,0,1,0,0,0],
		[0,0,0,0,0,1,0,0],
		[0,0,0,0,0,0,1,0],
		[0,0,0,0,0,0,0,1]]
	labels = [0,1,2,3,4,5,6,7]
	n_inputs = len(dataset[0])
	n_outputs = len(dataset[0])
	network = initialize_network(n_inputs, 3, n_outputs)
	train_network(network, dataset, labels, 0.1, 30, n_outputs)
	for row in dataset:
		prediction = predict(network, row)
		print('Expected=%d, Got=%d' % (row[-1], prediction))


np.seterr(all='ignore')
trainLabelsFile = 'train-labels.idx1-ubyte'
with open(trainLabelsFile, 'rb') as labels:
        _, _ = struct.unpack(">II", labels.read(8))
        train_labels = np.fromfile(labels, dtype=np.uint8)
trainImagesFile = 'train-images-pca.idx2-double'    
with open(trainImagesFile, 'rb') as images:
    _, _, dims = struct.unpack(">III", images.read(12))
    train_images = np.fromfile(images, dtype=np.dtype('float64').newbyteorder('>')).reshape(len(train_labels), dims)
train_images = preprocessing.scale(train_images) # scaling the training images data

testLabelsFile = 't10k-labels.idx1-ubyte'    
with open(testLabelsFile, 'rb') as labels:
    _, _ = struct.unpack(">II", labels.read(8))
    test_labels = np.fromfile(labels, dtype=np.uint8)
testImagesFile = 't10k-images-pca.idx2-double'
with open(testImagesFile, 'rb') as images:
    _, _, dims = struct.unpack(">III", images.read(12))
    test_images = np.fromfile(images, dtype=np.dtype('float64').newbyteorder('>')).reshape(len(test_labels), dims)
test_images = preprocessing.scale(test_images) # scaling the testing images data

l_rate = 0.1
momentum = 0.0
mini_batch_size = 1
epochs = 10
n_inputs = len(train_images[0])
n_outputs = 10
n_hidden = 10
network = initialize_network(n_inputs, n_hidden, n_outputs)
train_network(network, train_images, train_labels, test_images, test_labels, epochs, l_rate, momentum, mini_batch_size)
