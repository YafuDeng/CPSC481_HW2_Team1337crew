import random
import numpy as np


class MSE:  # use the mean squared error as your loss function

    def __init__(self):
        pass

    @staticmethod
    def loss_function(predictions, labels):
        diff = predictions - labels
        return 0.5 * sum(diff * diff)[0]  # define MSE as 0.5 times the square difference between predictions and labels

    @staticmethod
    def loss_derivative(predictions, labels):
        return predictions - labels  # the loss derivative is simply predictions-labels

class SequentialNetwork:  # In a sequential neural network we stack layers sequentially.
    def __init__(self, loss=None):
        print("Initialize Network...")
        self.layers = []
        if loss is None:
            self.loss = MSE()  # If no loss function is provided, MSE is used.

    # Whenever we add a layer, we connect it to its predecessor and let it describe itself
    def add(self, layer):
        self.layers.append(layer)
        layer.describe()
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])

    def train(self, training_data, epochs, mini_batch_size,
              learning_rate, test_data=None):
        n = len(training_data)
        for epoch in range(epochs):
            # To train our network, we pass over data for as many times as there are epochs.
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for
                k in range(0, n, mini_batch_size)  # we shuffle training data and create mini-batches.
            ]
            for mini_batch in mini_batches:
                self.train_batch(mini_batch, learning_rate)  # For each mini-batch we train our network.
            if test_data:
                n_test = len(test_data)
                # in case we provided test data, we evaluate our network on it after each epoch.
                print("Epoch {0}: {1} / {2}"
                      .format(epoch, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(epoch))

    def train_batch(self, mini_batch, learning_rate):
        self.forward_backward(mini_batch)
        # To train the network on a mini-batch, we compute feed-forward and backward pass...
        self.update(mini_batch, learning_rate)  # and then update model parameters accordingly.

    def update(self, mini_batch, learning_rate):
        # a common technique is to normalize the learning rate by the mini-batch size.
        learning_rate = learning_rate / len(mini_batch)

        for layer in self.layers:
            # <2> We then update parameters for all layers.
            layer.update_params(learning_rate)
        for layer in self.layers:
            # Afterwards we clear all deltas in each layer.
            layer.clear_deltas()

    def forward_backward(self, mini_batch):
        for x, y in mini_batch:
            self.layers[0].input_data = x
            for layer in self.layers:
                layer.forward()   # For each sample in the mini batch, feed the features forward layer by layer.
            self.layers[-1].input_delta = \
                self.loss.loss_derivative(self.layers[-1].output_data, y)   # compute the loss derivative
            for layer in reversed(self.layers):
                layer.backward()  # Finally, we do layer-by-layer backpropagation of error terms.

    def single_forward(self, x):  # Pass a single sample forward and return the result.
        self.layers[0].input_data = x
        for layer in self.layers:
            layer.forward()
        return self.layers[-1].output_data

    def evaluate(self, test_data):  # Compute accuracy on test data.
        test_results = [(
            np.argmax(self.single_forward(x)),
            np.argmax(y)
        ) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
