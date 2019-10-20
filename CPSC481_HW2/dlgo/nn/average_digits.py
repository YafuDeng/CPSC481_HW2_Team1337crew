import numpy as np
from dlgo.nn.load_mnist import load_data
from dlgo.nn.layers import sigmoid_double


def average_digit(data, digit):  # compute the average over all samples in your data representing a given digit
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)


train, test = load_data()
avg_eight = average_digit(train, 8)


from matplotlib import pyplot as plt

img = (np.reshape(avg_eight, (28, 28)))
plt.imshow(img)
plt.show()

x_3 = train[2][0]    # training sample at index 2 is a 4
x_18 = train[17][0]  # training sample at index 17 is an 8

W = np.transpose(avg_eight)
np.dot(W, x_3)   # this evaluates to about 20.1
np.dot(W, x_18)  # this term is much bigger, about 54.2


def sigmoid_double(x):
    return 1.0/(1.0+np.exp(-x))


def sigmoid(z):
    return np.vectorize(sigmoid_double(z))


def predict(x, W, b):  # prediction by applying sigmoid to the output of np.doc(W, x)+b
    return sigmoid_double(np.dot(W, x) + b)


b = -45  # set the bias term to -45

print(predict(x_3, W, b))   # prediction for the example with a 4 is close to 0
print(predict(x_18, W, b))  # prediction for an 8 is 0.96, seems to be onto something with the heuristic


def evaluate(data, digit, threshold, W, b):  # choose accuracy, the radio of correct predictions among all
    total_samples = 1.0 * len(data)
    correct_predictions = 0
    for x in data:
        if predict(x[0], W, b) > threshold and np.argmax(x[1]) == digit:  # predicting an 8 as 8 is a correct prediction
            correct_predictions += 1
        if predict(x[0], W, b) <= threshold and np.argmax(x[1]) != digit:  # if below threshold not an 8, also correct
            correct_predictions += 1
    return correct_predictions / total_samples


evaluate(data=train, digit=8, threshold=0.5, W=W, b=b)  # accuracy 78%

evaluate(data=test, digit=8, threshold=0.5, W=W, b=b)   # 77%

eight_test = [x for x in test if np.argmax(x[1]) == 8]
# evaluating only on the set of 8s in the test set results in only 67% accuracy
evaluate(data=eight_test, digit=8, threshold=0.5, W=W, b=b)
