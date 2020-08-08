import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import time


class MNISTClassifier():

    def __init__(self, X_train, X_test, y_train, y_test, features_size):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.features_size = features_size
        self.params = self.param_init()

    def sigmoid(self, z, backward=False):
        if not backward:
            return 1. / (1. + np.exp(-z))
        return (np.exp(-z)) / ( 1. + (np.exp(-z) ) ** 2 )

    def relu(self, z, backward=False):
        if not backward:
            z[z<=0] = 0
        else:
            z[z<=0] = 0
            z[z>0] = 1
        return z 

    def softmax(self, z):
        # Numerically stable with large exponentials
        exps = np.exp(z - z.max())
        return exps / np.sum(exps, axis=0)
    def compute_loss(self, y, y_hat):
        """
        compute cross-entropy loss function
        """
        L_sum = np.sum(np.multiply(y, np.log(y_hat)))
        m = y.shape[1]
        L = -(1./m) * L_sum

        return L

    def param_init(self):
        # number of nodes in each layer
        input_layer = self.features_size[0]
        layer_1 = self.features_size[1]
        layer_2 = self.features_size[2]
        output_layer = self.features_size[3]

        params = {
            'W1': np.random.randn(layer_1, input_layer) * np.sqrt(1. / layer_1),
            'W2': np.random.randn(layer_2, layer_1) * np.sqrt(1. / layer_2),
            'W3': np.random.randn(output_layer, layer_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.relu(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.relu(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def backward_pass(self, y, output):
        
        params = self.params
        delta_params = {}

        # Calculate W3 update
        error = output - y
        delta_params['W3'] = np.dot(error, params['A3']) 

        # Calculate W2 update
        error = np.multiply( np.dot(params['W3'].T, error), self.relu(params['Z2'], backward=True) )
        delta_params['W2'] = np.dot(error, params['A2'])

        # Calculate W1 update
        error = np.multiply( np.dot(params['W2'].T, error), self.relu(params['Z1'], backward=True) )
        delta_params['W1'] = np.dot(error, params['A1'])

        return delta_params

    def update_network_parameters(self, delta_params):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y), 
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''
        learning_rate = 0.0001
        for key, value in delta_params.items():
            for w_arr in self.params[key]:
                w_arr -= learning_rate * value

    def compute_accuracy(self, X_test, y_test):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []
        outputs = []
        y_hat = []
        for x, y in zip(X_test, y_test):
            output = self.forward_pass(x)
            outputs.append(output)
            pred = np.argmax(output)
            y_hat = np.argmax(y)
            predictions.append(pred==y_hat)
        # pred_class = np.array((predictions == y_hat))
        loss = self.compute_loss(y_test, outputs)
        summed = sum(pred for pred in predictions) / 100.0
        return np.average(summed), loss

    def train(self, epochs=10):
        start_time = time.time()
        for iteration in range(epochs):
            for x,y in zip(self.X_train, self.y_train):
                output = self.forward_pass(x)
                delta_w = self.backward_pass(y, output)
                self.update_network_parameters(delta_w)
            train_accuracy, _ = self.compute_accuracy(self.X_train, self.y_train) 
            test_accuracy, _ = self.compute_accuracy(self.X_test, self.y_test)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Train Accuracy: {2}  Test Accuracy: {3}'.format(
                iteration+1, time.time() - start_time, train_accuracy, test_accuracy 
            ))

## Fetch MNIST
X, y = fetch_openml('mnist_784', return_X_y=True)
## Normalize the data and one-hot encoding of labels
X = (X / 255).astype('float32')
y = to_categorical(y, num_classes=10)

## Define sample sizes 
total_samples_size = X.shape[0]
# training_samples_size = 60000
# test_samples_size = total_samples_size - training_samples_size
test_size = 0.33
# split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
# Initialize the classfier model
mnist_classifier = MNISTClassifier(X_train, X_test, y_train, y_test, features_size=[784, 128, 64, 10])
# Train the classifier
mnist_classifier.train(epochs=100)
