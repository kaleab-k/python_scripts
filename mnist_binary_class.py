#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:24:09 2019

@author: Kinfu Kaleab
"""

import numpy as np

from keras.datasets import mnist

from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
metrics = {"training_loss": [], "test_loss": [], "training_acc": [], "test_acc": []}

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return x*(1-x)

def feed_forward(X,W,b):
     Z = np.matmul(X,W) + b
     A = sigmoid(Z)
     return A
 
def back_prop(l_prev, l, y):
    err = l - y
    dl = np.multiply(err, sigmoid_der(l))
    dw = np.matmul(l_prev.T, dl)
    db = np.mean(err)
    return (dw,db)
 
def compute_loss(y, y_pred):
    mean_sum_loss = np.mean(y*np.log(1e-15 + y_pred))
    return -mean_sum_loss

def record_metrics(y, y_pred, training=False):
    loss = compute_loss(y, np.around(y_pred))
    acc = np.mean(y == np.around(y_pred))
    if training:
        metrics["training_loss"].append(loss)
        metrics["training_acc"].append(acc)
    else:
        metrics["test_loss"].append(loss)
        metrics["test_acc"].append(acc)
    return loss, acc


num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train  = X_train / 255
X_test  = X_test / 255


#We want to have a binary classification: digit 0 is classified 1 and 
#all the other digits are classified 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new

m = X_train.shape[1] #number of examples

#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[shuffle_index,:], y_train[shuffle_index,:]

# initialize weights with normal random distribution having 0 as its mean
W = 2*np.random.random((X_train.shape[1], 1)) - 1

b = np.random.rand(1)

alpha = 0.015  # the learning rate

epochs = 500

for i in range(epochs):
    # forward propagation
    l0 = X_train  # layer 0, i.e. the input
    l1 = feed_forward(l0, W, b)  # output of the single perceptron

    (dw, db) = back_prop(l0, l1, y_train) # back propagation to get change of weight and bias

    # update weights and bias
    W -= dw * alpha
    b -= db

    if i % 1 == 0:
        # Recording metrics for training and testing
        train_loss, train_acc = record_metrics(y_train, l1, True)

        y_pred = feed_forward(X_test, W, b)
        test_loss, test_acc = record_metrics(y_test, y_pred)

        print('Epoch: {}, Training Loss: {}, Training Acc: {}'.format(i, train_loss, train_acc))

## Printing training and testing metrics

test_pred = feed_forward(X_test, W, b)
train_pred =  feed_forward(X_train, W, b)

print("Training Loss: {}".format(compute_loss(y_train, np.around(train_pred))))
print("Training Acc: {}".format(np.mean(y_train == np.around(train_pred))))

print("Test Loss: {}".format(compute_loss(y_test, np.around(test_pred))))
print("Test Acc: {}".format(np.mean(y_test == np.around(test_pred))))

# Plotting
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(metrics['training_loss'], label="Training Loss")
plt.plot(metrics['test_loss'], label="Testing Loss")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(metrics['training_acc'], label="Training Accuracy")
plt.plot(metrics['test_acc'], label="Testing Accuracy")
plt.legend()

plt.show()
