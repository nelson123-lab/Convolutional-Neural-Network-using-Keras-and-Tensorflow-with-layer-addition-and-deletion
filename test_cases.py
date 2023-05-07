# Joseph, Nelson
# 1002_050_500
# 2023_04_17
# Assignment_04_02

import pytest
import numpy as np
import tensorflow as tf
from Joseph_04_01 import CNN

def test_training_1():
    tf.keras.utils.set_random_seed(100)
    # Loading mnist data.
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalizing the data.
    X_train = (X_train / 255.0 - 0.5).astype(np.float32)
    X_test = (X_test / 255.0 - 0.5).astype(np.float32)
    # Selecting the first 100 samples for training.
    X_train, y_train = X_train[0:100, :], y_train[0:100]

    # CNN Architecture taken from Jason test cases.
    my_cnn=CNN()
    my_cnn.set_loss_function('SparseCategoricalCrossentropy')
    my_cnn.set_optimizer('SGD')
    my_cnn.set_metric('accuracy')
    my_cnn.add_input_layer(shape=(28,28,1),name = "input")
    my_cnn.append_conv2d_layer(num_of_filters = 16, kernel_size = 3,padding = "same", activation = 'linear', name="conv1")
    my_cnn.append_conv2d_layer(num_of_filters = 8, kernel_size = 3, activation = 'relu', name = "conv2")
    my_cnn.append_maxpooling2d_layer(pool_size = 2, padding = "same", strides = 2,name = "pool1")
    my_cnn.append_flatten_layer(name = "flat1")
    my_cnn.append_dense_layer(num_nodes = 10,activation = "relu",name = "dense1")
    loss = my_cnn.train(X_train, y_train, batch_size = 32, num_epochs = 10)
    assert np.allclose(loss[0],11.136558532714844)
    assert np.allclose(loss[2],11.9644136428833)
    assert np.allclose(loss[5],11.964411735534668)

def test_training_2():
    tf.keras.utils.set_random_seed(100)
    # Loading mnist data.
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalizing the data.
    X_train = (X_train / 255.0 - 0.5).astype(np.float32)
    X_test = (X_test / 255.0 - 0.5).astype(np.float32)
    # Selecting the first 100 samples for training.
    X_train, y_train = X_train[0:100, :], y_train[0:100]

    # CNN Architecture taken from Jason test cases.
    my_cnn=CNN()
    my_cnn.set_loss_function('hinge')
    my_cnn.set_optimizer('SGD')
    my_cnn.set_metric('accuracy')
    my_cnn.add_input_layer(shape = (28,28,1),name = "input")
    my_cnn.append_conv2d_layer(num_of_filters = 16, kernel_size = 3,padding = "same", activation = 'linear', name = "conv1")
    my_cnn.append_conv2d_layer(num_of_filters = 8, kernel_size = 3, activation ='relu', name = "conv2")
    my_cnn.append_maxpooling2d_layer(pool_size = 2, padding = "same", strides=2,name="pool1")
    my_cnn.append_flatten_layer(name = "flat1")
    my_cnn.append_dense_layer(num_nodes = 10,activation = "relu",name = "dense1")
    loss = my_cnn.train(X_train, y_train, batch_size = 32, num_epochs = 10)
    assert np.allclose(loss[1],0.5993309020996094)
    assert np.allclose(loss[3],0.3366689682006836)
    assert np.allclose(loss[5],0.29685863852500916)

def test_evaluate_1():
    tf.keras.utils.set_random_seed(100)
    # Loading mnist data.
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalizing the data.
    X_train = (X_train / 255.0 - 0.5).astype(np.float32)
    X_test = (X_test / 255.0 - 0.5).astype(np.float32)
    # Selecting the first 100 samples for training.
    X_train, y_train = X_train[0:100, :], y_train[0:100]
    X_test, y_test = X_test[0:100, :], y_test[0:100]

    # # CNN Architecture taken from Jason test cases.
    my_cnn = CNN()
    my_cnn.set_loss_function('mse')
    my_cnn.set_optimizer('SGD')
    my_cnn.set_metric('accuracy')
    my_cnn.add_input_layer(shape = (28,28,1), name = "input")
    my_cnn.append_conv2d_layer(num_of_filters = 16, kernel_size = 3,padding = "same", activation = 'linear', name = "conv1")
    my_cnn.append_conv2d_layer(num_of_filters = 8, kernel_size = 3, activation ='relu', name = "conv2")
    my_cnn.append_maxpooling2d_layer(pool_size = 2, padding = "same", strides=2,name="pool1")
    my_cnn.append_flatten_layer(name = "flat1")
    my_cnn.append_dense_layer(num_nodes = 10, activation = "relu", name = "dense1")
    loss = my_cnn.train(X_train, y_train, batch_size = 32, num_epochs = 10)
    eval_loss, metric = my_cnn.evaluate(X_test, y_test)
    assert np.allclose(eval_loss, 14.075966835021973)
    assert np.allclose(metric, 0.09000000357627869)

def test_evaluate_2():
    # Setting seed value.
    tf.keras.utils.set_random_seed(500)
    # Importing and preparing the dataset.
    from tensorflow.keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, y_train = X_train[0:100, :], y_train[0:100]
    X_test, y_test = X_test[0:100, :], y_test[0:100]

    # CNN Architecture taken from Jason test cases.
    my_cnn=CNN()
    my_cnn.set_loss_function('hinge')
    my_cnn.set_optimizer('Adagrad')
    my_cnn.set_metric('accuracy')
    my_cnn.add_input_layer(shape = (32,32,3), name = "input")
    my_cnn.append_conv2d_layer(num_of_filters = 16, kernel_size = 3,padding = "same", activation = 'linear', name = "conv1")
    my_cnn.append_maxpooling2d_layer(pool_size = 2, padding = "same", strides = 2,name = "pool1")
    my_cnn.append_conv2d_layer(num_of_filters = 8, kernel_size = 3, activation = 'relu', name = "conv2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes = 10,activation = "relu",name = "dense1")
    my_cnn.append_dense_layer(num_nodes = 2,activation = "relu",name = "dense2")
    loss = my_cnn.train(X_train, y_train, batch_size = 5, num_epochs = 20)
    eval_loss, metric = my_cnn.evaluate(X_test, y_test)
    assert np.allclose(eval_loss, 0.9800000190734863)
