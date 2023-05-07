# Joseph, Nelson
# 1002_050_500
# 2023_04_17
# Assignment_04_01


import tensorflow as tf
import numpy as np
import tensorflow.keras as keras


class CNN(object):


    def __init__(self):
        """
        Initialize multi-layer neural network
        """
        # Initializing the metric for training
        self.metric = []
        # Initializing the keras sequential model.
        self.model = keras.models.Sequential()


    def add_input_layer(self, shape=(2,), name=""):
        """
         This function adds an input layer to the neural network. If an input layer exist, then this function
         should replcae it with the new input layer.
         :param shape: input shape (tuple)
         :param name: Layer name (string)
         :return: None
         """
        # IF the input layer is already present, then removing it new layer.
        if len(self.model.layers) > 0:
            self.new_model = keras.models.Sequential()
            # Making the old model new model.
            self.model = self.new_model

        # Adding the input layer to the sequential class.
        self.model.add(keras.layers.InputLayer(input_shape=shape, name=name))
        return None

    def append_dense_layer(self, num_nodes, activation="relu", name="", trainable=True):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: Number of nodes
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid",
         "Softmax"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: None
         """
        # Adding dense layer to Sequential class.
        self.model.add(keras.layers.Dense(num_nodes, activation = activation, name = name, trainable = trainable))
        return None

    def append_conv2d_layer(self, num_of_filters, kernel_size = 3, padding = 'same', strides = 1,
                            activation = "relu", name = "", trainable = True):
        """
         This function adds a conv2d layer to the neural network
         :param num_of_filters: Number of nodes
         :param num_nodes: Number of nodes
         :param kernel_size: Kernel size (assume that the kernel has the same horizontal and vertical size)
         :param padding: "same", "Valid"
         :param strides: strides
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: Layer object
         """
        # Making the Conv2d layer
        Conv2d_layer = keras.layers.Conv2D(filters = num_of_filters, kernel_size = kernel_size, padding = padding, strides = strides,
                                    activation = activation, name = name, trainable = trainable)
        # Adding the Conv2d layer to the Sequential model.
        self.model.add(Conv2d_layer)
        return Conv2d_layer

    def append_maxpooling2d_layer(self, pool_size = 2, padding = "same", strides = 2, name = ""):
        """
         This function adds a maxpool2d layer to the neural network
         :param pool_size: Pool size (assume that the pool has the same horizontal and vertical size)
         :param padding: "same", "valid"
         :param strides: strides
         :param name: Layer name (string)
         :return: Layer object
         """
        # Making the MaxPoolint layer
        MaxPooling2D_layer = keras.layers.MaxPooling2D(pool_size = pool_size, padding = padding, strides = strides, name = name)
        # Adding the Conv2d layer to the Sequential model.
        self.model.add(MaxPooling2D_layer)
        # Returing MaxPooling layer
        return MaxPooling2D_layer

    def append_flatten_layer(self, name = ""):
        """
         This function adds a flattening layer to the neural network
         :param name: Layer name (string)
         :return: Layer object
         """
        # Making the Flatten layer.
        Flatten_layer = keras.layers.Flatten(name = name)
        # Adding Flatten layer to the Sequential model.
        self.model.add(Flatten_layer)
        # Returning the Flatten layer.
        return Flatten_layer

    def set_training_flag(self, layer_numbers = [], layer_names = "", trainable_flag = True):
        """
        This function sets the trainable flag for a given layer
        :param layer_numbers: an integer or a list of numbers.Layer numbers start from layer 0.
        :param layer_names: a string or a list of strings (if both layer_number and layer_name are specified, layer number takes precedence).
        :param trainable_flag: Set trainable flag
        :return: None
        """
        # Checking if layer_numbers is not a list
        if type(layer_numbers) != list:
            # Iterating over all layers.
            for l in layer_numbers:
                # setting trainable flag for a given layer.
                self.model.get_layer(layer_number = l, layer_name=layer_names[l]).trainable = trainable_flag
        # setting trainable flag for a given layers.
        else: self.model.get_layer(layer_number = layer_numbers, layer_name = layer_names).trainable = trainable_flag
        # Returning None.
        return None

    def get_weights_without_biases(self, layer_number = None, layer_name=""):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: Weight matrix for the given layer (not including the biases). If the given layer does not have
          weights then None should be returned.
        """
        # Initializing the variable 1 = 0
        l = 0
        # Checking if layer_number = None.
        if layer_number == None:
            # Return None if if the layer with the specified layer_name has no weights.
            if len(self.model.get_layer(name = layer_name).get_weights()) <= 0: return None
            # Otherwise return the weight matrix without biases.
            else: return self.model.get_layer(name = layer_name).get_weights()[0]
        # layer_number not None
        else:
            # Returning None if the layer with the specified layer_number - 1 has no weights or if layer_number is 0.
            if len(self.model.layers[layer_number - 1].get_weights()) <= 0 or layer_number == 0: return None
            # Returning the weight matrix of the layer without biases when layer_number == l-1
            elif layer_number == (l-1): return self.model.layers[layer_number].get_weights()[0]
            # Return the weight matrix of the layer without biases otherwise.
            else: return self.model.layers[layer_number - 1].get_weights()[0]

    def get_biases(self, layer_number = None, layer_name = ""):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: biases for the given layer (If the given layer does not have bias then None should be returned)
         """
        # Checking if layer_number is None
        if layer_number is None:  
            # Storing the weights of the layer with the specified layer_name.
            weights = self.model.get_layer(name=layer_name).get_weights()
            # Returing None if the layer does not have weights.
            if len(weights) <= 0: return None  
            # Returning the biases of the layer using index [1] to get the biases otherwise.
            else: return weights[1]  
        else:
            # Checking if the layer with the specified layer_number - 1 has weights or if layer_number is 0 and returing None if not.
            if len(self.model.get_layer(index=layer_number - 1).get_weights()) <= 0 or layer_number == 0: return None 
            # Returning the biases of the layer using index [1] to get the biases If layer_number is -1,
            elif layer_number == -1: return self.model.layers[layer_number].get_weights()[1]  
            # Returning the biases of the layer using index [1] to get the biases otherwise.
            else: return self.model.layers[layer_number - 1].get_weights()[1]  


    def set_weights_without_biases(self, weights, layer_number=None, layer_name=""):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: None
         """
        # Returing None as the first layer does not have weights when layer_number = 0.
        if layer_number == 0:   return None  
        # If layer_number is None, set the weight matrix for the layer with the specified layer_name using the set_value() method
        # Weight matrix is stored at index 0 of the 'weights' property of the layer, so we access it with 'weights[0]'
        elif layer_number is None:  keras.backend.set_value(self.model.get_layer(name = layer_name).weights[0], weights)
        # Setting the weight matrix for the layer with the specified layer_number - 1 using the set_value() method otherwise.
        else:   keras.backend.set_value(self.model.get_layer(index = layer_number - 1).weights[0], weights)

    def set_biases(self, biases, layer_number=None, layer_name=""):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
        :return: none
        """
        # If layer_number is not None, set the biases for the layer with the specified layer_number - 1 using the set_value() method
        # Biases are stored at index 1 of the 'weights' property of the layer, so we access it with 'weights[1]'
        if layer_number is not None:    keras.backend.set_value(self.model.get_layer(index=layer_number - 1).weights[1], biases)
        # Setting the biases for the layer with the specified layer_name using the set_value() method otherwise.
        else:   keras.backend.set_value(self.model.get_layer(name=layer_name).weights[1], biases)
    
    def remove_last_layer(self):
        """
        This function removes a layer from the model.
        :return: removed layer
        """
        # Popping the last layer from the model and storing it in a variable.
        sliced_model = self.model.pop()
        # Reconstructing the model with the remaining layers using keras.Sequential
        self.model = keras.Sequential(self.model.layers)
        # Returning the removed layer as sliced_model.
        return sliced_model


    def load_a_model(self, model_name="", model_file_name=""):
        """
        This function loads a model architecture and weights.
        :param model_name: Name of the model to load. model_name should be one of the following:
        "VGG16", "VGG19"
        :param model_file_name: Name of the file to load the model (if both madel_name and
         model_file_name are specified, model_name takes precedence).
        :return: model
        """

        # Creating the new sequential model.
        self.model = keras.models.Sequential()
        # Loading the specified model architecture and weights based on model_name or model_file_name
        if model_name == "VGG16":   base_model = keras.applications.vgg16.VGG16()
        elif model_name == "VGG19": base_model = keras.applications.vgg19.VGG19()
        else:   base_model = keras.models.load_model(model_file_name)
        # Adding each layer from the loaded model to self.model by iterating through the base_model.layers.
        for layer in base_model.layers: self.model.add(layer)

    def save_model(self, model_file_name = ""):
        """
        This function saves the current model architecture and weights together in a HDF5 file.
        :param file_name: Name of file to save the model.
        :return: model
        """
        # Saving the model architecture and weights
        self.model.save(model_file_name)

    def set_loss_function(self, loss = "SparseCategoricalCrossentropy"):
        """
        This function sets the loss function.
        :param loss: loss is a string with the following choices:
        "SparseCategoricalCrossentropy",  "MeanSquaredError", "hinge".
        :return: none
        """
        # Setting the loss function which can be "SparseCategoricalCrossentropy",  "MeanSquaredError", "hinge".
        self.loss = loss

    def set_metric(self, metric):
        """
        This function sets the metric.
        :param metric: metric should be one of the following strings:
        "accuracy", "mse".
        :return: none
        """
        # Setting the metric which can be "accuracy", "mse".
        self.metric = metric

    def set_optimizer(self, optimizer = "SGD", learning_rate = 0.01, momentum = 0.0):
        """
        This function sets the optimizer.
        :param optimizer: Should be one of the following:
        "SGD" , "RMSprop" , "Adagrad" ,
        :param learning_rate: Learning rate
        :param momentum: Momentum
        :return: none
        """
        # Setting the optimizer based on the specified optimizer name as given in the function documentation.
        if optimizer == "SGD":  self.optimizer = keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum)
        elif optimizer == "RMSprop":    self.optimizer = keras.optimizers.RMSprop(learning_rate = learning_rate)
        elif optimizer == "Adagrad":    self.optimizer = keras.optimizers.Adagrad(learning_rate = learning_rate)
        else:   pass

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Input tensor.
        :return: Output tensor.
        """
        # Returning the output.
        return self.model.predict(X.astype('float32'))

    def evaluate(self, X, y):
        """
         Given array of inputs and desired ouputs, this function returns the loss value and metrics of the model.
         :param X: Array of input
         :param y: Array of desired (target) outputs
         :return: loss value and metric value
         """
        # Evaluating the model performance.
        return self.model.evaluate(x = X, y = y)

    def train(self, X_train, y_train, batch_size, num_epochs):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X_train: Array of input
         :param y_train: Array of desired (target) outputs
         :param X_validation: Array of input validation data
         :param y: Array of desired (target) validation outputs
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :return: list of loss values. Each element of the list should be the value of loss after each epoch.
         """
        # Compiling the model with the specified optimizer, loss function, and metric sent during the testing phase.
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metric)
        # Training the model using the provided input data and hyperparameters that are being sent during the testing phase.
        # Storing the training history in a variable output.
        output = self.model.fit(x = X_train, y = y_train, batch_size = batch_size, epochs = num_epochs, verbose = 2, shuffle=True)
        # Returning the training history's loss values.
        return output.history['loss']
