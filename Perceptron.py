#  Matthew Buchanan
#  Perceptron Project
#  Intro to Neural Networks
#  Fall 2023


class Perceptron(object):

    # Create a new Perceptron
    #
    # Params:	bias - arbitrarily chosen value that affects the overall output
    #			regardless of the inputs
    #
    #			synaptic_weights -	list of initial synaptic weights for this Perceptron
    def __init__(self, bias, synaptic_weights):
        self.bias = bias
        self.synaptic_weights = synaptic_weights

    # Activation function
    # Quantize the induced local field
    #
    # Params:	z - the value of the induced local field
    #
    # Returns:	an integer that corresponds to one of the two possible output values (usually 0 or 1)
    @staticmethod  # static since this method doesn't reference the self attribute of the class instance
    def activation_function(z):
        return 1 if z >= 0.0 else 0

    # Compute and return the weighted sum of all inputs (not including bias)
    #
    # Params:	inputs - a single input vector (which may contain multiple individual inputs)
    #
    # Returns:	a float value equal to the sum of each input multiplied by its
    #			corresponding synaptic weight
    def weighted_sum_inputs(self, inputs):
        weighted_inputs = 0.0
        for i in range(len(inputs) - 1):
            weighted_inputs += inputs[i] * self.synaptic_weights[i]
        return weighted_inputs

    # Compute the induced local field (the weighted sum of the inputs + the bias)
    #
    # Params:	inputs - a single input vector (which may contain multiple individual inputs)
    #
    # Returns:	the sum of the weighted inputs adjusted by the bias
    def induced_local_field(self, inputs):
        return self.bias + self.weighted_sum_inputs(inputs)

    # Predict the output for the specified input vector
    #
    # Params:	input_vector - a vector or row containing a collection of individual inputs
    #
    # Returns:	an integer value representing the final output, which must be one of the two
    #			possible output values (usually 0 or 1)
    def predict(self, input_vector):
        return self.activation_function(self.induced_local_field(input_vector))

    # Train this Perceptron
    #
    # Params:	training_set        - a collection of input vectors that represents a subset of the entire dataset
    #			learning_rate       - the amount to adjust the synaptic weights following an incorrect prediction
    #			number_of_epochs    - the number of times the entire training set is processed by the perceptron
    #
    # Returns:	no return value
    def train(self, training_set, learning_rate, number_of_epochs):
        # zero all the synaptic weights
        for i in range(len(self.synaptic_weights)):
            self.synaptic_weights[i] = 0.0
        # begin training
        for epoch in range(number_of_epochs):
            epoch_errors = 0
            for i in range(len(training_set)):
                prediction = self.predict(training_set[i])
                expected = training_set[i][60]
                error = expected - prediction
                if error != 0:
                    epoch_errors += 1
                    self.bias += learning_rate * error  # adjust bias as well as weights by error
                    for j in range(len(self.synaptic_weights)):
                        self.synaptic_weights[j] += learning_rate * error * training_set[i][j]
            # output epoch stats
            if epoch_errors > 0:
                print('Epoch number: ' + str(epoch) + '  Errors this epoch: ' + str(epoch_errors) + '   Bias: ' + str(self.bias))

    # Test this Perceptron
    #
    # Params:	test_set - the set of input vectors to be used to test the perceptron after it has been trained
    #
    # Returns:	a collection or list containing the actual output (predictions) for each input vector
    def test(self, test_set):
        predictions = list()
        for i in range(len(test_set)):
            predictions.append(self.predict(test_set[i]))
        return predictions
