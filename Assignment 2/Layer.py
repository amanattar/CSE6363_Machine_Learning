import numpy as np
import pickle

class Layer:
    def __init__(self, input, output):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, output_error, learning_rate):
        raise NotImplementedError
    

class Linear_Layer(Layer):
    def __init__(self, no_of_inputs, no_of_outputs):
        self.weights = np.random.randn(no_of_inputs, no_of_outputs) * 0.1
        self.biases = np.random.randn(1,no_of_outputs)

    def forward(self, input):
        self.input = input
        temp = np.matmul(self.input, self.weights)
        self.output = np.add(temp,self.biases)
        return self.output
    
    def backward(self, output_error, learning_rate):
        input_error = np.matmul(output_error, self.weights.T)
        weights_error = np.matmul(self.input.T, output_error)
        biased_error = np.sum(output_error, axis=0, keepdims=True)
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * biased_error
        return input_error


class Sigmoid_Function(Layer):
    def __init__(self):
       self.output = None

    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self, learning_rate):
        grad =  self.output @ np.subtract(1, self.output)
        return grad @ learning_rate


class Hyperbolic_Tangent_Function(Layer):
    def __init__(self):
       self.output = None

    def forward(self, input):
        return np.tanh(input)
    
    def backward(self, learning_rate):
        return 1-np.tanh(learning_rate)**2
    
class Softmax_Function(Layer):
    def __init__(self):
        self.output = None

    def forward(self, input):
        return (np.exp(input - np.max(input))) / np.sum(np.exp(input - np.max(input)), axis=0, keepdims=True)
    
    def backward(self, soft_output, error_grad):
        return soft_output
    
class Cross_Entropy_Loss(Layer):
    def __init__(self):
        self.output = None

    def forward(self, predic, target):
       return -target * np.log(predic)
    
    def backward(self, predic, target):
        return target - predic
    
class Activation_Function(Layer):
    def __init__(self, state, sensitivity):
        self.state = state
        self.sensitivity = sensitivity

    def forward(self, input):
        self.input = input
        return self.state(input)
    
    def backward(self, loss, learning_rate):
        return self.sensitivity(self.input) * loss
    
class Sequential(Layer):
    def __init__(self):
        self.layers_l = []
        self.weight_tensor = None

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_weights, f)

    def load(self, path):
        with open(path, "rb") as f:
            weights = pickle.load(f)
        return weights


    def mse(self, predic, target):
        return np.mean(np.square(target - predic))
    
    def msegrad(self, target, predic):
        return 2 * (predic - target) / len(target)
    
    def add(self, layer):
        self.layers_l.append(layer)

    def mod_predic(self, data):
        neural_depth = len(data)
        predected_result = []
        for i in range(neural_depth):
            neural_output = data[i]
            for layer in self.layers_l:
                neural_output = layer.forward(neural_output)
            predected_result.append(neural_output)
        return predected_result
    
    def pred(self, data):
        neural_depth = len(data)
        predected_result = []
        for i in range(neural_depth):
            neural_output = data[i]
            for layer in self.layers_l:
                neural_output = layer.forward(neural_output)
            predected_result.append(neural_output)
        return predected_result
    
    def train(self, data, target,  epoch, learning_rate):
        train_loss = []
        patience = 0
        error = []
        for i in range(epoch):
            err = 0
            for j in range(len(data)):
                neural_output = data[j]
                for layer in self.layers_l:
                    neural_output = layer.forward(neural_output)
                err += self.mse(target[j], neural_output)
                loss = self.msegrad(target[j], neural_output)
                for layer in reversed(self.layers_l):
                    loss = layer.backward(loss, learning_rate)
            err = err / len(data)
            train_loss.append(err)

            if len(train_loss) != 0:
                if train_loss[-1] < err:
                    patience += 1

            #print("Epoch: ", i+1, "Loss: ", err)

            if patience == 5:
                break

        return train_loss
    
    def get_weights(self):
        self.weight_tensor = list()
        for layer in self.layers_l:
            if isinstance(layer, Linear_Layer):
                self.weight_tensor.append(layer.weights)
        return self.weight_tensor
    


