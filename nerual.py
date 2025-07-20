import numpy as np
import pandas as pd
import os
import processing
from categories import getCategories

label_n = len(getCategories())

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is None or biases is None:
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))
        else:
            self.weights = weights
            self.biases = biases
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.biases)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalue):
        self.dinputs = dvalue.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, keepdims=True, axis=1))
        probabilities = exp_values / np.sum(exp_values, keepdims=True, axis=1)
        self.output = probabilities
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalue)

class Loss:
    def caculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidiences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidiences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidiences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(y_true)
        labels = len(y_true[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true=None):
        self.activation.forward(inputs)
        self.output = self.activation.output
        if y_true is not None:
            return self.loss.caculate(self.activation.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optmizer_SGD:
    def __init__(self, learning_rate, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update_params(self, layer):
        if self.momentum:
            weight_updates = self.momentum * layer.weight_momentum - self.learning_rate * layer.dweights
            bias_updates = self.momentum * layer.bias_momentum - self.learning_rate * layer.dbiases
            layer.weight_momentum = weight_updates
            layer.bias_momentum = bias_updates
            layer.weights += weight_updates
            layer.biases += bias_updates
        else:
            layer.weights -= self.learning_rate * layer.dweights
            layer.biases -= self.learning_rate * layer.dbiases

class Model:
    def __init__(self, use_existing, folder_path=None, add_gaussian_noise=False):
        self.add_gaussian_noise = add_gaussian_noise
        self.folder_path = folder_path or "parameters"


        self.dense1 = Layer_Dense(784, 300)
        self.activation1 = Activation_ReLu()
        self.dense2 = Layer_Dense(300, 60)
        self.activation2 = Activation_ReLu()
        self.dense3 = Layer_Dense(60, 50)
        self.activation3 = Activation_ReLu()
        self.dense4 = Layer_Dense(50, 40)
        self.activation4 = Activation_ReLu()
        self.dense5 = Layer_Dense(40, label_n)
        self.output_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

        if use_existing:
            self.dense1.weights = pd.read_csv(f"{self.folder_path}/Dense1W.csv", header=None).values
            self.dense1.biases = pd.read_csv(f"{self.folder_path}/Dense1B.csv", header=None).values
            self.dense2.weights = pd.read_csv(f"{self.folder_path}/Dense2W.csv", header=None).values
            self.dense2.biases = pd.read_csv(f"{self.folder_path}/Dense2B.csv", header=None).values
            self.dense3.weights = pd.read_csv(f"{self.folder_path}/Dense3W.csv", header=None).values
            self.dense3.biases = pd.read_csv(f"{self.folder_path}/Dense3B.csv", header=None).values
            self.dense4.weights = pd.read_csv(f"{self.folder_path}/Dense4W.csv", header=None).values
            self.dense4.biases = pd.read_csv(f"{self.folder_path}/Dense4B.csv", header=None).values
            self.dense5.weights = pd.read_csv(f"{self.folder_path}/Dense5W.csv", header=None).values
            self.dense5.biases = pd.read_csv(f"{self.folder_path}/Dense5B.csv", header=None).values

    def forward(self, inputs, labels):
        if self.add_gaussian_noise:
            inputs = processing.random_augment_image(inputs)
        self.dense1.forward(inputs)
        self.activation1.forward(self.dense1.output)
        self.dense2.forward(self.activation1.output)
        self.activation2.forward(self.dense2.output)
        self.dense3.forward(self.activation2.output)
        self.activation3.forward(self.dense3.output)
        self.dense4.forward(self.activation3.output)
        self.activation4.forward(self.dense4.output)
        self.dense5.forward(self.activation4.output)
        self.loss = self.output_activation.forward(self.dense5.output, labels)

    def backward(self, labels):
        self.output_activation.backward(self.output_activation.output, labels)
        self.dense5.backward(self.output_activation.dinputs)
        self.activation4.backward(self.dense5.dinputs)
        self.dense4.backward(self.activation4.dinputs)
        self.activation3.backward(self.dense4.dinputs)
        self.dense3.backward(self.activation3.dinputs)
        self.activation2.backward(self.dense3.dinputs)
        self.dense2.backward(self.activation2.dinputs)
        self.activation1.backward(self.dense2.dinputs)
        self.dense1.backward(self.activation1.dinputs)

    def predict(self):
        self.predictions = np.argmax(self.output_activation.output, axis=1)
        return self.predictions

    def predict_one(self, inputs):
        self.dense1.forward(inputs)
        self.activation1.forward(self.dense1.output)
        self.dense2.forward(self.activation1.output)
        self.activation2.forward(self.dense2.output)
        self.dense3.forward(self.activation2.output)
        self.activation3.forward(self.dense3.output)
        self.dense4.forward(self.activation3.output)
        self.activation4.forward(self.dense4.output)
        self.dense5.forward(self.activation4.output)
        self.output_activation.forward(self.dense5.output)
        return self.output_activation

    def get_accuracy(self, predictions, labels):
        return np.sum(predictions == labels) / labels.size

    def print_model(self, labels, i):
        predictions = self.predict()
        self.accuracy = self.get_accuracy(predictions, labels)
        print("-" * 12)
        print("Iteration: ", i, "Loss: ", self.loss)
        print(f"{np.sum(self.predictions == labels)} / {labels.size}")
        print("Accuracy: ", self.accuracy)

    def gradientDesent(self, inputs, y_true, iterations, target_accuracy):
        sgd = Optmizer_SGD(0.001, momentum=0.9)
        for i in range(iterations):
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
            for start in range(0, len(indices), 64):
                end = start + 64
                batch_indices = indices[start:end]
                X_batch = inputs[batch_indices]
                y_batch = y_true[batch_indices]

                self.forward(X_batch, y_batch)
                self.backward(y_batch)
                sgd.update_params(self.dense1)
                sgd.update_params(self.dense2)
                sgd.update_params(self.dense3)
                sgd.update_params(self.dense4)
                sgd.update_params(self.dense5)

            self.forward(inputs, y_true)
            self.print_model(y_true, i)
            if self.accuracy >= target_accuracy:
                break
            if i != 0 and i % 10 == 0:
                self.save_to_csv()

    def save_to_csv(self):
        os.makedirs(self.folder_path, exist_ok=True)
        pd.DataFrame(self.dense1.weights).to_csv(f"{self.folder_path}/Dense1W.csv", header=None, index=None)
        pd.DataFrame(self.dense1.biases).to_csv(f"{self.folder_path}/Dense1B.csv", header=None, index=None)
        pd.DataFrame(self.dense2.weights).to_csv(f"{self.folder_path}/Dense2W.csv", header=None, index=None)
        pd.DataFrame(self.dense2.biases).to_csv(f"{self.folder_path}/Dense2B.csv", header=None, index=None)
        pd.DataFrame(self.dense3.weights).to_csv(f"{self.folder_path}/Dense3W.csv", header=None, index=None)
        pd.DataFrame(self.dense3.biases).to_csv(f"{self.folder_path}/Dense3B.csv", header=None, index=None)
        pd.DataFrame(self.dense4.weights).to_csv(f"{self.folder_path}/Dense4W.csv", header=None, index=None)
        pd.DataFrame(self.dense4.biases).to_csv(f"{self.folder_path}/Dense4B.csv", header=None, index=None)
        pd.DataFrame(self.dense5.weights).to_csv(f"{self.folder_path}/Dense5W.csv", header=None, index=None)
        pd.DataFrame(self.dense5.biases).to_csv(f"{self.folder_path}/Dense5B.csv", header=None, index=None)
