import numpy as np
#import pylance
import Layer
from Layer import Layer, Linear_Layer, Sigmoid_Function, Hyperbolic_Tangent_Function, Softmax_Function
from Layer import Activation_Function
from Layer import Sequential
from sklearn.metrics import accuracy_score

from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

def normalizeInput(x):
    x = x.reshape(x.shape[0], 1, 28*28)
    x = x.astype('float32')
    x /= 255
    return x

def toBinary(arr):
    bi_arr = []
    for i in arr:
        max = np.argmax(i)
        binary = np.zeros(10)
        binary[max] = 1
        bi_arr.append(binary)
    return np.array(bi_arr)

def accuracy(y_true, y_pred):
    y_pred_index = np.argmax(y_pred, axis=1)
    y_true_index = np.argmax(y_true, axis=1)
    crr = np.sum(y_pred_index == y_true_index)
    return (crr/len(y_true) * 100)



x_train = normalizeInput(x_train)
x_val = normalizeInput(x_val)
x_test = normalizeInput(x_test)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)


model1 = Sequential()
model1.add(Linear_Layer(28*28, 100))
tanh = Hyperbolic_Tangent_Function()
model1.add(Activation_Function(tanh.forward, tanh.backward))
model1.add(Linear_Layer(100, 50))
model1.add(Activation_Function(tanh.forward, tanh.backward))
model1.add(Linear_Layer(50, 10))
model1.add(Activation_Function(tanh.forward, tanh.backward))

print("----------------------------------------------------")
print("Model 1")
print("784 -> 100 -> 50 -> 10")
print("Activation Function: Hyperbolic Tangent")
print("Learning rate: 0.1")
print("Max Epochs: 10")
print("Early stopping: 5")
print("----------------------------------------------------")

print("Model 1 is training on training set")
model1_loss_train = model1.train(x_train, y_train, 10, 0.1)
print("Model 1 is successfully train on training set")

print("----------------------------------------------------")

print("Model 1 is training on validation set")
model1_loss_val = model1.train(x_val, y_val, 10, 0.1)
print("Model 1 is successfully train on validation set")

model1_pred_test = model1.pred(x_test)
temp1 = toBinary(model1_pred_test)

print("----------------------------------------------------")
print("Model 1")
print("Test Accuracy: ", accuracy(y_test, temp1))
print("----------------------------------------------------")


model1.get_weights()
model1.save("MNIST_model1.w")

plt.plot(model1_loss_train)
plt.plot(model1_loss_val)
plt.title('Model 1 Loss on training and validation set')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

model2 = Sequential()
model2.add(Linear_Layer(28*28, 50))
tanh = Hyperbolic_Tangent_Function()
model2.add(Activation_Function(tanh.forward, tanh.backward))
model2.add(Linear_Layer(50, 10))
model2.add(Activation_Function(tanh.forward, tanh.backward))

print("----------------------------------------------------")
print("Model 2")
print("784 -> 50 -> 10")
print("Activation Function: Hyperbolic Tangent")
print("Learning rate: 0.1")
print("Max Epochs: 10")
print("Early stopping: 5")
print("----------------------------------------------------")

print("Model 2 is training on training set")
model2_loss_train = model2.train(x_train, y_train, 10, 0.1)
print("Model 2 is successfully train on training set")

print("----------------------------------------------------")

print("Model 2 is training on validation set")
model2_loss_val = model2.train(x_val, y_val, 10, 0.1)
print("Model 2 is successfully train on validation set")

model2_pred_test = model2.pred(x_test)
temp2 = toBinary(model2_pred_test)

print("----------------------------------------------------")
print("Model 2")
print("Test Accuracy: ", accuracy(y_test, temp2))
print("----------------------------------------------------")


model2.get_weights()
model2.save("MNIST_model2.w")

plt.plot(model2_loss_train)
plt.plot(model2_loss_val)
plt.title('Model 2 Loss on training and validation set')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


model3 = Sequential()
model3.add(Linear_Layer(28*28, 100))
tanh = Hyperbolic_Tangent_Function()
model3.add(Activation_Function(tanh.forward, tanh.backward))
model3.add(Linear_Layer(100, 30))
model3.add(Activation_Function(tanh.forward, tanh.backward))
model3.add(Linear_Layer(30, 10))
model3.add(Activation_Function(tanh.forward, tanh.backward))

print("----------------------------------------------------")
print("Model 3")
print("784 -> 100 -> 30 -> 10")
print("Activation Function: Hyperbolic Tangent")
print("Learning rate: 0.1")
print("Max Epochs: 10")
print("Early stopping: 5")
print("----------------------------------------------------")

print("Model 3 is training on training set")
model3_loss_train = model3.train(x_train, y_train, 10, 0.1)
print("Model 3 is successfully train on training set")

print("----------------------------------------------------")

print("Model 3 is training on validation set")
model3_loss_val = model3.train(x_val, y_val, 10, 0.1)
print("Model 3 is successfully train on validation set")

model3_pred_test = model3.pred(x_test)
temp3 = toBinary(model3_pred_test)

print("----------------------------------------------------")
print("Model 3")
print("Test Accuracy: ", accuracy(y_test, temp3))
print("----------------------------------------------------")


model3.get_weights()
model3.save("MNIST_model3.w")


plt.plot(model3_loss_train)
plt.plot(model3_loss_val)
plt.title('Model 3 Loss on training and validation set')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()



