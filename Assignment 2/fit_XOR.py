import numpy as np
#import pylance
import Layer
from Layer import Layer, Linear_Layer, Sigmoid_Function, Hyperbolic_Tangent_Function, Softmax_Function
from Layer import Activation_Function
from Layer import Sequential
from sklearn.metrics import accuracy_score

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

model1 = Sequential()
model1.add(Linear_Layer(2, 3))
tanh = Hyperbolic_Tangent_Function()
model1.add(Activation_Function(tanh.forward, tanh.backward))
model1.add(Linear_Layer(3, 1))
model1.add(Activation_Function(tanh.forward, tanh.backward))
model1.train(x_train, y_train, 1000, 0.1)
output1 = model1.pred(x_train)
#print(output1)
threshold = 0.5
predicted = output1
binary_predicted_output1 = [int(p > threshold) for p in predicted]
#print(binary_predicted_output1)

acc_for_Model1 = accuracy_score([0,1,1,0], binary_predicted_output1)
#print("Accuracy Model 1 (hyperbolic tangent activations) : " ,acc_for_Model1)
print("----------------------------------------------------")
print("Model 1")
print("2 -> 3 -> 1")
print("Activation Function: Hyperbolic Tangent")
print("Learning rate: 0.1")
print("Max Epochs: 1000")
print("Early stopping: 5")
print("----------------------------------------------------")
print("Model 1")
print("Test Accuracy: ", acc_for_Model1)
print("----------------------------------------------------")

model1.get_weights()
model1.save("XOR_solved_tan.w")


model2 = Sequential()
model2.add(Linear_Layer(2, 3))
sigmoid = Sigmoid_Function()
model2.add(Activation_Function(sigmoid.forward, sigmoid.backward))
model2.add(Linear_Layer(3, 1))
model2.add(Activation_Function(sigmoid.forward, sigmoid.backward))
model2.train(x_train, y_train, 1000, 0.1)
output2 = model2.pred(x_train)
#print(output2)
threshold = 0.5
predicted = output2
binary_predicted_output2 = [int(p > threshold) for p in predicted]
#print(binary_predicted_output2)

acc_for_Model2 = accuracy_score([0,1,1,0], binary_predicted_output2)
#print("Accuracy Model 2 (sigmoid activations) : " ,acc_for_Model2)
print("----------------------------------------------------")
print("Model 2")
print("2 -> 3 -> 1")
print("Activation Function: Sigmoid")
print("Learning rate: 0.1")
print("Max Epochs: 1000")
print("Early stopping: 5")
print("----------------------------------------------------")
print("Model 2")
print("Test Accuracy: ", acc_for_Model2)
print("----------------------------------------------------")

model2.get_weights()
model2.save("XOR_solved_sigmoid.w")