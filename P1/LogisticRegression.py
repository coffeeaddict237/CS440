"""
LogisticRegression.py

CS440/640: Lab-Week5

Lab goal: 1) Implement logistic regression classifier
"""

import numpy as np 
import matplotlib.pyplot as plt 
import csv
from sklearn.metrics import confusion_matrix

class LogisticRegression:
    """
    This class implements a Logistic Regression Classifier.
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initializes the parameters of the logistic regression classifer to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
        """
        
        self.theta = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.bias = np.zeros((1, output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total cost on the dataset.
        
        args:
            X: Data array
            y: Labels corresponding to input data
        
        returns:
            cost: average cost per data sample
        """
        #TODO:
        z = np.dot(X, self.theta) + self.bias
        exp_z = np.exp(z);
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        #print(str(softmax_scores))
        #print(len(softmax_scores))

        mean_cost = 0
        for i in range(len(X)):
            new_y = np.zeros(self.output_dim)
            new_y[int(y[i])] = 1
            mean_cost += -np.sum(new_y * np.log(softmax_scores[i]))
        mean_cost /= len(X)

        
        return mean_cost

    
    #--------------------------------------------------------------------------
 
    def predict(self,X):
        """
        Makes a prediction based on current model parameters.
        
        args:
            X: Data array
            
        returns:
            predictions: array of predicted labels
        """
        #TODO:
        z = np.dot(X, self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis=1)
        return predictions
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y):
        """
        Learns model parameters to fit the data.
        """  
        #TODO:
        current = self.compute_cost(X,y)
        previous = 0.0
        difference = current - previous
        convergence_pt = 0.0001

        while difference >= convergence_pt:
            z = np.dot(X, self.theta) + self.bias
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            differences = []
            
            for i in range(len(X)):
                new_y = np.zeros(self.output_dim)
                new_y[int(y[i])] = 1
                difference_y = softmax_scores[i] - new_y
                differences.append(difference_y)

            gradient_wrt_weight = np.dot(np.transpose(X), differences)
            gradient_wrt_bias = np.dot( np.transpose( np.ones( ( len(X), 1) ) ), differences)
            #w = w - learning_rate * gradient of cost wrt weights
            self.theta = self.theta - 0.001 * gradient_wrt_weight
            #b = b - learning_rate * gradient of cost wrt biases
            self.bias = self.bias - 0.001 * gradient_wrt_bias

            previous = current
            difference = abs(current - previous)

        return self

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def plot_decision_boundary(model, X, y):
    """
    Function to print the decision boundary given by model.
    
    args:
        model: model, whose parameters are used to plot the decision boundary.
        X: input data
        y: input labels
    """
    
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()


################################################################################    

linear = True
if linear:
    X_values = np.genfromtxt('DATA/Linear/X.csv', delimiter=",")
    y_values = np.genfromtxt('DATA/Linear/y.csv', delimiter=",")
else:
    X_values = np.genfromtxt('DATA/NonLinear/X.csv', delimiter=",")
    y_values = np.genfromtxt('DATA/NonLinear/y.csv', delimiter=",")

v = LogisticRegression(2,2)
v.fit(X_values, y_values)
plot_decision_boundary(v, X_values, y_values)

##########################Question 6###########################################

dig = LogisticRegression(64, 10)
X_dig = np.genfromtxt('DATA/Digits/X_train.csv',delimiter=",")
y_dig = np.genfromtxt('DATA/Digits/y_train.csv',delimiter=",")

dig.fit(X_dig, y_dig)

X = np.genfromtxt('DATA/Digits/X_train.csv',delimiter=",")
prediction = dig.predict(X)
actual = np.genfromtxt('DATA/Digits/y_train.csv',delimiter=",")
m = confusion_matrix(actual, prediction)
print(str(m))

predictions = dig.predict(X.dig)
correct = 0

for i in range(len(predictions)):
    if predictions[i] == y_dig[i]:
        correct += 1
correct /= len(predictions)
print("Accuracy: " + str(correct * 100) + "%")
