"""
LogisticRegression.py

CS440/640: Lab-Week5

Lab goal: 1) Implement logistic regression classifier
"""

import numpy as np 
import matplotlib.pyplot as plt 
import csv
from sklearn.metrics import confusion_matrix
import numpy as np 
import matplotlib.pyplot as plt 
import csv
from sklearn.metrics import confusion_matrix

np.seterr(divide='ignore', invalid='ignore', over='ignore')

class NeuralNet:
    """
    This class implements a Logistic Regression Classifier.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim, epsilon):
        """
        Initializes the parameters of the logistic regression classifer to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
            hidden_dim: Number of nodes in hidden layer
            epsilon: learning rate
        """
        
        self.theta = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.bias = np.zeros((1, hidden_dim))
        self.theta_hidden = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.bias_hidden = np.zeros(1, output_dim)
        self.epsilon = epsilon
        
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
        sigm = sigmoid(z)
        z_hidden = np.dot(sigm, self.theta_hidden) + self.bias_hidden
        exp_z = np.exp(z_hidden);
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        #print(str(softmax_scores))
        #print(len(softmax_scores))

        cost_per_pt = -np.log(softmax_scores[range(len(X)), y.astype('int64')])
        total = np.sum(cost_per_pt)
        return 1./len(X) * total

    
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
        sigm = sigmoid(z)
        z_hidden = np.dot(sigm, self.theta_hidden) + self.bias_hidden
        exp_z = np.exp(z_hidden)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis=1)
        return predictions
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y):
        """
        Learns model parameters to fit the data.
        """  
        #TODO:

        for i in range(5000):
            z_1 = X.dot(self.theta) + self.bias
            sigm_1 = sigmoid(z_1)
            z_2 = sigm_1.dot(self.theta_hidden) + self.bias_hidden
            exp = np.exp(z_2)

            d_3 = softmax_scores
            d_3[range(len(X)), y.astype('int64')] -= 1
            dw_2 = (sigm_1.T).dot(d_3)
            db_2 = np.sum(d_3, axis=0, keepdims=True)
            d_2 = d_3.dot(self.theta_hidden.T) * sigmoidPrime(z_1)
            dw_1 = np.dot(X.T, d_2)
            db_1 = np.sum(d_2, axis=0)

            self.theta -= self.epsilon * dw_1
            self.bias -= self.epsilon * db_1
            self.theta_hidden -= self.epsilon * dw_2
            self.bias_hidden -= self.epsilon * db_2

        return self

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidPrime(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

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

v = NeuralNet(2,2,10,0.01)
v.fit(X_values, y_values)
plot_decision_boundary(v, X_values, y_values)

##############################Question 7########################################

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

##################################Question 4####################################

def learning_rate_graph(X, y, rate):
    graph_y = [0] * 5
    for i in range(len(graph_y)):
        N = NeuralNet(2,2,5,rate)
        N.fit(X, y)
        graph_y[i] = N.compute_cost(X, y)

    return graph_y

##################################Question 5####################################

def hidden_nodes_graph(X, y, num_nodes):
    graph_y = [0] * 10
    for i in range(len(graph_y)):
        N = NeuralNet(2,2, num_nodes, 0.01)
        N.fit(X, y)
        graph_y[i] = N.compute_cost(X, y)

    return graph_y
