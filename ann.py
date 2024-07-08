import numpy as np
import pandas as pd

class ANN:
    def __init__(self, hidden_layer_node_sizes, featureDataframe, activationFunctions, threshold=0.5, numberOfClasses=1):
        self.features = featureDataframe
        self.numberOfClasses = numberOfClasses
        hidden_layer_node_sizes.append(numberOfClasses)
        self.node_sizes = hidden_layer_node_sizes
        self.layer_count = len(self.node_sizes)
        self.activationFunctions = activationFunctions
        self.hiddenLayers = []
        self.threshold = threshold
        self.results = []

        # Error handling for invalid inputs
        if self.features.shape[1] <= 0:
            raise ValueError("Number of features must be greater than zero!")
        if self.node_sizes[0] != self.features.shape[1]:
            raise ValueError("First layer size must be equal to number of features!")
        if len(self.activationFunctions) != self.layer_count+1:
            raise ValueError("Number of activation functions must match number of layers!")

    class HiddenLayer:
        def __init__(self):
            self.nodes = []
            self.activationValues = []

        class Node:
            def __init__(self):
                self.weights = None
                self.bias = None
                self.activationValue = None

    def __linearModel(self, w, x, b):
        return np.dot(w, x) + b

    def __activationFunction(self, z, activation):
        if activation == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif activation == "relu":
            return np.maximum(0, z)
        elif activation == "softmax":
            exp_z = np.exp(z - np.max(z))
            return exp_z / np.sum(exp_z)
        else:
            raise ValueError("Activation function must be 'sigmoid', 'relu' or 'softmax'!")

    def model(self, node, features, activation):
        linear_output = self.__linearModel(node.weights, features, node.bias)
        return self.__activationFunction(linear_output, activation)

    def __predictResult(self, features):
        for L in range(self.layer_count):
            hidden_layer = self.HiddenLayer()
            self.hiddenLayers.append(hidden_layer)
            
            node_count = self.node_sizes[L]
            
            for N in range(node_count):
                node = self.HiddenLayer.Node()
                hidden_layer.nodes.append(node)
                
                if L > 0:
                    node_size = self.node_sizes[L-1]
                else:
                    node_size = len(self.features[0])
                
                node.weights = np.random.randn(node_size)
                node.bias = np.random.randn()
                
                activation = self.activationFunctions[L]
                node.activationValue = self.model(node, features, activation)
                
                hidden_layer.activationValues.append(node.activationValue)
        return self.hiddenLayers[-1].activationValues
        
    def predict(self):
        for f in range(self.features.shape[0]):
            preds = self.__predictResult(self.features[f])
            if 'softmax' in self.activationFunctions:
                result = np.argmax(preds)
            else:
                result = preds
            self.results.append(result)

        if 'softmax' in self.activationFunctions:
            def to_one_hot(class_indices, num_classes):
                one_hot = np.zeros((len(class_indices), num_classes))
                for i, index in enumerate(class_indices):
                    one_hot[i, index] = 1
                return one_hot

            one_hot_predictions = to_one_hot(self.results, self.numberOfClasses)
            return one_hot_predictions

        return np.array(self.results).flatten()


# Dataset for classification
df = pd.read_csv('Iris.csv')
df.drop(columns='Id', inplace=True)
df = pd.concat([df, pd.get_dummies(df.Species)], axis=1)
df.drop(columns='Species', inplace=True)
df = df.sample(frac=1).reset_index(drop=True) 

y_true_classification = df.iloc[:, 4:].values

# Features
featureDataframe = df.iloc[:, :4].values

# Activation Functions for each layer
activationFunctions_classification = ["relu", "relu", "softmax"]

# Forward Propagation for Classification
ann_classification = ANN(
    hidden_layer_node_sizes=[featureDataframe.shape[1]],
    featureDataframe=featureDataframe,
    activationFunctions=activationFunctions_classification,
    numberOfClasses=3
)

y_pred_classification = ann_classification.predict()

# Dataset for Regression
df_reg = pd.read_csv('Rent.csv').head(100)
df_reg = df_reg.iloc[:, 2:]
y_true_regression = df_reg['Rent']
df_reg.drop(columns='Rent',inplace=True)
featureDataframe_reg = pd.get_dummies(df_reg).values

# Forward Propagation for Regression
activationFunctions_regression = ["relu", "relu", "sigmoid"]

ann_regression = ANN(
    hidden_layer_node_sizes=[featureDataframe_reg.shape[1]],
    featureDataframe=featureDataframe_reg,
    activationFunctions=activationFunctions_regression,
)

y_pred_regression = ann_regression.predict()


# This implementation is just for understanding fundamentals of forward propagation.