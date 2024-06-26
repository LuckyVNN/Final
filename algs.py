import numpy as np
import math
import copy
import collections
import pickle


class KNN():
  def __init__(self, k):
      self.k = k
  def euclidean_distance(self, x1, x2):
     return np.sqrt(np.sum((x1 - x2) ** 2))
  def accuracy(self, y_test):
    acc = np.sum(self.y_pred == y_test) / len(y_test) * 100
    print('Accuracy: {}'.format(acc))
    return acc
  def fit(self, X_train, y_train, X_test):
    self.y_pred = []
    for i in range(len(X_test)):
      distances = [self.euclidean_distance(X_test[i], x) for x in X_train]
      k_idx = np.argsort(distances)[:self.k]
      k_labels = [y_train[idx] for idx in k_idx]  
      most_common = collections.Counter(k_labels).most_common(1)
      self.y_pred.append(most_common[0][0])
    return np.array(self.y_pred)
  
  def save(self, filename):
    with open(filename, 'wb') as file:
      pickle.dump(self, file)
  @staticmethod
  def load(filename):
      with open(filename, 'rb') as file:
            return pickle.load(file)

class LogisticRegression(object):
  def __init__(self, epochs, alpha):
    self.epochs = epochs
    self.alpha = alpha

  def sigmoid(self, z):
    s = 1/(1+np.exp(-z))
    return s
  def initialize_with_zeros(self, dim):
    w = np.random.randn(dim, 1) * 0.01
    b = 0
    return w, b
  def propagate(self, w, b, X, y):
    m = X.shape[1]

    A = self.sigmoid(np.dot(w.T, X) + b)
    loss = y*np.log(A) + (1-y)*np.log(1-A)
    cost = -1/m * np.sum(loss, axis = 1, keepdims = True)

    dw = 1/m * np.dot(X, (A- y).T)
    db = 1/m * np.sum(A - y)

    cost = np.squeeze(np.array(cost))

    grads = {'dw': dw,
              'db': db}
    return grads, cost

  def optimize(self, w, b, X, y, print_cost = True):
      w = copy.deepcopy(w)
      b = copy.deepcopy(b)
      
      self.costs = []

      for i in range(self.epochs):
        grads, cost = self.propagate(w, b, X, y)
        dw = grads['dw']
        db = grads['db']

        #Updating
        w = w - self.alpha * dw
        b = b - self.alpha * db
        self.costs.append(cost)
        if i % math.ceil(self.epochs / 10) == 0:
              if print_cost:
                  print ("Cost after iteration %i: %f" %(i, cost))
      
      params = {"w": w,
                "b": b}
      
      grads = {"dw": dw,
              "db": db}
      
      return params, grads
  def predict(self, w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = self.sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
      Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    return Y_prediction
  def fit(self, X_train, y_train, X_test, y_test, print_cost = False):
    w, b = self.initialize_with_zeros(X_train.shape[0])
    parameters, grads = self.optimize(w, b, X_train, y_train, print_cost)
    w = parameters['w']
    b = parameters['b']

    Y_prediction_test = self.predict(w, b, X_test)
    Y_prediction_train = self.predict(w, b, X_train)


    if print_cost:
      print("Train accuracy: {}".format(100 - np.mean(np.abs(Y_prediction_train - y_train)) * 100))
      print('Test accuracy: {}'.format(100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100))
    d = {
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : self.alpha,
         "num_iterations": self.epochs}
    
    return d
  def save(self, filename):
    with open(filename, 'wb') as file:
      pickle.dump(self, file)


  @staticmethod
  def load(filename):
      with open(filename, 'rb') as file:
            return pickle.load(file)