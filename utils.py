import numpy as np
from PIL import Image
import os

def load_data(data_dir = 'D:/Documents/Waste Classification/archive (1)/Garbage classification/Garbage classification'):
    X = []
    y = []
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for label, folder in enumerate(subfolders):
        for file_name in os.listdir(folder):
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                image = Image.open(os.path.join(folder, file_name)).convert('L').resize((128, 128))
                image_array = np.array(image)
                X.append(image_array)
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    classes = os.listdir(data_dir)

    print('Found', len(X), 'images belonging to', len(classes), 'classes.')
    return X, y, classes

def train_test_split(X, y, test_size = 0.2):
    n = X.shape[0]
    indices = np.random.permutation(n)
    X = X[indices]
    y = y[indices]
    split = int(n * test_size)
    X_train, X_test = X[split:], X[:split]
    y_train, y_test = y[split:], y[:split]
    return X_train, X_test, y_train, y_test

def calculate_acc(y_pred, y_test):
    acc = np.sum(y_pred == y_test) / len(y_test) * 100
    print('Accuracy: {}'.format(acc))