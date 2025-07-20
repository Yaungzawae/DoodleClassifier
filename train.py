import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from nerual import Model

from categories import getCategories
categories = getCategories()

X = np.load("X.npy")
X = X.reshape(X.shape[0], -1)  # Flatten the images
y = np.load("y.npy")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Model(use_existing=False, folder_path="parameters", add_gaussian_noise=True)
model.gradientDesent(x_train, y_train, iterations=500, target_accuracy=0.99)
