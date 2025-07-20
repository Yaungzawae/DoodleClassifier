import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from nerual import Model
import processing
from categories import getCategories
from collections import Counter
categories = getCategories()

X = np.load("X.npy")
X = X.reshape(X.shape[0], -1)  # Flatten the images
y = np.load("y.npy")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

correct = 0
model = Model(use_existing=True, folder_path="parameters")


model.forward(x_test, y_test)
a = model.predict()
incorrect = a[a != y_test]
correct = a[a == y_test]
incorrect.sort()
print(Counter(incorrect))
print(Counter(correct))
model.print_model(y_test, 0)
