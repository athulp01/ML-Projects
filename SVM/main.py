from sklearn.ensemble import RandomForestClassifier
import csv
import numpy as np

# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot

with open("./dataset/fashion-mnist_train.csv", "r") as f:
    x_train = list(csv.reader(f))
    x_train = np.array(x_train[1:], dtype=np.float)
    y_train = x_train[:, 0:1]
    y_train = np.ravel(y_train)
    x_train = x_train[:, 1:]

with open("./dataset/fashion-mnist_test.csv", "r") as f:
    x_test = list(csv.reader(f))
    x_test = np.array(x_test[1:], dtype=np.float)
    y_test = x_test[:, 0:1]
    y_test = np.ravel(y_test)
    x_test = x_test[:, 1:]

model = RandomForestClassifier()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))