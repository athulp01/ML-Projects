from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import csv
import numpy as np

with open("./dataset/creditcard.csv", "r") as f:
    x = list(csv.reader(f))
    x = np.array(x[1:], dtype=np.float)
    y = x[:, 30:]
    y = np.ravel(y)
    x = x[:, 0:30]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
model = GaussianNB()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))