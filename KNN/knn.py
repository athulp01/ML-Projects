import sklearn.neighbors
import  numpy as np
import csv
with open("C:\\Users\\ASUS\\Documents\\heart_x.csv", 'r') as f:
    heart_x = list(csv.reader(f, delimiter=","))
    heart_x = np.array(heart_x[1:], dtype=np.float)
with open("C:\\Users\\ASUS\\Documents\\heart_result.csv", 'r') as f:
    heart_y = list(csv.reader(f, delimiter=","))
    heart_y = np.array(heart_y[1:], dtype=np.int)
    heart_y = np.ravel(heart_y)

knn = sklearn.neighbors.KNeighborsClassifier()

train_x,test_x = heart_x[:280],heart_x[280:]
train_y, test_y = heart_y[:280],heart_y[280:]
knn.fit(train_x, train_y)
print(knn.score(test_x, test_y))
