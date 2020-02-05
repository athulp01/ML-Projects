import numpy as np
from sklearn.model_selection import train_test_split
import csv
from sklearn.tree import DecisionTreeClassifier

with open("C:\\Users\\ASUS\\Documents\\GitHub\\ML-Projects\\Decision_Tree\\Dataset\\breast cancer classification.csv",'r') as f:
    x=list(csv.reader(f))
    for i in range(0, len(x)):
        if x[i][1] == 'M':
            x[i][1] = 0
        if x[i][1] == 'B':
            x[i][1] = 1
    x = np.array(x[1:],dtype=np.float)
    y=x[:,1:2]
    y=np.ravel(y)
    x=x[:,2:]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_test)
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
print(classifier.score(x_test,y_test))