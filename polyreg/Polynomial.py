import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import OrderedDict
from scipy import linalg


class PolynomialRegressor():
#   order : Degree of the fitting function
    def __init__(self, order):
        self.order = order
    
    def fit(self, x, y):
        self.x = x
        self.y = y
        d = [np.ones([1, len(x)])[0]]
#       compute the powers of x from 1 to self.order
        for i in np.arange(1, self.order+1):
            d.append(self.x ** i)
        X = np.column_stack(d)
#       Use normal equation directly to find theta
        self.theta = np.matmul(np.matmul(linalg.pinv(np.matmul(np.transpose(X),X)), np.transpose(X)), y)

    def predict(self, x):
#       Find the predicted value
        y = self.theta[0]
        for i in np.arange(1, len(self.theta)):
            y += self.theta[i]*x ** i
        return y

    def plot(self):
        plt.figure()
        plt.scatter(self.x, self.y, s = 30, c = 'b') 
        line = self.theta[0] 
        label_holder = []
        label_holder.append('%.*f' % (2, self.theta[0]))
        for i in np.arange(1, len(self.theta)):            
            line += self.theta[i] * self.x ** i 
            label_holder.append(' + ' +'%.*f' % (2, self.theta[i]) + r'$x^' + str(i) + '$') 
        plt.plot(self.x, line, label = ''.join(label_holder))        
        plt.title('Polynomial Fit: Order ' + str(len(self.theta)-1))
        plt.xlabel('x')
        plt.ylabel('y') 
        plt.legend(loc = 'best')    
        plt.show()


def main():
    style.use('ggplot')
    x = [1, 2, 3, 4, 5, 7]
    y = [1, 4 ,9, 16, 25, 49]
    reg = PolynomialRegressor(2)
    reg.fit(x, y)
    reg.plot()

if __name__ == "__main__":
    main()