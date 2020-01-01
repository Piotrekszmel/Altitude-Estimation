import numpy as np
from math import sqrt
from matrix_calculator import matrix_calc
from data_generator import data_generator
import bisect


class Spline:
    """Cubic Spline class"""
    def __init__(self, x, y, algorithm=None):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]
        
        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        print(A, "\n")
        
        if algorithm == "gauss":
            self.c = matrix_calc(A, B).Gauss()
        elif algorithm == "jacobi":
            self.c = matrix_calc(A, B).Jacobi(5)
        else:
            self.c = np.linalg.solve(A, B)
        

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)
            
    def calc(self, t):
        """Calc position if t is outside of the input x, return None"""

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        
        result = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
        return result

    def __search_index(self, x):
        """search data segment index"""
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """calc matrix A for spline coefficient c"""
        A = np.zeros(shape=(self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h):
        """calc matrix B for spline coefficient c"""
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return B


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    from random import uniform
    import math
    random.seed(45)
    data_gen = data_generator(1000)
    
    #x = [-0.5, 0.0, 0.5, 1.0, 1.5]
    #y = [3.2, 2.7, 6, 5, 6.5]
    
    
    x = np.arange(0.0, 1000.0, 20)
    y = [uniform(0, 10) for i in range(50)]
    
    y = data_gen.generate("flat")
    yy = data_gen.generate("mutable")
    spline = Spline(x, y, "jacobi")
    spline2 = Spline(x, yy, "gauss")
    
    plt.scatter(x, y)
    plt.scatter(x, yy)  
    
    rxx = np.arange(0.0, 980.0, 0.01)
    ryy = [spline.calc(i) for i in rxx]
    
    rx = np.arange(0.0, 980.0, 0.01)
    ry = [spline2.calc(i) for i in rx]

    plt.plot(x, y, label="flat")
    plt.plot(x, yy, label="mutable")
    plt.plot(rx, ry, label="gauss")
    plt.plot(rxx, ryy, label="jacobi")
    plt.legend()
    plt.show()
