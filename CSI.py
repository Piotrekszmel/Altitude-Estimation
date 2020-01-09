import numpy as np
from math import sqrt
from matrix_calculator import matrix_calc
from data_generator import data_generator
import bisect
import time

class Spline:
    def __init__(self, x, y, algorithm=None):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]
        self.time = 0
        # calc coefficient c
        
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        
        if algorithm == "gauss":
            
            self.matrix = matrix_calc(A, B).Gauss()
            
        elif algorithm == "jacobi":
            
            self.matrix, _= np.asarray(matrix_calc(A, B).Jacobi(10))
               
        elif algorithm == "gaussSeidel":
            
            self.matrix, _ = matrix_calc(A, B).GaussSeidel(10)
            
        else:
            self.matrix = np.linalg.solve(A, B)
        
        for i in range(self.nx - 2):
            self.c.append(self.matrix[i] / 2)
            self.b.append(((self.a[i+1] - self.a[i]) / h[i]) - (((2*self.matrix[i] + self.matrix[i+1]) / 6) * h[i+1]))
            self.d.append((self.matrix[i+1] - self.matrix[i]) / (6 * h[i+1]))

    def calc(self, t):
        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = abs(t - self.x[i])
        
        result = self.a[i] + self.b[i] * dx + self.matrix[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
        return result

    def __search_index(self, x):
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
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
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return B


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    from random import uniform
    import math 
    
    random.seed(100)
    gauss_time = []
    jacobi_time = []
    gaussSeidel_time = []

    for d_type in  ["mutable"]:
        gauss_time = []
        jacobi_time = []
        gaussSeidel_time = []
        
        for value in [100]:
            n = value
            jmp = 1
            
            data_gen = data_generator(n, jmp)
            
            xx = np.arange(0.0, n, jmp)
            yy = data_gen.generate(d_type)
            y = [v for i, v in enumerate(yy) if i % 2 == 0]
            x = np.arange(0.0, n, jmp * 2)
           
            spline = Spline(x, y, "gauss")
            rx = np.arange(0.0, n - jmp*2, 1)
            ry = [spline.calc(i) for i in rx[:-2]]

            plt.plot(xx, yy, label="true")
            plt.plot(rx[:-2], ry, label="gauss")
            plt.legend()
            plt.show()
    print(len(ry))
    print(len(y))
    print(len(yy))      
    error = 0
    for j, i in enumerate(range(1, len(y)-2)):
        error += abs(ry[i*2] - yy[i*2])
    print(error / j)

    error = 0
    for j, i in enumerate(range(len(ry))):
        error += abs(ry[i] - yy[i])
    print(error / j)
