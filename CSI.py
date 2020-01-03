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
        
        
        if algorithm == "gauss":
            self.c = matrix_calc(A, B).Gauss()
        elif algorithm == "jacobi":
            self.c = matrix_calc(A, B).Jacobi(100)
        elif algorithm == "gaussSeidel":
            self.c = matrix_calc(A, B).GaussSeidel(100)
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
    

    for value in [200, 300, 500]:
        n = value
        jmp = 1
        
        data_gen = data_generator(n, jmp)
        
        xx = np.arange(0.0, n, jmp)
        yy = data_gen.generate("flat")
        
        y = [v for i, v in enumerate(yy) if i % 2 == 0]
        x = np.arange(0.0, n, jmp * 2)
        
        spline = Spline(x, y, "gauss")
        
        rx = np.arange(0.0, n - jmp*2, 1)
        ry = [spline.calc(i) for i in rx]
        
        spline2 = Spline(x, y, "jacobi")
        rxx = np.arange(0.0, n - jmp*2, 1)
        ryy = [spline2.calc(i) for i in rxx]
        
        spline3 = Spline(x, y, "gaussSeidel")
        rxxx = np.arange(0.0, n - jmp*2, 1)
        ryyy = [spline3.calc(i) for i in rxxx]
        
        gauss_diff = []
        jacobi_diff = []
        gaussSeidel_diff = []
        basic_diff = []
        
        for k, i in enumerate(range(len(ry))):
            gauss_diff.append(ry[i] - yy[k])
            jacobi_diff.append(ryy[i] - yy[k])
            gaussSeidel_diff.append(ryyy[i] - yy[k])
            
        print("N = ", n)
        print("gauss = ", sum(gauss_diff))
        print("seidel = ", sum(gaussSeidel_diff))
        print("jacobi = ", sum(jacobi_diff))
        print("\n")
    
    
    #plt.plot(x, y, label="flat")
    
    plt.plot(xx[:-3], yy[:-3], label="flat")
    #plt.plot(xx, y, label="basic")
    plt.plot(rx, ry, label="gauss", color="purple")
    plt.plot(rxx, ryy, label="jacobi", color="black")
    #plt.plot(rxxx, ryyy, label="gaussSeidel", color="green")
    plt.legend()
    plt.show()
    