import numpy as np
from math import sqrt
from data_generator import data_generator
import bisect


class Spline:
    """Cubic Spline class"""

    def __init__(self, x, y):
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
        #print(A)
        #print("\n", B)
        self.c = np.linalg.solve(A, B)
        #print(self.c)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)
            
    def calc(self, t):
        u"""
        Calc position
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        #print(i, " i")
        #print(len(self.d), " d")
        #print(len(self.c), " c\n")
        #print(self.d)
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        u"""
        Calc first derivative
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        u"""
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        u"""
        search data segment index
        """
        print(self.x)
        print(x, "\n")
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        u"""
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        u"""
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        #  print(B)
        return B




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from random import uniform

    data_gen = data_generator(100)
    
    x = [-0.5, 0.0, 0.5, 1.0, 1.5]
    y = [3.2, 2.7, 6, 5, 6.5]
    
    x = np.arange(0.0, 100.0, 20)
    y = data_gen.generate("mutable")
    spline = Spline(x, y)
    print(x, y)
    plt.scatter(x, y)

    rx = np.arange(0, 80, 1)
    print(len(rx))
    print(len(x))
    ry = [spline.calc(i) for i in rx]

    plt.plot(x, y, label="basic")
    #plt.plot(rx, cubic_interp1d(rx, x, y), label="cubic")
    plt.plot(rx, ry, label="github")
    plt.legend()
    plt.show()