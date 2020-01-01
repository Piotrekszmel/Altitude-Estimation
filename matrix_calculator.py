import numpy as np
import random
import copy


class matrix_calc:
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.row = 1
        self.X = np.ones_like(self.B, dtype=np.float64)
        self.temp_A = copy.deepcopy(self.A)
        self.temp_B = copy.deepcopy(self.B)

    def Gauss(self):
        for i in range(self.row, len(self.A)):
            
            values = [(self.A[j, i - 1], j) for j in range(i - 1, len(self.A))]
            max_value = max(abs(value[0]) for value in values)
            index_max = [val[1] for val in values if abs(val[0]) == max_value][0]
            
            B_temp = copy.deepcopy(self.B[i - 1])
            self.B[i - 1] = self.B[index_max]
            self.B[index_max] = B_temp
            
            for j in range(len(self.A)):
                r = copy.deepcopy(self.A[i - 1, j])
                self.A[i - 1, j] = self.A[index_max, j]
                self.A[index_max, j] = r
            for ii in range(i, len(self.A)):
                self.calculation_A_B(ii)
            
            self.row += 1
        
        self.X = self.count()
        return self.X

    def Jacobi(self, n):
        X = [0] * len(self.B)
        X_coeff = [[0] * len(self.B)] * len(self.B)
        B_divided = []
        
        for i in range(len(self.B)):
            values = []
            for j in range(len(self.B)):
                values.append(self.A[i][j] / self.A[i][i])
            X_coeff[i] = [val if self.B[i] >= 0 else -val  for val in values]
            
            B_divided.append(self.B[i] / self.A[i][i])
        
        for _ in range(n):
            X_c = copy.deepcopy(X)
            for j in range(len(X)):
                x_sum = np.sum(np.asarray(X_c[:j]) * np.asarray(X_coeff[j][:j])) + np.sum(np.asarray(X_c[j+1:]) * np.asarray(X_coeff[j][j+1:]))
                b = B_divided[j]
                #print("sum = ", x_sum)
                #print("b = ", b)
                #print("X =", X[j])
                if X[j] >= 0 and b >= 0:
                    x_sum *= -1
                elif X[j] < 0 and b >= 0:
                    b *= -1
                elif X[j] >= 0 and b < 0:
                    x_sum *= -1
                #print("AFTER b = ", b, "\n")
                X[j] = b + x_sum
            
            print(X)
        return X

    def count(self):
        for row in range(len(self.A) -1, -1, -1):
            self.A[row, :row] = np.multiply(self.A[row, :row], self.X[:row].T)
            self.A[row, row + 1:] = np.multiply(self.A[row, row + 1:], self.X[row + 1:].T)
            values = np.float64(np.sum(self.A[row]) - self.A[row, row])  
            values *= -1
            values += np.float64(self.B[row])
            self.X[row] = np.float64(values / self.A[row, row])
            
        self.A = self.temp_A
        self.B = self.temp_B

        return self.X
    
    def calculation_A_B(self, i):
        self.B[i] = np.float64(self.B[i] + (self.B[self.row - 1] * (-self.A[i, self.row - 1] / self.A[self.row - 1, self.row - 1])))
        temp = self.A[i , self.row - 1]
        for j in range(len(self.A)):
            self.A[i, j] = np.float64((self.A[i, j] + (self.A[self.row - 1, j] * (-temp / self.A[self.row - 1, self.row - 1]))))



m_cal = matrix_calc([[4, -1, -0.2, 2], [-1, 5, 0, -2], [0.2, 1, 10, -1], [0, -2, -1, 4]], [30, 0, -10, 5])
print("\n", m_cal.Jacobi(5))