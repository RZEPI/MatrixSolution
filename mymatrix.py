import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time


class MyMatrix:
    f = 1
    name = "direct method"

    def __init__(self, x1, x2, x3, N=903):
        self.N = N
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

        self.generate_matrix()
        self.generate_b_vector()

    def generate_matrix(self):
        """Generates A matrix"""
        self.matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            self.matrix[i, i] = self.x1
            if i > 0:
                self.matrix[i, i-1] = self.x2
            if i < self.N-1:
                self.matrix[i, i+1] = self.x2
            if i > 1:
                self.matrix[i, i-2] = self.x3
            if i < self.N-2:
                self.matrix[i, i+2] = self.x3

    def generate_b_vector(self):
        """Generates b vector"""
        self.b_vec = np.zeros(self.N)
        for i in range(self.N):
            self.b_vec[i] = np.sin(i*(self.f+1))

    def compute_result(self):
        start = time()
        self.result = self.solve(self.matrix, self.b_vec)
        end = time()
        self.time_of_computations = end - start

    def solve(self, matrix, vector):
        n = matrix.shape[0]

        x = np.zeros((n,1))

        vector_copy = deepcopy(vector)
        matrix_copy = deepcopy(matrix)

        for i in range(n-1, -1, -1):
            for j in range(i+1, n):
                vector_copy[i] = vector_copy[i] - matrix_copy[i,j] * x[j]
            x[i] = vector_copy[i] / matrix_copy[i,i]

        return x

    def norm(self):
        sum = 0
        for i in self.residuum:
            sum += i**2
        return math.sqrt(sum)

    def print_components(self):
        print(f"Matrix:\n{self.matrix}\nVector:\n{self.b_vec}")

    def print_result(self, show_matrix=False):
        output = f"The {self.name} method ended\n"
        if show_matrix:
            output += f"Result:{self.result}"

        print(output)
