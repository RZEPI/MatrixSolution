import numpy as np
import math
import matplotlib.pyplot as plt

class MyMatrix:
    f = 1
    name = "numpy's solve"

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
        self.result =  np.linalg.solve(self.matrix, self.b_vec)

    def print_components(self):
        print(f"Matrix:\n{self.matrix}\nVector:\n{self.b_vec}")

    def print_result(self, show_matrix=False):
        output = f"The {self.name} method ended\n"

        if show_matrix:
            output += f"Result:{self.result}"

        print(output)
           

    