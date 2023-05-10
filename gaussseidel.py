from iterativemethod import *


class GaussSeidel(IterativeMethod):
    name = "GaussSeidel's"

    def __init__(self, x1, x2, x3, N=903):
        super().__init__(x1, x2, x3, N)

    def define_expressions(self):
        pass

    def compute_iteration(self):
        for i in range(self.matrix.shape[0]):
            self.result[i] = (self.b_vec[i] - (self.matrix[i, :i] @ self.result[:i]) - (
                self.matrix[i, (i+1):] @ self.result[i+1:])) / self.matrix[i, i]

    def compute_result(self):
        super().compute_result(self)
