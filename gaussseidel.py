from iterativemethod import *


class GaussSeidel(IterativeMethod):
    name = "GaussSeidel's"

    def __init__(self, x1, x2, x3, N=903):
        super().__init__(x1, x2, x3, N)

    def define_expressions(self):
        self.expression1 = self.D + self.L
        self.expression2 = np.linalg.solve(self.expression1, self.b_vec)

    def compute_iteration(self):
        self.result = -np.linalg.solve(self.expression1,
                            (self.U @ self.result)) + self.expression2

    def compute_result(self):
        super().compute_result(self)
