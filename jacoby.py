from iterativemethod import *


class Jacoby(IterativeMethod):
    name = "Jacoby's"

    def __init__(self, x1, x2, x3, N=903):
        super().__init__(x1, x2, x3, N)

    def define_expressions(self):
        self.expression1 = -np.linalg.solve(self.D, (self.L + self.U))
        self.expression2 = np.linalg.solve(self.D, self.b_vec)

    def compute_iteration(self):
        self.result = (self.expression1 @ self.result) + self.expression2

    def compute_result(self):
        super().compute_result(self)
