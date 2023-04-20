from mymatrix import *

class Jacoby(MyMatrix):
    name = "Jacoby's"
    def __init__(self,x1,x2,x3, N=903):
        super().__init__(x1,x2,x3,N)

    def compute_result(self):
        r = np.ones(self.N)
        D = np.diag(np.diag(self.matrix))
        L = np.tril(self.matrix, -1)
        U = np.triu(self.matrix, 1)
        residuum = 1
        self.iterations = 0

        expression1 = -np.linalg.solve(D, (L+U))
        expression2 = np.linalg.solve(D, self.b_vec)

        while np.linalg.norm(residuum) > super().NORM:
            r = (expression1 @ r) + expression2
            residuum = (self.matrix @ r) - self.b_vec
            self.iterations += 1
        self.result = r