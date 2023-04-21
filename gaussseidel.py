from mymatrix import *


class GaussSeidel(MyMatrix):
    name = "GaussSeidel's"

    def __init__(self,x1,x2,x3, N=903):
        super().__init__(x1,x2,x3,N)

    def compute_result(self):
        r = np.ones(self.N)
        D = np.diag(np.diag(self.matrix))
        L = np.tril(self.matrix, -1)
        U = np.triu(self.matrix, 1)
        residuum = 1

        expression1 = D + L
        expression2 = np.linalg.solve(expression1, self.b_vec)
        self.iterations = 0
        
        while np.linalg.norm(residuum) > super().NORM:
            r = -np.linalg.solve(expression1, (U @ r)) + expression2
            residuum = (self.matrix @ r) - self.b_vec
            self.iterations += 1

            if self.iterations > super().MAX_ITERS:
                self.iterations = math.inf
                break
        self.result = r
    
