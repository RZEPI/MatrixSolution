from mymatrix import *


class GaussSeidel(MyMatrix):
    NORM = 10**(-9)

    def __init__(self,x1,x2,x3, N=903):
        super().__init__(x1,x2,x3,N)

    def compute_result(self):
        r = np.ones(self.N)
        D = np.diag(np.diag(self.matrix))
        L = np.tril(self.matrix, -2)
        U = np.triu(self.matrix, 2)
        residuum = 1

        expression1 = D + L
        expression2 = np.linalg.solve(expression1, self.b_vec)
        self.iterations = 0
        
        while np.linalg.norm(residuum) > self.NORM:
            r = -np.linalg.solve(expression1, (U @ r)) + expression2
            residuum = (self.matrix @ r) - self.b_vec
            self.iterations += 1
    
    def print_results(self):
        print(f"The Gauss-Seidel method ended in {self.iterations} iterations\nResult:\n")
        print(self.result)
