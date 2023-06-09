from mymatrix import *
from copy import deepcopy


class LUDecomp(MyMatrix):
    name = "LU decomposition"

    def __init__(self, x1, x2, x3, N=903):
        super().__init__(x1, x2, x3, N)

    def prepare_params(self):
        """Prepares parametrs for iterative method
        order:
            - L - lower matrix
            - U - upper matrix
            - y - supp vector"""
        self.lu()

    def make_eye(self):
        eye = np.zeros(self.matrix.shape, dtype=np.double)
        for i in range(self.matrix.shape[0]):
            eye[i, i] = 1
        return eye

    def lu(self):
        n = self.matrix.shape[0]
        U = deepcopy(self.matrix)
        L = self.make_eye()

        for i in range(n):
            factor = U[i+1:, i] / U[i, i]
            L[i+1:, i] = factor
            U[i+1:] -= factor[:, np.newaxis] * U[i]

        self.U = U
        self.L = L

    def forward_substitution(self):
        n = self.L.shape[0]

        y = np.zeros_like(self.b_vec, dtype=np.double)

        y[0] = self.b_vec[0] / self.L[0, 0]

        for i in range(1, n):
            y[i] = (self.b_vec[i] - (self.L[i,:i] @ y[:i])) / self.L[i,i]

        self.y = y

    def back_substitution(self):
        n = self.U.shape[0]

        x = np.zeros_like(self.y, dtype=np.double)

        x[-1] = self.y[-1] / self.U[-1, -1]

        for i in range(n-2, -1, -1):
            x[i] = (self.y[i] - (self.U[i, i:] @ x[i:])) / self.U[i, i]

        self.result = x

    def compute_result(self):
        start = time()
        self.prepare_params()
        self.forward_substitution()
        self.back_substitution()
        end = time()
        self.time_of_computations = end - start
        print(f"LU decomposition ended in {self.time_of_computations} seconds")
        self.compute_residuum()

    def compute_residuum(self):
        self.residuum = self.matrix @ self.result - self.b_vec
        self.norm_residuum = super().norm()
        print(f"Its residuum norm: {self.norm_residuum}")
