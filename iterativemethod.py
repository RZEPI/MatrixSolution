from mymatrix import *


class IterativeMethod(MyMatrix):
    MAX_ITERS = 1000
    NORM = 10**(-9)

    def __init__(self, x1, x2, x3, N=903):
        super().__init__(x1, x2, x3, N)
        self.iterations = 0

    def prepare_params(self):
        """Prepares parametrs for iterative method
            order:
             - r - results for following iteration
             - D - diagonal matrix
             - L - lower matrix
             - U - upper matrix
             - residuum - value needed for stop an algorithm
             - norms - norm of residuum for the following iteration"""

        self.result = np.ones(self.N)
        self.D = self.diag()
        self.L, self.U = self.tril_and_triu()
        self.residuum = 1
        self.norms = [1]

    def compute_result(self, method):
        start = time()
        self.prepare_params()
        method.define_expressions()
        while self.norms[self.iterations] > self.NORM:
            method.compute_iteration()
            self.residuum = (self.matrix @ self.result) - self.b_vec
            self.iterations += 1
            self.norms.append(super().norm())

            if self.iterations > self.MAX_ITERS:
                self.iterations = math.inf
                break
        end = time()
        self.time_of_computations = end - start
        self.norms = self.norms[1:]

    def solve(self, matrix, vector):
        super().solve(matrix, vector)

    def diag(self):
        diag = np.zeros(self.matrix.shape, dtype=np.double)
        for i in range(self.matrix.shape[0]):
            diag[i, i] = self.matrix[i, i]
        return diag

    def diag_flat(self):
        diag = np.zeros(self.matrix.shape[0], dtype=np.double)
        for i in range(self.matrix.shape[0]):
            diag[i] = self.matrix[i, i]
        return diag

    def tril_and_triu(self):
        lower = np.zeros(self.matrix.shape, dtype=np.double)
        upper = np.zeros(self.matrix.shape, dtype=np.double)
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[0]):
                if i < j:
                    upper[i, j] = self.matrix[i, j]

                if i > j:
                    lower[i, j] = self.matrix[i, j]
        return lower, upper

    def print_result(self, show_matrix=False):
        output = f"The {self.name}'s method ended in {self.time_of_computations} seconds"

        if self.iterations == math.inf:
            output += f".\nMethod doesn't converage."
        else:
            output += f" in {self.iterations} iterations."

        if show_matrix:
            output += f"\nResult:{self.result}"

        print(output)

    def generate_plot(self):
        fig, ax = plt.subplots()
        if self.iterations != math.inf:
            idx = [n for n in range(self.iterations)]
            ax.plot(idx, self.norms, 'k-')
        else:
            idx = [n for n in range(self.MAX_ITERS+1)]
            ax.semilogy(idx, self.norms, 'k-')

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residuum norm in following iteration")
        ax.set_title(f"Residuum norm - {self.name}")

        plt.savefig(f'{self.name}_method_residuum_norm.png')
        plt.show(block=False)
        plt.pause(2)
        plt.close()
