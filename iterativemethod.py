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
        self.D = np.diag(np.diag(self.matrix))
        self.L = np.tril(self.matrix, -1)
        self.U = np.triu(self.matrix, 1)
        self.residuum = 1
        self.norms = [np.linalg.norm(self.residuum)]

    def compute_result(self, method):
        self.prepare_params()
        method.define_expressions()
        start = time()
        while self.norms[self.iterations] > self.NORM:
            method.compute_iteration()
            self.residuum = (self.matrix @ self.result) - self.b_vec
            self.iterations += 1
            self.norms.append(np.linalg.norm(self.residuum))

            if self.iterations > self.MAX_ITERS:
                self.iterations = math.inf
                break
        end = time()
        self.time_of_computations = end - start
        self.norms = self.norms[1:]

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
        ax.set_title("Residuum norm")

        plt.savefig(f'{self.name}_method_residuum_norm.png')
        plt.show(block=False)
        plt.pause(2)
        plt.close()
