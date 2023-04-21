from mymatrix import MyMatrix
from gaussseidel import GaussSeidel
from jacoby import Jacoby

e = 7

def solve_matrix(data):
    matrix = MyMatrix(*data)
    matrix.compute_result()
    # matrix.print_components()
    matrix.print_result()

    gauss_seidel = GaussSeidel(*data)
    gauss_seidel.compute_result()
    gauss_seidel.generate_plot()
    gauss_seidel.print_result()

    jacoby = Jacoby(*data)
    jacoby.compute_result()
    jacoby.generate_plot()
    jacoby.print_result()


solve_matrix((5+e, -1, -1))


solve_matrix((3, -1, -1))

