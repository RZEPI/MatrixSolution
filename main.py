from mymatrix import MyMatrix
from gaussseidel import GaussSeidel
from jacoby import Jacoby

e = 7
matrix = MyMatrix(5+e, -1, -1)
matrix.compute_result()
# matrix.print_components()
matrix.print_result()

gauss_seidel = GaussSeidel(5+e, -1, -1)
gauss_seidel.compute_result()
gauss_seidel.print_result()

jacoby = Jacoby(5+e, -1, -1)
jacoby.compute_result()
jacoby.print_result()