from mymatrix import MyMatrix
from mymatrix import plt
from gaussseidel import GaussSeidel
from jacoby import Jacoby
from ludecomposition import LUDecomp

e = 7

def print_times(times, names, iteration):
    plt.bar(names, times)
    plt.title("Time of computations for each method")
    plt.xlabel("Names of methods")
    plt.ylabel("Time in seconds")
    plt.savefig(f'time_of_computations{iteration}.png')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def solve_matrix(data, iteration):
    matrix = MyMatrix(*data)
    matrix.compute_result()
    matrix.print_result()

    gauss_seidel = GaussSeidel(*data)
    gauss_seidel.compute_result()
    gauss_seidel.generate_plot()
    gauss_seidel.print_result(show_matrix=True)

    jacoby = Jacoby(*data)
    jacoby.compute_result()
    jacoby.generate_plot()
    jacoby.print_result()

    ludecomp = LUDecomp(*data)
    ludecomp.compute_result()
    ludecomp.print_result()

    times = [matrix.time_of_computations, gauss_seidel.time_of_computations,
             jacoby.time_of_computations, ludecomp.time_of_computations]
    names = [matrix.name, gauss_seidel.name, jacoby.name, ludecomp.name]
    print_times(times, names, iteration)


iteration = 1
solve_matrix((5+e, -1, -1), iteration)

iteration += 1

solve_matrix((3, -1, -1), iteration)
