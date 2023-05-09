from mymatrix import MyMatrix, np
from mymatrix import plt
from gaussseidel import GaussSeidel
from jacoby import Jacoby
from ludecomposition import LUDecomp

e = 7

def print_times(times, iteration):
    plt.bar(get_names(), times)
    plt.title("Time of computations for each method")
    plt.xlabel("Names of methods")
    plt.ylabel("Time in seconds")
    plt.savefig(f'time_of_computations{iteration}.png')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def get_names():
    return [MyMatrix.name, GaussSeidel.name, Jacoby.name, LUDecomp.name]

def print_comp_results(times, sizes):
    plt.plot(sizes, times['gauss'], 'r-', label='Gauss-Seidel')
    plt.plot(sizes, times['jacobian'], 'g-',label='Jacobi')
    plt.plot(sizes, times['ludecomp'], 'm-',label='LU Decomposition')
    plt.title('Time for different sizes of matrix')
    plt.legend(loc='upper left')
    plt.xlabel('Sizes of matrix')
    plt.ylabel('Time(s)')
    plt.show(block=False)
    plt.savefig("comparision_of_methods.png")
    plt.pause(2)
    plt.close()

def compare_times(data, iter_numbers):
    matrix_sizes = [ i for i in range(1000, iter_numbers*1000, 1000) ]
    matrix_sizes = [100, 500] + matrix_sizes
    times =  {'gauss':[], 'jacobian':[], 'ludecomp':[]}

    for size in matrix_sizes:
        gauss_seidel = GaussSeidel(*data, N=size)
        gauss_seidel.compute_result()
        times['gauss'].append(gauss_seidel.time_of_computations)

        jacoby = Jacoby(*data, N=size)
        jacoby.compute_result()
        times['jacobian'].append(jacoby.time_of_computations)

        ludemcomp = LUDecomp(*data, N=size)
        ludemcomp.compute_result()
        times['ludecomp'].append(ludemcomp.time_of_computations)

    print_comp_results(times, matrix_sizes)
    

def solve_matrix(data, iteration):
    matrix = MyMatrix(*data)
    matrix.compute_result()
    matrix.print_result(show_matrix=True)

    gauss_seidel = GaussSeidel(*data)
    gauss_seidel.compute_result()
    gauss_seidel.generate_plot()
    gauss_seidel.print_result()

    jacoby = Jacoby(*data)
    jacoby.compute_result()
    jacoby.generate_plot()
    jacoby.print_result()

    ludecomp = LUDecomp(*data)
    ludecomp.compute_result()
    ludecomp.print_result()

    times = [matrix.time_of_computations, gauss_seidel.time_of_computations,
                jacoby.time_of_computations, ludecomp.time_of_computations]
    print_times(times, iteration)


iteration = 1
data_a = (5+e, -1, -1)

solve_matrix(data_a, iteration)

iteration += 1

solve_matrix((3, -1, -1), iteration)
compare_times(data_a, 4)
