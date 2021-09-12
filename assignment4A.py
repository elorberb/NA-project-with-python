"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # i had an error and this line solved it


class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        # replace these lines with your solution
        initial_T = time.time()

        def get_points(f, a, b, n):
            x = np.linspace(a, b, n)
            y_lst = f(x)
            y = np.array(y_lst)

            return x, y

        def build_vector(x, y, size, deg):
            if deg == size:
                y_n = y.sum()
            else:
                y_n = np.dot(y, (x ** (size - deg)))
            pol1 = np.zeros(size + 1)
            for i in range(size + 1):
                pol1[i] = (x ** (2 * size - deg - i)).sum()
            return pol1, y_n

        def build_coeff_matrix(f, a, b, n, d):
            x, y = get_points(f, a, b, n)
            deg = d
            b = np.array([])
            coeff_matrix = np.array([])
            for i in range(d + 1):
                x_n, y_n = build_vector(x, y, d, i)
                deg -= 1
                coeff_matrix = np.append(coeff_matrix, x_n)
                b = np.append(b, y_n)
            coeff_matrix = np.reshape(coeff_matrix, (d + 1, d + 1))
            return coeff_matrix, b

        def solve_coeff_matrix(coeff_matrix, b):
            A_inverse = np.linalg.inv(coeff_matrix)
            coeffs = A_inverse.dot(b)
            return coeffs

        def build_function_from_coeffs(coeffs):
            f = np.poly1d(coeffs)
            return f

        n = 100  # fisrt sample size
        while (time.time() - initial_T) + 0.2 < maxtime:  # while i still have time loop again
            Ax, B = build_coeff_matrix(f, a, b, n, d=d)
            if time.time() - initial_T + 0.2 >= maxtime:  # first break point
                break
            coeffs = solve_coeff_matrix(coeff_matrix=Ax, b=B)
            if time.time() - initial_T + 0.2 >= maxtime:  # second break point
                break
            result = build_function_from_coeffs(coeffs)
            if time.time() - initial_T + 0.2 >= maxtime:  # third break point
                break
            n += 200  # increasing sample size

        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEqual(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print("mse", mse)


if __name__ == "__main__":
    unittest.main()
