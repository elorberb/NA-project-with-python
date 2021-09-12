"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate2(self, b_diag: callable, a: float, b: float, n: int) -> callable:

        def get_points_from_func(f, a, b, n):
            x = np.linspace(a, b, n)
            y = f(x)
            points = np.array([np.array([x[i], y[i]]) for i in range(len(x))], dtype=np.float64)
            return points

        def build_coeff_matrix(n):
            C = 4 * np.identity(n)
            np.fill_diagonal(C[1:], 1)
            np.fill_diagonal(C[:, 1:], 1)
            C[0, 0] = 2
            C[n - 1, n - 1] = 7
            C[n - 1, n - 2] = 2
            return C

        def build_points_vector(points, n):
            P = [np.array([2 * (2 * points[i][0] + points[i + 1][0]), 2 * (2 * points[i][1] + points[i + 1][1])]) for i
                 in
                 range(n)]
            P[0] = np.array([points[0][0] + 2 * points[1][0], points[0][1] + 2 * points[1][1]])
            P[n - 1] = np.array([8 * points[n - 1][0] + points[n][0], 8 * points[n - 1][1] + points[n][1]])
            return P

        def get_diagonal_vectors(A):
            a = A.diagonal(-1)
            b = A.diagonal()
            c = A.diagonal(1)
            return a, b, c

        def thomas_algo(a, b, c, d):
            nf = len(d)  # number of equations
            ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
            for it in range(1, nf):
                mc = ac[it - 1] / bc[it - 1]
                bc[it] = bc[it] - mc * cc[it - 1]
                dc[it] = dc[it] - mc * dc[it - 1]

            xc = dc
            xc[-1] = dc[-1] / bc[-1]

            for il in range(nf - 2, -1, -1):
                xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

            return xc

        def get_Bi_points(nsplines, A, points):
            B = np.zeros(shape=(nsplines, 2))
            for i in range(nsplines - 1):
                B[i] = 2 * points[i + 1] - A[i + 1]
            B[-1] = (A[nsplines - 1] + points[nsplines]) / 2
            return B

        def get_cubic_poly(a, b, c, d):
            return lambda t: a * (1 - t) ** 3 + 3 * b * t * ((1 - t) ** 2) + 3 * c * (1 - t) * t ** 2 + d * t ** 3

        def get_bezier_curves(A, B, points, nsplines):
            funcs = np.array([get_cubic_poly(points[i], A[i], B[i], points[i + 1]) for i in range(nsplines)])
            ranges = np.array([np.array([points[i], points[i + 1]]) for i in range(nsplines)])
            return funcs, ranges

        if n == 1:
            return lambda x: b_diag(x)

        nsplines = n - 1
        input_points = get_points_from_func(b_diag, a, b, n)
        a_matrix = build_coeff_matrix(nsplines)
        P = build_points_vector(input_points, nsplines)
        a_diag, b_diag, c_diag = get_diagonal_vectors(a_matrix)
        a_solutions = thomas_algo(a_diag, b_diag, c_diag, P)
        B = get_Bi_points(nsplines, a_solutions, input_points)
        funcs, ranges = get_bezier_curves(a_solutions, B, input_points, nsplines)

        def g(x):
            for i in range(len(ranges)):
                interpolate_func = funcs[i]
                if ranges[i][0][0] <= x < ranges[i][1][0]:
                    t = (x - ranges[i][0][0]) / (
                            ranges[i][1][0] - ranges[i][0][0])
                    return interpolate_func(t)[1]

        return g

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        # replace this line with your solution to pass the second test

        def bezier3(P1, P2, P3, P4):
            M = np.array(
                [[-1, +3, -3, +1],
                 [+3, -6, +3, 0],
                 [-3, +3, 0, 0],
                 [+1, 0, 0, 0]],
                dtype=np.float32
            )
            P = np.array([P1, P2, P3, P4], dtype=np.float64)

            def f(t):
                T = np.array([t ** 3, t ** 2, t, 1], dtype=np.float64)
                return T.dot(M).dot(P)

            return f

        if n == 1:
            return lambda x: f(x)

        nsplines = n // 2 - 2
        dots = {}
        x_vals = np.linspace(a, b, n)

        func_intervals_range = []
        functions = []

        for i in range(0, nsplines): #organizing points of every spline
            x1 = x_vals[i * 2 + 1]
            if x1 in dots.keys():
                y1 = dots[x1]
            else:
                y1 = f(x1)
                dots[x1] = y1
            p0 = np.array([x1, y1])
            x2 = 2 * x_vals[i * 2 + 1] - x_vals[i * 2]
            if x2 in dots.keys():
                y2 = dots[x2]
            else:
                y2 = f(x2)
                dots[x2] = y2
            p1 = np.array([x2, y2])
            x3 = x_vals[i * 2 + 2]
            if x3 in dots.keys():
                y3 = dots[x3]
            else:
                y3 = f(x3)
                dots[x3] = y3
            p2 = np.array([x3, y3])
            x4 = x_vals[i * 2 + 3]
            if x4 in dots.keys():
                y4 = dots[x4]
            else:
                y4 = f(x4)
                dots[x4] = y4
            p3 = np.array([x4, y4])
            curve = bezier3(p0, p1, p2, p3)
            functions.append(curve)
            func_intervals_range.append((p0, p3))

        def g(x):
            for i in range(len(func_intervals_range)):#looping threw functions range list to find specific location
                interpolate_func = functions[i]
                if func_intervals_range[i][0][0] <= x < func_intervals_range[i][1][0]:
                    t = (x - func_intervals_range[i][0][0]) / (
                            func_intervals_range[i][1][0] - func_intervals_range[i][0][0]) #converting x to t
                    return interpolate_func(t)[1]

        result = g
        return result


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):
    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 300
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)



            ff = ass1.interpolate(f, -10, 10, 300 + 1)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate2(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

    def test_with_poly_restrict2(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = np.poly1d([1, 0, 0])
        ff = ass1.interpolate2(f, 1, 3, 1)

        yy = ff(4)
        print(yy)

    def test_with_poly1(self):
        T = time.time()

        ass1 = Assignment1()

        a = np.random.uniform(low=-1, high=1, size=10)
        f = np.poly1d(a)
        ff = ass1.interpolate(f, -10, 10, 10)
        fff = ass1.interpolate2(f, -10, 10, 10)
        xs = np.random.random(200)
        err1 = 0
        err2 = 0
        for x in xs:
            yy = ff(x)
            yyy = fff(x)
            y = f(x)
            err1 += abs(y - yyy)
            err2 += abs(y - yy)

        err1 = err1 / 200
        err2 = err2 / 200
        print("yyy error", err1)
        print("yy error", err2)


if __name__ == "__main__":
    unittest.main()
