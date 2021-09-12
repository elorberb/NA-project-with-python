"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable



class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # replace this line with your solution
        def intersections_lambda(f1: callable, f2: callable, a: float, b: float, maxerr=0.001):

            def deriv(f, x):
                h = 0.00000000001
                top = f(x + h) - f(x)
                bottom = h
                ans = top / bottom
                return ans

            def newt_safe(f1, f2, a, b, maxerr):
                z = (a + b) / 2
                for j in range(10):  # using a mix of Newton Raphson and Bisection
                    f_z = f1(z) - f2(z)
                    Df_z = deriv(f1, z) - deriv(f2, z)
                    f_a = f1(a) - f2(a)
                    if (np.abs(f_z) < maxerr):
                        return z
                    if (Df_z == 0):  # Cant use Newton - Raphson
                        if f_a * f_z < 0:
                            b = z
                        else:
                            a = z
                        z = (a + b) / 2
                    else:
                        z_1 = z - f_z / Df_z
                        if (z_1 > a and z_1 < b):
                            if f_a * f_z < 0:
                                b = z_1
                            else:
                                a = z_1
                            z = z_1
                return None

            X = set()

            if a == b:
                return
            if a > b:
                c = a
                a = b
                b = c
            intervals = np.linspace(a, b, (b - a) * 100)

            for i in range(0, len(intervals) - 1, 1):  # iterating over intervals to find all roots
                low = intervals[i]
                high = intervals[i + 1]
                f_low = f1(low) - f2(low)
                f_high = f1(high) - f2(high)
                Df_low = deriv(f1, low) - deriv(f2, low)
                Df_high = deriv(f1, high) - deriv(f2, high)
                if f_low == 0:  # root is found
                    X.add(low)
                    continue

                if f_high == 0:  # root is found
                    X.add(high)
                    continue
                if (f_low * f_high > 0) and (Df_low * Df_high > 0):  # No roots in interval
                    continue
                else:
                    z = newt_safe(f1, f2, low, high, maxerr)
                    if z is not None:
                        X.add(z)

            return X

        def intersections_numpy(f1: callable, f2: callable, a: float, b: float, maxerr=0.001):
            def newt_safe(f, a, b, maxerr):
                Df = np.poly1d.deriv(f)
                z = (a + b) / 2
                for j in range(10):  # using a mix of Newton Raphson and Bisection
                    if (np.abs(f(z)) < maxerr):
                        return z
                    if (Df(z) == 0):  # Cant use Newton - Raphson
                        if f(a) * f(z) < 0:
                            b = z
                        else:
                            a = z
                        z = (a + b) / 2
                    else:
                        z_1 = z - f(z) / Df(z)
                        if (z_1 > a and z_1 < b):
                            if f(a) * f(z) < 0:
                                b = z_1
                            else:
                                a = z_1
                            z = z_1
                return None

            X = set()
            f = f1 - f2
            if a == b:
                return
            if a > b:
                c = a
                a = b
                b = c
            intervals = np.linspace(a, b, (b - a) * 100)
            df = np.poly1d.deriv(f)
            for i in range(0, len(intervals) - 1, 1):  # iterating over intervals to find all roots
                low = intervals[i]
                high = intervals[i + 1]
                if (f(low) == 0):  # root is found
                    X.add(low)
                    continue
                if (f(high) == 0):  # root is found
                    X.add(high)
                    continue
                if (f(low) * f(high) > 0) and (df(low) * df(high) > 0):  # No roots in interval
                    continue
                else:
                    z = newt_safe(f, low, high, maxerr)
                    if (z is not None):
                        X.add(z)

            return X
        if type(f1) == np.poly1d and type(f2) == np.poly1d:
            return intersections_numpy(f1,f2,a,b,maxerr)
        else:
            return intersections_lambda(f1,f2,a,b,maxerr)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        f1 = lambda x: -x ** 2 + 1
        f2 = lambda x: x ** 2 - 1

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly2(self):  # tangent point
        ass2 = Assignment2()

        f1 = np.poly1d([1, 10, 27, 0, -57, -30, 29, 21])
        f2 = np.poly1d([0, 0, 1])

        X = ass2.intersections(f1, f2, -10, 5, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly3(self):
        ass2 = Assignment2()

        f1 = np.poly1d([1, -10, 26])
        f2 = np.poly1d([0, 0, 1])

        X = ass2.intersections(f1, f2, -7, 6, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
