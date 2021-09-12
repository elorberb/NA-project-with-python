"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random
from assignment2 import Assignment2
import sympy as sym


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        # replace this line with your solution


        def composite_trapezoid_rule(f, a, b, n):

            x = np.linspace(a, b, n + 1)  # n+1 points because we use n+1 points
            y = f(x)
            y_right = y[1:]  # right endpoints
            y_left = y[:-1]  # left endpoints
            dx = (b - a) / n
            T = (dx / 2) * np.sum(y_right + y_left)
            return T

        def composite_simpson_rule(f, a, b, n):

            dx = (b - a) / n
            x = np.linspace(a, b, n + 1)
            y = f(x)
            S = dx/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])
            return S

        def simpson_rule2(f, a, b, n): # i've tried with another simpson rule to solve the hard case
            h = (b - a) / n
            x = np.linspace(a, b, n + 1)
            y = f(x)
            S = h / 48 * np.sum(
                17 * (y[0] + y[-1]) + 59 * (y[-2] + y[1]) + 43 * (y[-3] + y[2]) + 49 * (y[-4] + y[3]) + np.sum(
                    48 * y[4:-5:1]))
            return S

        intervals = n - 1
        if intervals % 2 == 0:
            result = composite_simpson_rule(f, a, b, intervals)
        else:
            result = composite_trapezoid_rule(f, a, b, intervals)

        return np.float32(result)



    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        # replace this line with your solution
        intersections = list(Assignment2.intersections(self, f1=f1, f2=f2, a=-1, b=100))
        intersections.sort()
        result = 0
        for i in range(0, len(intersections) - 1, 1):
            a = intersections[i]
            b = intersections[i + 1]
            z = (a + b) / 2
            if f1(z) > f2(z):
                f = f1 - f2
            else:
                f = f2 - f1
            result += self.integrate(f, a, b, 50)
        result = np.float32(result)
        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEqual(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_area_in_between(self):  # if i use i need to change the point area bounderies (-5,5)
        ass3 = Assignment3()
        f1 = np.poly1d([1, -2, 0, 1])
        f2 = np.poly1d([1, 0])
        r = ass3.areabetween(f1=f1, f2=f2)
        true_result = 2.76555
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_area_in_between2(self):  # if i use i need to change the point area bounderies (-3,5)
        ass3 = Assignment3()
        f1 = np.poly1d([2, 0, 10])
        f2 = np.poly1d([4, 16])
        r = ass3.areabetween(f1=f1, f2=f2)
        true_result = 64 / 3
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()
