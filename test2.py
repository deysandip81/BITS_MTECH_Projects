import sympy as sym
import numpy as np
from scipy.optimize import fsolve

def myFunction(p,z):
   x = z[0]
   y = z[1]
   F = np.empty((2))

   x, y = symbols('x y')
   x_derivative = diff(p, x)
   print("The x derivative is :")
   print(x_derivative)

   F[0] = 27*x**2 - 6*x*y + 16*x + 5*y**2 - y - 1
   F[1] = -3*x**2 + 10*x*y - x - 3*y**2 + 6*y + 3
   return F

zGuess = np.array([1,1])
p = "9 * x**3 * y**0 - 3 * x**2 * y**1 + 5 * x**1 * y**2 - 1 * x**0 * y**3 + 8 * x**2 * y**0 - 1 * x**1 * y**1 + 3 * x**0 * y**2 - 1 * x**1 * y**0 + 3 * x**0 * y**1 - 6 * x**0 * y**0"

z = fsolve(myFunction,p,zGuess)
print(z)
zGuess = np.array([-1,-1])
z = fsolve(myFunction,p,zGuess)
print(z)