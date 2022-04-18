import sympy as sym
import numpy as np
from scipy.optimize import fsolve

def myFunction(z,*args):
   x = z[0]
   y = z[1]
   F = np.empty((2))
   F[0] = eval(args[0])
   F[1] = eval(args[1])
   return F
def call_critical_point(x_der,y_der):

    zGuess = np.array([1,1])
    x_der = "27*x**2 - 6*x*y + 16*x + 5*y**2 - y - 1"
    y_der =  "-3*x**2 + 10*x*y - x - 3*y**2 + 6*y + 3"
    z = fsolve(myFunction,zGuess,args=(x_der,y_der))
    print(z)

    zGuess = np.array([-1,-1])
    x_der = "27*x**2 - 6*x*y + 16*x + 5*y**2 - y - 1"
    y_der =  "-3*x**2 + 10*x*y - x - 3*y**2 + 6*y + 3"
    z = fsolve(myFunction,zGuess,args=(x_der,y_der))
    print(z)