import sympy as sym
import numpy as np
from sympy import *                   # load all math functions
init_printing( use_latex='mathjax' )  # use pretty math output

from scipy.optimize import fsolve

def poly(x,y,n,p):
    ph_len = len(p)
    poly_list = []
    counter = 0
    str1 = ""
    for nc in range(n,-1,-1):
        for i in range(n+1):
            if nc-i >= 0:
                if p[counter] == "0":
                    coeff = "3"
                else :
                    coeff = p[counter]
                if counter == ph_len -1 :

                    poly_list.append(coeff + " * " + x + "**" + str(nc - i) + " * " + y + "**" + str(i))
                elif  counter % 2 == 0:
                    poly_list.append(coeff+ " * "+ x+ "**"+ str(nc - i)+ " * "+ y+ "**"+ str(i))
                    poly_list.append(" - ")
                else :
                    poly_list.append(coeff+ " * "+ x+ "**"+ str(nc - i)+ " * "+ y+ "**"+ str(i) )
                    poly_list.append(" + ")
                counter += 1
    return str1.join(poly_list)


def myFunction(z, *args):
    x = z[0]
    y = z[1]
    F = np.empty((2))
    F[0] = eval(args[0])
    F[1] = eval(args[1])
    return F


def call_critical_point(x_der, y_der):
    zGuess = np.array([1, 1])
    x_der = "27*x**2 - 6*x*y + 16*x + 5*y**2 - y - 1"
    y_der = "-3*x**2 + 10*x*y - x - 3*y**2 + 6*y + 3"
    z = fsolve(myFunction, zGuess, args=(x_der, y_der))
    cp = z

    zGuess = np.array([-1, -1])
    x_der = "27*x**2 - 6*x*y + 16*x + 5*y**2 - y - 1"
    y_der = "-3*x**2 + 10*x*y - x - 3*y**2 + 6*y + 3"
    z = fsolve(myFunction, zGuess, args=(x_der, y_der))
    cn = z

    return cp,cn

def Get_Critical_Point(p,zp,zn):
    cp_x = []
    cp_y = []
    eqtn = []
    x, y = symbols('x y')
    x_derivative = diff(p,x)
    print("The x derivative is :")
    print(x_derivative)

    x_eqtn = str(x_derivative)

    y_derivative = diff(p,y)
    print("The y derivative is :")
    print(y_derivative)

    y_eqtn = str(y_derivative)

    cpp,cpn = call_critical_point(x_eqtn,y_eqtn)

    return cpp, cpn

def maxima_minima_check(p,xy_val):
    x, y = symbols('x y')

    x_derivative = str(diff(p, x))
    xx_derivative = str(diff(x_derivative, x))
    xy_derivative = str(diff(x_derivative, y))
    y_derivative = str(diff(p, y))
    yy_derivative = str(diff(y_derivative,y))

    x = xy_val[0]
    y = xy_val[1]

    if eval(xx_derivative) * eval(yy_derivative) - (eval(xy_derivative)**2) < 0:
        print("The Polynomial {} is having Saddle Point at x = {} amd y = {}".format(p, x, y))
    else:
        if eval(xx_derivative) < 0 and eval(yy_derivative) <0:
            print("The Polynomial {} is having Maxima at x = {} amd y = {}".format(p, x, y))
        elif eval(xx_derivative) > 0 and eval(yy_derivative) >0:
            print("The Polynomial {} is having Minima at x = {} amd y = {}".format(p, x, y))
        else :
            print("The Polynomial {} is having neither Minima nor Maxima at x = {} amd y = {}. \n More advanced methods are required to classify the stationary point properly.".format(p, x, y))




if __name__ == "__main__":
    p = list(input('Enter your phone number: '))
    poly_list = poly("x","y",3,p)
    print("The generated polynomial is :\n")
    print(poly_list)
    zGuess_positive = np.array([1, 1])
    zGuess_negative = np.array([-1, -1])
    cpp, cpn = Get_Critical_Point(poly_list,zGuess_positive,zGuess_negative)
    print("The Critical Points are \n")
    print(cpp)
    print(cpn)

    maxima_minima_check(poly_list,cpp)

    maxima_minima_check(poly_list,cpn)


