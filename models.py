import numpy as np
print("defining the linear and quadratic models") 

def linear_prediction(beta, t):
    return beta[0] + beta[1] * t

def quadratic_prediction(beta, t):
    return beta[0] + beta[1] * t + beta[2] * t**2
print("calculating the error from the two models")
def linear_error (beta, t, y):
    return np.sum((y - linear_prediction(beta, t))**2)

def quadratic_error(beta, t, y):
    return np.sum((y - quadratic_prediction(beta, t))**2)
