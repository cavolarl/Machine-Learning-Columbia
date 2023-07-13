import numpy as np

### Written by Carl Lavo
### Uni: cgl2131
### E-mail: cgl2131@columbia.edu
### COMS 4771 Machine Learning

# Our gradient descent function
def gradient_descent(f, compute_gradient, x_0, eta, max_iter=1000, tol=1e-6):
    x = x_0
    for _ in range(max_iter):
        grad = compute_gradient(x)
        x_next = x - eta * grad
        if np.abs(f(x_next) - f(x)) < tol:
            break
        x = x_next
    return x

# Here we define the function f(x)
def f(x):
    return (x-4)**2+2*np.exp(x)

# Here we define the gradient of f(x)
def compute_gradient(x):
    return 2*(x-4)+2*np.exp(x)

# Here we run the algorithm
x_0 = 0
eta = 0.1
x_min = gradient_descent(f, compute_gradient, x_0, eta)
print(f"The minimum of f(x) is: {x_min}")
print(f"The value of f(x) at the minimum is: {f(x_min)}")
