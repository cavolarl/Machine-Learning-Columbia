import numpy as np

def gradient_descent(f, compute_gradient, x_0, eta, max_iter=1000, tol=1e-6):
    x = x_0
    for _ in range(max_iter):
        grad = compute_gradient(x)
        x_next = x - eta * grad
        if np.abs(f(x_next) - f(x)) < tol:
            break
        x = x_next
    return x