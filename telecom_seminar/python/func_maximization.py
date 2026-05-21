import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

plt.style.use('dark_background')
np.random.seed(123)

# Initial values 
mu = -1
sigma = 0.1

# Learning rate 
eta = 0.005

# Number of iterations
n_it = 1000

# Number of samples per iteration
n_mc = 300


# Funtion to maximize
def func(x):
    # return -np.power((x-1), 2)
    return 5*np.abs( np.cos(10*np.pi*x)*np.exp(-np.power(x/0.5,2)) )

# Gradient of J(theta) w.r.t mu
def grad_mu(x):
    return func(x)*(x-mu)/(sigma**2)

# Get pdf
def gaussian(x):
    sig = max(0.001, sigma)
    aux = 0.5 * np.power(x-mu, 2)/sig**2
    return (1.0/np.sqrt(2*np.pi)/sig)*np.exp(-1.0*aux)

# 
x_plot = np.linspace(-1, 1, 1000)

# Run 
fig = plt.figure()
camera = Camera(fig)
for it in range(n_it):
    x = mu + sigma*np.random.randn(n_mc)
    mu += eta*grad_mu(x).mean()/np.abs(grad_mu(x)).mean()
    if it % 10 == 0:
        print("it = {}, mu={}, sigma={}, grad = {}".format(it, mu, sigma, grad_mu(x).mean()))
        p1 = plt.plot(x_plot, func(x_plot), 'r')
        p2 = plt.plot(x_plot, gaussian(x_plot), 'y--')
        plt.xlabel('x')
        camera.snap()

animation = camera.animate()
animation.save('temp.mp4')