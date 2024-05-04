from math import sqrt, exp, pi, log
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for plot
import random
from random import uniform
from scipy.stats import multivariate_normal as norm

def generate_sigma(K, D):
    sigma = np.zeros(shape=(K, D, D))

    for k in range(K):
        # initialize sigma (invertible)
        sigma_k = np.random.uniform(1, 2, (D, D)) + np.random.uniform(1, 2, (D)) * np.diag((D, D))
        sigma[k, :, :] = sigma_k+sigma_k.T

    return sigma

def generate_data(n, k,  w, prior_mean, prior_var, seed=None):
    # set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    dim = len(prior_mean[0])  # Dimensionality of the data

    # initialize arrays
    samples = np.empty((n, dim))
    z = np.empty((n, 1))

    # generate samples
    for iter in range(n):
        # get random number to select the mixture component with probability according to mixture weights
        DrawComponent = random.choices(range(k), weights=w, cum_weights=None, k=1)[0]
        # draw sample from selected mixture component
        DrawSample = np.random.multivariate_normal(prior_mean[DrawComponent], prior_var[DrawComponent], 1)
        # store results
        z[iter] = DrawComponent
        samples[iter, :] = DrawSample

    return samples, z

def plot_ggm(obs, prior_mean, z, iter = None):

    locs = prior_mean
    locs = np.array(locs)
    plt.clf()

    # add iteration number to the plot if it's provided
    if iter is not None:
        plt.text(np.min(obs[:, 0]), np.max(obs[:, 1]), f"Iteration {iter}", fontsize=12, verticalalignment='top')
    else:
        plt.text(np.min(obs[:, 0]), np.max(obs[:, 1]), "Real Simulated Data", fontsize=12, verticalalignment='top')


    # plot observations with colors based on z
    plt.scatter(obs[:, 0], obs[:, 1], c=z, cmap='viridis', s=30, alpha=0.75)

    # plot mixture components
    plt.scatter(locs[:, 0], locs[:, 1], c='red', s=50, marker='X', label='Mixture Components')

    # add legend and labels
    plt.legend()
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Mixture Components and Observations')

    plt.pause(1)
    # Show plot
    plt.show()

def my_EM(X, K, epsilon, iteration):
    N = X.shape[0]  # sample size
    D = X.shape[1]  # dimension

    # initialize pi, mean, sigma
    pi_update = np.ones(shape=K) / K  # 1 by K
    mu_update = np.random.uniform(-2, 2, (K, D))  # for each dimension of K; 1 by D
    sigma_update = generate_sigma(K, D)
    epsilon = 1e-25
    likelihood_old = float('inf')  # initialize with a large value

    for i in range(iteration):
        # γ(Zk) the posterior probability when we observe x
        R = np.zeros(shape=(N, K))

        # E-Step: Computing posterior (Pr(z^i = k | x^i)) probabilities
        for k in range(K):
            R[:, k] = pi_update[k] * norm.pdf(X, mean=mu_update[k], cov=sigma_update[k])
        R = R / np.sum(R, axis=1, keepdims=True)

        # likelihood function
        likelihood = 0
        for k in range(K):
            likelihood += np.sum(R[:, k] * (np.log(pi_update[k]) + np.log(norm.pdf(X, mean=mu_update[k, :], cov=sigma_update[k, :, :]))))
        print('Input Gaussian {:}: likelihood = {:.10f}'.format("1", likelihood))

        # M-Step
        for k in range(K):
            mu_update[k, :] = (R[:, k].T @ X) / np.sum(R[:, k])
            sigma_update[k, :, :] = (X - mu_update[k, :]).T @ np.diag(R[:, k]) @ (X - mu_update[k, :]) / np.sum(R[:, k])
            pi_update[k] = np.sum(R[:, k]) / N

        z = np.argmax(R, axis=1) # class assignments ; updated in each iteration
        plot_ggm(X, mu_update, z, iter = i+1)


        # check loop-break conditions
        if abs(likelihood - likelihood_old) <= epsilon:
            break
        likelihood_old = likelihood

    return pi_update, mu_update, sigma_update, likelihood


# example:
n = 500
k = 3
prior_mean = [[0, 0], [-5, 5], [5, 5]]
prior_var = [np.array([[1, 0], [0, 1]]),
              np.array([[1, 0.8], [0.8, 1]]),
              np.array([[1, -0.8], [-0.8, 1]])]
# Mixture weights (non-negative, sum to 1)
w = [0.5, 0.25, 0.25]
samples, z = generate_data(n, k, w, prior_mean, prior_var)
# plot real data
plot_ggm(samples, prior_mean, z)

iter = 100
K = 3

# test it!
pi_update, mu_update, sigma_update, likelihood = my_EM(samples, K, epsilon=1e-25, iteration=iter)

