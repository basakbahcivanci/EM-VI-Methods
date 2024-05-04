import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp, pi, log


def generate_data(n, k, prior_mean, prior_var, seed=None):
    if seed is not None:
        np.random.seed(seed)
    p = len(prior_mean) # dimension of X_1:n's & therefore Mu_1:k's

    # generate k mixture locations from the prior distribution,
    locs = np.random.multivariate_normal(mean=prior_mean, cov=np.diag([prior_var] * p), size=k)

    # generate the data
    obs = np.zeros((n, p))
    z = np.zeros(n, dtype=int)
    for i in range(n):
        # draw the mixture components uniformly
        z[i] = np.random.choice(range(k), size=1)
        # draw the observation from the corresponding mixture location
        obs[i, :] = np.random.multivariate_normal(mean=locs[z[i]], cov=np.diag([1] * p), size=1) # the variables are uncorrelated and have a variance of 1.
    true_phi = np.array([[float(value == 0), float(value == 1)] for value in z])
    return {'locs': locs, 'z': z, 'obs': obs, 'true_phi': true_phi}
def plot_ggm(obs, locs, z, iter=None):

    plt.clf()

    # add iteration number to the plot if it's provided
    if iter is not None:
        plt.text(np.min(obs[:, 0]), np.max(obs[:, 1]), f"Iteration {iter}", fontsize=12, verticalalignment='top')
    else:
        plt.text(np.min(obs[:, 0]), np.max(obs[:, 1]), "Simulated Data", fontsize=12, verticalalignment='top')

    # plot observations with colors based on z
    plt.scatter(obs[:, 0], obs[:, 1], c=z, cmap='viridis', s=30, alpha=0.75)

    # plot mixture components
    plt.scatter(locs[:, 0], locs[:, 1], c='red', s=90, marker='X', label='Mixture Components')

    # add legend and labels, position it to the right top
    plt.legend(loc="upper right")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Mixture Components and Observations')

    plt.pause(1)
    # show plot
    plt.show()

def ELBO_update(obs, var_mu, var_sigma, phi, mu0, var0, varX):
    N = obs.shape[0]  # sample size
    P = obs.shape[1]  # dimension
    K = mu0.shape[0]  # number of mixture components
    a = 0
    for k in range(K):
        a -= P * log(2 * pi * var0) / 2
        b_sum = 0
        for p in range(P):
            b = (1 / (2 * var0)) * (var_sigma[k] + var_mu[k, p] ** 2 - 2 * mu0[k] * var_mu[k, p] + mu0[k] ** 2)
            b_sum += b
        a -= b_sum
        a += (P * log(2 * pi * var_sigma[k]) + P) / 2
    c = 0
    for n in range(N):
        c -= log(K)
        d_sum = 0
        for k in range(K):
            d = phi[n, k] * log(phi[n, k] + 1e-50)
            d += phi[n, k] * log(2 * pi * varX)
            e_sum = 0
            for p in range(P):
                e = (1 / 2 * varX) * (- obs[n, p] ** 2 + 2 * obs[n, p] * var_mu[k, p] - var_sigma[k] - var_mu[k, p] ** 2)
                e_sum += e
            e_sum *= phi[n, k]
            d += e_sum
            d_sum += d
        c += d_sum
    elbo = a + c

    return elbo

def gmm_VI(data, mu0, var0, varX, epsilon, max_iter):
    obs = data['obs']
    N = obs.shape[0]
    P = obs.shape[1]
    K = 2

    # initialize the variational parameters: means and related variance (which is fixed hyperparameter)
    var_mu = np.random.multivariate_normal(mean=np.mean(obs, axis=0), cov=np.diag([var0] * P), size=K)
    var_sigma = np.full(K, var0).astype(float)

    # initialize c_i's as uniform (1/k).
    phi = np.full((N, K), 1 / K).astype(float)

    elbo_init = ELBO_update(obs, var_mu, var_sigma, phi, mu0, var0, varX)
    # Print ELBO and iteration
    print(f"Iteration 0: ELBO = {elbo_init}")
    # convergence criterion
    epsilon = 1e-25
    # initialize iteration counter
    iteration = 0

    # initializing a list to store elbo_new values
    elbo_new_list = []
    elbo_new_list.append(elbo_init)
    # initialize elbo_old
    elbo_old = elbo_init

    while True:  # break based on ELBO or iteration criteria
        # update q( φ_i = k) or update phi for each n and k
        for n in range(N):
            for k in range(K):
                a = 0
                for p in range(P):
                    a += (1 / varX) * (obs[n, p] * var_mu[k, p] - (var_sigma[k] + var_mu[k, p] ** 2) / 2)
                phi[n, k] = exp(a)
        # normalizing phi to have component membership probabilities
        phi = phi / phi.sum(axis=1)[:, None]

        # update mₖ and sₖ² OR update var_mu and var_sigma for each K
        for k in range(K):
            var_sigma[k] = 1 / ((1 / var0) + np.sum(phi[:, k]))
            b = (1 / var0) + np.sum(phi[:, k])
            for p in range(P):
                c = 0  # for each p
                for n in range(N):
                    c += phi[n, k] * obs[n, p]
                var_mu[k, p] = c * (1 / b)
        # update ELBO
        elbo_new = ELBO_update(obs, var_mu, var_sigma, phi, mu0, var0, varX)

        # Print ELBO and iteration
        print(f"Iteration {iteration}: ELBO = {elbo_new}")
        print(f"epsilon = {elbo_new - elbo_old}")
        elbo_new_list.append(elbo_new)  # append new elbo  to the list

        # class assignments ; updated in each iteration
        z = np.argmax(phi, axis=1)

        # plot in each iteration
        plot_ggm(obs, var_mu, z, iter=iteration+1)

        # check loop-break conditions
        if iteration > 0 and (abs(elbo_new - elbo_old)) <= epsilon:
            break
        if iteration == max_iter:
            break

        # update elbo_old for the next iteration
        elbo_old = elbo_new
        iteration += 1

    # return the list of elbo_new values
    return elbo_new_list, z, var_mu, phi




# prior mean vectors and prior variance for mixture locations
prior_mean = [0,10]
prior_var = 1 # for each dimension

data = generate_data(n=200, k=2, prior_mean=prior_mean, prior_var=prior_var, seed=2024)
plot_ggm(data['obs'], data['locs'], data['z'])

# priors for μₖ
var0 = 1
mu0 = np.mean(data['obs'], axis=0)
# prior sigma for X_i | c_i,M
varX = 1

# test it!
elbo_list, z, var_mu, phi  = gmm_VI(data, mu0, var0, varX, epsilon=1e-10, max_iter=25)

