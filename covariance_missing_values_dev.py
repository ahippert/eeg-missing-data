import numpy as np
#from generic_functions import tyler_estimator_covariance
from scipy.linalg import toeplitz
from scipy.linalg import lapack
import copy
import time
import permutations as perm
import gaps_gen as gg
import matplotlib.pyplot as plt

inds_cache = {}

def upper_triangular_to_symmetric(ut):
    n = ut.shape[0]
    try:
        inds = inds_cache[n]
    except KeyError:
        inds = np.tri(n, k=-1, dtype=np.bool)
        inds_cache[n] = inds
    ut[inds] = ut.T[inds]


def fast_positive_definite_inverse(m):
    cholesky, info = lapack.dpotrf(m)
    if info != 0:
        raise ValueError('dpotrf failed on input {}'.format(m))
    inv, info = lapack.dpotri(cholesky)
    if info != 0:
        raise ValueError('dpotri failed on input {}'.format(cholesky))
    upper_triangular_to_symmetric(inv)
    return inv

def tyler_estimator_covariance_normalisedet(𝐗, low_rank=False, rank=None, tol=1e-3, iter_max=50, init=None):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
                    and normalisation by determinant
                    Inputs:
                        * 𝐗 = a matrix of size p*N with each observation along column dimension
                        * tol = tolerance for convergence of estimator
                        * iter_max = number of maximum iterations
                        * init = Initialisation point of the fixed-point, default is identity matrix
                    Outputs:
                        * 𝚺 = the estimate
                        * δ = the final distance between two iterations
                        * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    if init is None:
        #𝚺 = np.eye(p) # Initialise estimate to identity
        𝚺 = (1/N)*𝐗@𝐗.conj().T # Initialise estimate to SCM
        𝚺 = p*𝚺/np.trace(𝚺)
    else:
        𝚺 = init
    iteration = 0

    τ=np.zeros((p,N))
    # Recursive algorithm
    while (δ>tol) and (iteration<iter_max):

        # Computing expression of Tyler estimator (with matrix multiplication)
        v = np.linalg.inv(np.linalg.cholesky(𝚺))@𝐗
        a = np.mean(v*v.conj(),axis=0)

        τ[0:p,:] = np.sqrt(np.real(a))
        𝐗_bis = 𝐗 / τ
        𝚺_new = (1/N) * 𝐗_bis@𝐗_bis.conj().T

        # Imposing det constraint: det(𝚺) = 1 DOT NOT WORK HERE
        # 𝚺 = 𝚺/(np.linalg.det(𝚺)**(1/p))

        # Performs Low Rank decomposition
        if low_rank:
            𝚺_new = estim_lowrank_covar(𝚺_new, rank)

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')
        iteration = iteration + 1

        # Updating 𝚺
        𝚺 = 𝚺_new

    # Calcul textures
    𝛕 = np.zeros(N)
    v = np.linalg.inv(np.linalg.cholesky(𝚺))@𝐗
    a = np.mean(v*v.conj(),axis=0)
    𝛕 = np.real(a)
    𝛕 = 𝛕*(np.linalg.det(𝚺)**(1/p))

    # Imposing det constraint: det(𝚺) = 1
    𝚺 = 𝚺/(np.linalg.det(𝚺)**(1/p))

    return 𝚺, 𝛕

def estim_cmpd_gaussian(Y, rank=5, lowrank=False, tol=1e-4, iter_max=30):
    """
    m : variable index where missing data start
    r : same for observation index
    """

    # Initialisation
    (p, N) = Y.shape

    # Get row indices where values are missing
    N_mis_indices = np.where(np.sum(np.isnan(Y), axis=0))[0]
    N_obs_indices = np.where(np.sum(np.isnan(Y), axis=0)==False)[0]
    N_mis = len(N_mis_indices)
    N_obs = len(N_obs_indices)

    # Get number of observed components where observations are missing
    M = p - np.sum(np.isnan(Y[:,N_mis_indices]), axis=0)

    # Permute missing data at the bottom of each vector
    ind = np.empty((N_mis,p), dtype=int)
    for i, i_mis in enumerate(N_mis_indices):
        Y[:,i_mis], ind[i] = perm.permut_row(Y[:,i_mis])

    # Form permutation matrices
    P = np.array([np.eye(p)[ind[i]].T for i in range(N_mis)])

    # Initialisation of Sigma and tau
    S = np.eye(p)
    tau = np.ones(N)

    # EM loop
    conv = 0
    LL, LL_full_estim, err = [], [], []
    delta_em = np.inf # Distance between two iterations
    while (delta_em>tol) and (conv<iter_max):

        # Init matrix B (block matrix which is part of the M-step solution)
        B = np.zeros((N_mis,p,p))
        C = np.zeros((N,p,p))

        # Compute covariance and normalize by number of observations
        for i_obs in N_obs_indices:
            C[i_obs] = Y[:,i_obs,np.newaxis] @ Y[np.newaxis,:,i_obs]

        # Fill matrix B
        for (i, i_mis), j in zip(enumerate(N_mis_indices), M):
            D = P[i]@S@P[i].T # Apply them on each side of covariance
            D_mm = D[j:,j:]
            D_mo = D[j:,:j]
            D_oo = D[:j,:j]
            D_om = D[:j,j:]
            inv_D_oo = np.linalg.inv(D_oo)

            cov = D_mm - D_mo@inv_D_oo@D_om
            mu_mis = D_mo@inv_D_oo@np.vstack(Y[:j,i_mis])
            B[i, :j, :j] = Y[:j,i_mis,np.newaxis] @ Y[np.newaxis,:j,i_mis] # Upper left block
            B[i, j:, j:] = tau[i_mis]*cov + mu_mis @ mu_mis.T # Lower right block
            C[i_mis] = P[i].T@B[i]@P[i]

        # Fixed point loop
        delta_fp = np.inf
        iteration = 0
        S_em = S
        while (delta_fp>tol) and (iteration<iter_max):

            # Compute fixed point covariance (S_fp)
            S_fp = np.zeros((p,p))
            for i in range(N):
                S_fp += C[i].T/np.trace(C[i]@np.linalg.inv(S_em))

            S_fp = (p/N)*S_fp

            # Performs Low Rank decomposition
            if lowrank: #and rank>=opt_rank:
                S_fp = estim_lowrank_covar(S_fp, rank)

            # Condition for stopping
            delta_fp = np.linalg.norm(S_fp - S_em, 'fro') / np.linalg.norm(S_em, 'fro')
            iteration = iteration + 1

            # Updating 𝚺
            S_em = S_fp

        # Compute delta Sigma
        delta_em = np.linalg.norm(S_em - S, 'fro') / np.linalg.norm(S, 'fro')
        #print(delta_em)
        err.append(delta_em)
        conv = conv + 1

        # Update Sigma and tau
        S = S_em

        # Using estimated S to compute tau_new
        for i in range(N):
            tau[i] = (1./p) * np.trace(C[i]@np.linalg.inv(S))
    tau = tau*(np.linalg.det(S)**(1/p))
    S = S/(np.linalg.det(S)**(1/p))
    #LL.append(compute_incomplete_log_likelihood_compound(X, S, tau, P, M))

    return (S, tau)#, LL)

def estim_cmpd_gaussian_V1(Y, rank=5, lowrank=False, tol=1e-4, iter_max=30):
    """
    m : variable index where missing data start
    r : same for observation index
    """

    # Initialisation
    (p, N) = Y.shape

    # Get row indices where values are missing
    N_mis_indices = np.where(np.sum(np.isnan(Y), axis=0))[0]
    N_obs_indices = np.where(np.sum(np.isnan(Y), axis=0)==False)[0]
    N_mis = len(N_mis_indices)
    N_obs = len(N_obs_indices)

    # Get number of observed components where observations are missing
    M = p - np.sum(np.isnan(Y[:,N_mis_indices]), axis=0)

    # Permute missing data at the bottom of each vector
    ind = np.empty((N_mis,p), dtype=int)
    ind_perm = np.empty((N_mis,p), dtype=int)
    for i, i_mis in enumerate(N_mis_indices):
        Y[:,i_mis], ind[i] = perm.permut_row(Y[:,i_mis])

    # Form permutation matrices
    P = np.array([np.eye(p)[ind[i]].T for i in range(N_mis)])

    # Initialisation of Sigma and tau
    S = np.eye(p)
    tau = np.ones(N)

    # EM loop
    conv = 0
    LL, LL_full_estim, err = [], [], []
    delta_em = np.inf # Distance between two iterations
    while (delta_em>tol) and (conv<iter_max):

        # Init matrix B (block matrix which is part of the M-step solution)
        B = np.zeros((N_mis,p,p))
        C = np.zeros((N,p,p))

        # Compute covariance and normalize by number of observations
        for i_obs in N_obs_indices:
            C[i_obs] = Y[:,i_obs,np.newaxis] @ Y[np.newaxis,:,i_obs]

        # Fill matrix B
        for (i, i_mis), j in zip(enumerate(N_mis_indices), M):
            D = P[i]@S@P[i].T # Apply them on each side of covariance
            D_mm = D[j:,j:]
            D_mo = D[j:,:j]
            D_oo = D[:j,:j]
            D_om = D[:j,j:]
            inv_D_oo = fast_positive_definite_inverse(D_oo)
            cov = D_mm - D_mo@inv_D_oo@D_om
            mu_mis = D_mo@inv_D_oo@np.vstack(Y[:j,i_mis])
            B[i, :j, :j] = Y[:j,i_mis,np.newaxis] @ Y[np.newaxis,:j,i_mis] # Upper left block
            B[i, j:, j:] = tau[i_mis]*cov + mu_mis @ mu_mis.T # Lower right block
            C[i_mis] = P[i].T@B[i]@P[i]

        # Fixed point loop
        delta_fp = np.inf
        iteration = 0
        S_em = S
        while (delta_fp>tol) and (iteration<iter_max):

            # Compute fixed point covariance (S_fp)
            S_fp = np.zeros((p,p))
            for i in range(N):
                S_fp += C[i].T/np.trace(C[i]@np.linalg.inv(S_em))

            S_fp = (p/N)*S_fp

            # Performs Low Rank decomposition
            if lowrank: #and rank>=opt_rank:
                S_fp = estim_lowrank_covar(S_fp, rank)

            # Condition for stopping
            delta_fp = np.linalg.norm(S_fp - S_em, 'fro') / np.linalg.norm(S_em, 'fro')
            iteration = iteration + 1

            # Updating 𝚺
            S_em = S_fp

        # Compute delta Sigma
        delta_em = np.linalg.norm(S_em - S, 'fro') / np.linalg.norm(S, 'fro')
        #print(delta_em)
        err.append(delta_em)
        conv = conv + 1

        # Update Sigma and tau
        S = S_em

        # Using estimated S to compute tau_new
        for i in range(N):
            tau[i] = (1./p) * np.trace(C[i]@np.linalg.inv(S))
    tau = tau*(np.linalg.det(S)**(1/p))
    S = S/(np.linalg.det(S)**(1/p))
    #LL.append(compute_incomplete_log_likelihood_compound(X, S, tau, P, M))

    return (S, tau)#, LL)

def estim_cmpd_gaussian_V2(Y, rank=5, lowrank=False, tol=1e-4, iter_max=30):
    """
    m : variable index where missing data start
    r : same for observation index
    """

    # Initialisation
    (p, N) = Y.shape

    # Get row indices where values are missing
    N_mis_indices = np.where(np.sum(np.isnan(Y), axis=0))[0]
    N_obs_indices = np.where(np.sum(np.isnan(Y), axis=0)==False)[0]
    N_mis = len(N_mis_indices)
    N_obs = len(N_obs_indices)

    # Get number of observed components where observations are missing
    M = p - np.sum(np.isnan(Y[:,N_mis_indices]), axis=0)

    # Permute missing data at the bottom of each vector
    ind = np.empty((N_mis,p), dtype=int)
    ind_perm = np.empty((N_mis,p), dtype=int)
    for i, i_mis in enumerate(N_mis_indices):
        ind_perm[i] = perm.permutation_index(Y[:,i_mis])
        Y[:,i_mis], ind[i] = perm.permut_row(Y[:,i_mis])

    # Form permutation matrices
    P = np.array([np.eye(p)[ind[i]].T for i in range(N_mis)])

    # Initialisation of Sigma and tau
    S = np.eye(p)
    tau = np.ones(N)

    # EM loop
    conv = 0
    LL, LL_full_estim, err = [], [], []
    delta_em = np.inf # Distance between two iterations
    while (delta_em>tol) and (conv<iter_max):

        # Init matrix B (block matrix which is part of the M-step solution)
        B = np.zeros((N_mis,p,p))
        C = np.zeros((N,p,p))

        # Compute covariance and normalize by number of observations
        for i_obs in N_obs_indices:
            C[i_obs] = Y[:,i_obs,np.newaxis] @ Y[np.newaxis,:,i_obs]

        # Fill matrix B
        for (i, i_mis), j in zip(enumerate(N_mis_indices), M):
            #D = P[i]@S@P[i].T # Apply them on each side of covariance
            D = S[ind_perm[i]][:,ind_perm[i]]
            D_mm = D[j:,j:]
            D_mo = D[j:,:j]
            D_oo = D[:j,:j]
            D_om = D[:j,j:]
            inv_D_oo = fast_positive_definite_inverse(D_oo)
            cov = D_mm - D_mo@inv_D_oo@D_om
            mu_mis = D_mo@inv_D_oo@np.vstack(Y[:j,i_mis])
            B[i, :j, :j] = Y[:j,i_mis,np.newaxis] @ Y[np.newaxis,:j,i_mis] # Upper left block
            B[i, j:, j:] = tau[i_mis]*cov + mu_mis @ mu_mis.T # Lower right block
            #C[i_mis] = P[i].T@B[i]@P[i]
            C[i_mis] = B[i][ind[i]][:,ind[i]]

        # Fixed point loop
        delta_fp = np.inf
        iteration = 0
        S_em = S
        while (delta_fp>tol) and (iteration<iter_max):

            # Compute fixed point covariance (S_fp)
            S_fp = np.zeros((p,p))
            for i in range(N):
                S_fp += C[i].T/np.trace(C[i]@np.linalg.inv(S_em))

            S_fp = (p/N)*S_fp

            # Performs Low Rank decomposition
            if lowrank: #and rank>=opt_rank:
                S_fp = estim_lowrank_covar(S_fp, rank)

            # Condition for stopping
            delta_fp = np.linalg.norm(S_fp - S_em, 'fro') / np.linalg.norm(S_em, 'fro')
            iteration = iteration + 1

            # Updating 𝚺
            S_em = S_fp

        # Compute delta Sigma
        delta_em = np.linalg.norm(S_em - S, 'fro') / np.linalg.norm(S, 'fro')
        #print(delta_em)
        err.append(delta_em)
        conv = conv + 1

        # Update Sigma and tau
        S = S_em

        # Using estimated S to compute tau_new
        for i in range(N):
            tau[i] = (1./p) * np.trace(C[i]@np.linalg.inv(S))
    tau = tau*(np.linalg.det(S)**(1/p))
    S = S/(np.linalg.det(S)**(1/p))
    #LL.append(compute_incomplete_log_likelihood_compound(X, S, tau, P, M))

    return (S, tau)#, LL)

# def estim_cmpd_gaussian_V2(Y, rank=5, lowrank=False, tol=1e-4, iter_max=30):
#     """
#     m : variable index where missing data start
#     r : same for observation index
#     """

#     # Initialisation
#     (p, N) = Y.shape

#     # Get row indices where values are missing
#     N_mis_indices = np.where(np.sum(np.isnan(Y), axis=0))[0]
#     N_obs_indices = np.where(np.sum(np.isnan(Y), axis=0)==False)[0]
#     N_mis = len(N_mis_indices)
#     N_obs = len(N_obs_indices)

#     # Get number of observed components where observations are missing
#     M = p - np.sum(np.isnan(Y[:,N_mis_indices]), axis=0)

#     # Permute missing data at the bottom of each vector
#     ind = np.empty((N_mis,p), dtype=int)
#     ind_perm = np.empty((N_mis,p), dtype=int)
#     for i, i_mis in enumerate(N_mis_indices):
#         ind_perm[i] = perm.permutation_index(Y[:,i_mis])
#         Y[:,i_mis], ind[i] = perm.permut_row(Y[:,i_mis])

#     # Initialisation of Sigma and tau
#     S = np.eye(p)
#     tau = np.ones(N)

#     # EM loop
#     conv = 0
#     LL, LL_full_estim, err = [], [], []
#     delta_em = np.inf # Distance between two iterations
#     while (delta_em>tol) and (conv<iter_max):

#         # Init matrix B (block matrix which is part of the M-step solution)
#         B = np.zeros((N_mis,p,p))
#         C = np.zeros((N,p,p))

#         # Compute covariance and normalize by number of observations
#         for i_obs in N_obs_indices:
#             C[i_obs] = Y[:,i_obs,np.newaxis] @ Y[np.newaxis,:,i_obs]

#         # Fill matrix B
#         for (i, i_mis), j in zip(enumerate(N_mis_indices), M):
#             D = S[ind_perm[i]][:,ind_perm[i]]
#             D_mm = D[j:,j:]
#             D_mo = D[j:,:j]
#             D_oo = D[:j,:j]
#             D_om = D[:j,j:]
#             inv_D_oo = np.linalg.inv(D_oo)
#             cov = D_mm - D_mo@inv_D_oo@D_om
#             mu_mis = D_mo@inv_D_oo@np.vstack(Y[:j,i_mis])
#             B[i, :j, :j] = Y[:j,i_mis,np.newaxis] @ Y[np.newaxis,:j,i_mis] # Upper left block
#             B[i, j:, j:] = tau[i_mis]*cov + mu_mis @ mu_mis.T # Lower right block
#             C[i_mis] = B[i][ind[i]][:,ind[i]]

#         # Fixed point loop
#         delta_fp = np.inf
#         iteration = 0

#         # Compute fixed point covariance (S_fp)
#         S_fp = np.zeros((p,p))
#         for i in range(N):
#             S_fp += C[i].T/np.trace(C[i]@np.linalg.inv(S))

#         S_fp = (p/N)*S_fp

#         # Performs Low Rank decomposition
#         if lowrank: #and rank>=opt_rank:
#             S_fp = estim_lowrank_covar(S_fp, rank)

#         # Updating 𝚺
#         S_em = S_fp

#         # Compute delta Sigma
#         delta_em = np.linalg.norm(S_em - S, 'fro') / np.linalg.norm(S, 'fro')
#         #print(delta_em)
#         err.append(delta_em)
#         conv = conv + 1

#         # Update Sigma and tau
#         S = S_em

#         # Using estimated S to compute tau_new
#         for i in range(N):
#             tau[i] = (1./p) * np.trace(C[i]@np.linalg.inv(S))
#     tau = tau*(np.linalg.det(S)**(1/p))
#     S = S/(np.linalg.det(S)**(1/p))
#     #LL.append(compute_incomplete_log_likelihood_compound(X, S, tau, P, M))

#     return (S, tau)#, LL)

def estim_cmpd_gaussian_V3(Y, rank=5, lowrank=False, tol=1e-4, iter_max=30):
    """
    m : variable index where missing data start
    r : same for observation index
    """

    # Initialisation
    (p, N) = Y.shape

    # Get row indices where values are missing
    N_mis_ind = np.where(np.sum(np.isnan(Y), axis=0))[0]
    N_obs_ind = np.where(np.sum(np.isnan(Y), axis=0)==False)[0]
    N_mis = len(N_mis_ind)
    N_obs = len(N_obs_ind)

    # Get number of observed components where observations are missing
    M = p - np.sum(np.isnan(Y[:,N_mis_ind]), axis=0)

    # Permute missing data at the bottom of each vector
    ind = np.empty((N_mis,p), dtype=int)
    ind_perm = np.empty((N_mis,p), dtype=int)
    for i, i_mis in enumerate(N_mis_ind):
        ind_perm[i] = perm.permutation_index(Y[:,i_mis])
        Y[:,i_mis], ind[i] = perm.permut_row(Y[:,i_mis])

    # Initialisation of Sigma and tau
    S = np.eye(p)
    tau = np.ones(N)

    # EM loop
    # Init matrix B (block matrix which is part of the M-step solution)
    B = np.zeros((N_mis,p,p))
    C = np.zeros((N,p,p))
    conv = 0
    LL, LL_full_estim, err = [], [], []
    delta_em = np.inf # Distance between two iterations
    while (delta_em>tol) and (conv<iter_max):

        # Compute covariance and normalize by number of observations
        C[N_obs_ind] = np.einsum('ij,jk->jik',Y[:,N_obs_ind],Y[:,N_obs_ind].T)

        # Fill matrix B
        for (i, i_mis), j in zip(enumerate(N_mis_ind), M):
            D = S[ind_perm[i]][:,ind_perm[i]]
            D_mm = D[j:,j:]
            D_mo = D[j:,:j]
            D_oo = D[:j,:j]
            D_om = D[:j,j:]
            inv_D_oo = fast_positive_definite_inverse(D_oo)
            cov = D_mm - D_mo@inv_D_oo@D_om
            mu_mis = D_mo@inv_D_oo@np.vstack(Y[:j,i_mis])
            B[i, :j, :j] = Y[:j,i_mis,np.newaxis] @ Y[np.newaxis,:j,i_mis] # Upper left block
            B[i, j:, j:] = tau[i_mis]*cov + mu_mis @ mu_mis.T # Lower right block
            C[i_mis] = B[i][ind[i]][:,ind[i]]

        # Fixed point loop
        delta_fp = np.inf
        iteration = 0
        S_em = S
        while (delta_fp>tol) and (iteration<iter_max):

            # Compute fixed point covariance (S_fp)
            S_fp = np.zeros((p,p))
            for i in range(N):
                S_fp += C[i].T/np.trace(C[i]@np.linalg.inv(S))

            S_fp = (p/N)*S_fp

            # Performs Low Rank decomposition
            if lowrank: #and rank>=opt_rank:
                S_fp = estim_lowrank_covar(S_fp, rank)

            # Condition for stopping
            delta_fp = np.linalg.norm(S_fp - S_em, 'fro') / np.linalg.norm(S_em, 'fro')
            iteration = iteration + 1

            # Updating 𝚺
            S_em = S_fp

        # Compute delta Sigma
        delta_em = np.linalg.norm(S_em - S, 'fro') / np.linalg.norm(S, 'fro')

        conv = conv + 1

        # Update 𝚺 and tau
        S = S_em

        # Using estimated S to compute tau_new
        for i in range(N):
            tau[i] = (1./p) * np.trace(C[i]@np.linalg.inv(S))
            #tau[i] = (1./p) * np.einsum("ij,ji", C[i], np.linalg.inv(S))
    tau = tau*(np.linalg.det(S)**(1/p))
    S = S/(np.linalg.det(S)**(1/p))
    #LL.append(compute_incomplete_log_likelihood_compound(X, S, tau, P, M))

    return (S, tau)#, LL)

def estim_cmpd_gaussian_V4(Y, rank, lowrank, tol, iter_max):
    """
    m : variable index where missing data start
    r : same for observation index
    """

    # Initialisation
    (p, N) = Y.shape

    # Get row indices where values are missing
    N_mis_indices = np.where(np.sum(np.isnan(Y), axis=0))[0]
    N_obs_indices = np.where(np.sum(np.isnan(Y), axis=0)==False)[0]
    N_mis = len(N_mis_indices)
    N_obs = len(N_obs_indices)

    # Get number of observed components where observations are missing
    M = p - np.sum(np.isnan(Y[:,N_mis_indices]), axis=0)

    # Permute missing data at the bottom of each vector
    ind = np.empty((N_mis,p), dtype=int)
    ind_perm = np.empty((N_mis,p), dtype=int)
    for i, i_mis in enumerate(N_mis_indices):
        ind_perm[i] = perm.permutation_index(Y[:,i_mis])
        Y[:,i_mis], ind[i] = perm.permut_row(Y[:,i_mis])

    # Initialisation of Sigma and tau
    S = np.eye(p)
    tau = np.ones(N)

    # EM loop
    # Init matrix B (block matrix which is part of the M-step solution)
    B = np.zeros((N_mis,p,p))
    C = np.zeros((N,p,p))
    conv = 0
    LL, LL_full_estim, err = [], [], []
    delta_em = np.inf # Distance between two iterations
    while (delta_em>tol) and (conv<iter_max):

        # Compute covariance and normalize by number of observations
        for i_obs in N_obs_indices:
            C[i_obs] = Y[:,i_obs,np.newaxis] @ Y[np.newaxis,:,i_obs]

        # Fill matrix B
        for (i, i_mis), j in zip(enumerate(N_mis_indices), M):
            D = S[ind_perm[i]][:,ind_perm[i]]
            D_mm = D[j:,j:]
            D_mo = D[j:,:j]
            D_oo = D[:j,:j]
            D_om = D[:j,j:]
            #inv_D_oo = np.linalg.inv(D_oo)
            U , _ = lapack.dpotrf(D_oo)
            inv_D_oo , _ = lapack.dpotri(U)
            diag_D_oo = np.diag(inv_D_oo)
            inv_D_oo = inv_D_oo + inv_D_oo.T
            np.fill_diagonal(inv_D_oo, diag_D_oo)

            cov = D_mm - D_mo@inv_D_oo@D_om
            mu_mis = D_mo@inv_D_oo@np.vstack(Y[:j,i_mis])
            B[i, :j, :j] = Y[:j,i_mis,np.newaxis] @ Y[np.newaxis,:j,i_mis] # Upper left block
            B[i, j:, j:] = tau[i_mis]*cov + mu_mis @ mu_mis.T # Lower right block
            C[i_mis] = B[i][ind[i]][:,ind[i]]

        # Fixed point loop
        delta_fp = np.inf
        iteration = 0

        # Compute fixed point covariance (S_fp)
        S_fp = np.zeros((p,p))
        for i in range(N):
            S_fp += C[i].T/np.trace(C[i]@np.linalg.inv(S))

        S_fp = (p/N)*S_fp

        # Performs Low Rank decomposition
        if lowrank: #and rank>=opt_rank:
            S_fp = estim_lowrank_covar(S_fp, rank)

        # Updating 𝚺
        S_em = S_fp

        # Compute delta Sigma
        delta_em = np.linalg.norm(S_em - S, 'fro') / np.linalg.norm(S, 'fro')
        #print(delta_em)
        err.append(delta_em)
        conv = conv + 1

        # Update Sigma and tau
        S = S_em

        # Using estimated S to compute tau_new
        for i in range(N):
            tau[i] = (1./p) * np.trace(C[i]@np.linalg.inv(S))
    tau = tau*(np.linalg.det(S)**(1/p))
    S = S/(np.linalg.det(S)**(1/p))
    #LL.append(compute_incomplete_log_likelihood_compound(X, S, tau, P, M))

    return (S, tau)#, LL)


def estim_gaussian(Y, rank, lowrank, normdet=False, tol=1e-9, iter_max=100):
    """ Estimation of the covariance matrix of an incomplete dataset with
    general missing pattern under a Gaussian distribution.
    """

    # Initialisation
    (p, N) = Y.shape

    # Get row indices where values are missing
    N_mis_indices = np.where(np.sum(np.isnan(Y), axis=0))[0]
    N_obs_indices = np.where(np.sum(np.isnan(Y), axis=0)==False)[0]
    N_mis = len(N_mis_indices)
    N_obs = len(N_obs_indices)

    # Get number of observed components where observations are missing
    M = p - np.sum(np.isnan(Y[:,N_mis_indices]), axis=0)

    # Permute missing data at the bottom of each vector
    ind = np.empty((N_mis,p), dtype=int)
    for i, i_mis in enumerate(N_mis_indices):
        Y[:,i_mis], ind[i] = perm.permut_row(Y[:,i_mis])

    # Form permutation matrices
    P = np.array([np.eye(p)[ind[i]].T for i in range(N_mis)])

    # Initialisation of the covariance
    S = np.eye(p)

    # Expectation-Maximization loop
    conv = 0       # Iteration number
    LL = []        # Log-likelihood values are going here
    delta = np.inf # Distance error between two iterations
    while (delta>tol) and (conv<iter_max):

        # Init matrix B (block matrix which is part of the M-step solution)
        B = np.zeros((N_mis,p,p))

        # Fill matrix B
        for (i, i_mis), j in zip(enumerate(N_mis_indices), M):
            D = P[i]@S@P[i].T # Apply P on each side of covariance
            D_mm = D[j:,j:]
            D_mo = D[j:,:j]
            D_oo = D[:j,:j]
            D_om = D[:j,j:]
            cov = D_mm - D_mo@np.linalg.inv(D_oo)@D_om
            #mu_mis = D_mo@np.linalg.inv(D_oo)@np.vstack(Y[:j,i_mis])
            B[i, :j, :j] = Y[:j,i_mis,np.newaxis] @ Y[np.newaxis,:j,i_mis] # Upper left block
            B[i, j:, j:] = cov #+ mu_mis @ mu_mis.T # Lower right block

        # Compute covariance and normalize by number of observations
        C = Y[:,N_obs_indices]@Y[:,N_obs_indices].T
        for i in range(N_mis):
            C += P[i].T@B[i]@P[i]
        S_hat = C.T/N

        # Low-rank approximation
        if lowrank: #and rank>=opt_rank:
            S_hat = estim_lowrank_covar(S_hat, rank)

        # Compute distance
        delta = np.linalg.norm(S_hat - S, 'fro') / np.linalg.norm(S, 'fro')

        # Replace previous covariance by current one
        S = S_hat

        conv = conv + 1

        #LL.append(compute_incomplete_log_likelihood(X, S, P, M))
    #print(conv)
    if normdet:
        S = S/(np.linalg.det(S)**(1/p))

    return (S, LL)

def estim_lowrank_covar(S, rank):
    """ Performs Low Rank (LR) reconstruction from Algorithm 5 of Sun and Palomar (2016)
        Input:
            * S => covariance matrix
            * W => known matrix in R = W + H
            * rank => desired rank for LR
        Ouput:
            * R => Low rank covariance matrix """

    # Get shape of covariance
    p = S.shape[0]

    # EVD of covariance matrix
    v, l, _ = np.linalg.svd(S)

    # Get mean of last p-rank eigenvalues
    sig = np.mean(l[rank:])

    # Reconstruct R with Low Rank structure
    R = (l[:rank]-sig) * v[:,:rank]@v[:,:rank].T
    R += sig*np.eye(p) # Low eigenvalues part of the signal
    R /= np.trace(R) # Normalize by trace

    return R

def compute_scm_obs(X, assume_centered=True):
    """ A function that computes the SCM for covariance matrix estimation
            Inputs:
                * X = a np.array of dim (p, N)
                with each observation along column dimension
                * assume_centered = bool.
                If False, data are centered with empirical mean.
            Outputs:
                * Sigma = the estimate"""
    p, N = X.shape
    if not assume_centered:
        mean = np.mean(X, axis=1, keepdims=True)
        X = X - mean
    scm = np.zeros((p,p))
    col_sum = np.sum(~np.isnan(X), axis=0)
    N_obs = np.ravel(np.argwhere(col_sum==p)) # Fully observed columns
    for i in range(p):
        for j in range(p):
            scm[i,j] = np.nansum(X[i,N_obs]*X[j,N_obs])
    return scm/len(N_obs)


# Test
# p = 10
# rho = 0.65
# 𝚺  = toeplitz(np.power(rho, np.arange(0, p)))
# # 𝚺 = 𝚺 / (np.linalg.det(𝚺)**(1/p))
# Y = np.random.multivariate_normal(np.zeros(p), 𝚺, 300).T
# Y[0,1:60] = np.nan
# Y[3, 5:60] = np.nan

# scm = compute_scm_obs(Y)
# print(scm)

# start = time.time()
# C1, tau1 = estim_cmpd_gaussian(Y1, rank=5, lowrank=False, tol=1e-4, iter_max=50)
# print(time.time()-start)

# start = time.time()
# C2, tau2 = estim_cmpd_gaussian_V1(Y2, rank=5, lowrank=False, tol=1e-4, iter_max=50)
# print(time.time()-start)

# start = time.time()
# C3, tau3 = estim_cmpd_gaussian_V2(Y3, rank=5, lowrank=False, tol=1e-4, iter_max=50)
# print(time.time()-start)

# start = time.time()
# C4, tau4 = estim_cmpd_gaussian_V3(Y4, rank=5, lowrank=False, tol=1e-4, iter_max=50)
# print(time.time()-start)

# # start = time.time()
# # C3, tau3 = estim_cmpd_gaussian_V4(Y3, rank=5, lowrank=False, tol=1e-4, iter_max=50)
# # print(time.time()-start)

# print(np.linalg.norm(C1 - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro'))
# print(np.linalg.norm(C2 - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro'))
# print(np.linalg.norm(C3 - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro'))
# print(np.linalg.norm(C4 - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro'))

# print(tau1[150:200])
# print(tau2[150:200])
# print(tau3[150:200])
# print(tau4[150:200])
# plt.figure()
# plt.imshow(C1)
# plt.figure()
# plt.imshow(C2)
# plt.figure()
# plt.imshow(C3)
# plt.show()
