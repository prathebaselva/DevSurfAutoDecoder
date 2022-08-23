import torch
import numpy as np
import math

deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)

def getwnnm(hessian_matrix, _m):
    m = _m
    C = math.sqrt(m)
    #C = math.sqrt(0.05)
    #C = math.sqrt(0.1)
    epsilon = 1e-10
    lambdae = 1/math.sqrt(m)
    X_hat, E_hat = wnnm(hessian_matrix, C, epsilon, lambdae)
    return X_hat, E_hat
    

def wnnm(hessian_matrix, C, epsilon, lambdae):
    norm_fro = torch.linalg.norm(hessian_matrix, ord='fro', dim=(1,2))
    norm_two = torch.linalg.norm(hessian_matrix, dim=(1,2)).mean()

    X = hessian_matrix.clone().to(device)
    rho = 1.05
    mu = 1 / norm_two
    #print("mu = ", mu)
    bs = hessian_matrix.shape[0]
    L = torch.zeros(bs,3,3).to(device)
    iter = 0
    converged  = False

    while not converged:
        iter = iter + 1
        E = proxy_l1( hessian_matrix + L / mu - X, lambdae / mu)
        #print(hessian_matrix + L/mu - E)
        X = proxy_reweighted_wnnp(hessian_matrix + L/mu - E, C/mu, epsilon)
        #print(X)
        L =  L + mu * (hessian_matrix - X - E)
        mu = rho * mu
        #print("mu = ",mu)
        v = (torch.linalg.norm(hessian_matrix - X - E, ord='fro', dim=(1,2))/ norm_fro).mean()
        if v  < 1e-8:
            converged = True
      
        if iter % 5 == 0:
            print(v)
       
        if iter > 1000:
            print("not converged")
            converged = True
            break

    #print(hessian_matrix)
    #print(X)
    #print(E)
    return X, E


def proxy_l1(X, tau):
    X_hat = torch.sign(X)*torch.max(torch.abs(X) - tau , torch.tensor(0))
    return X_hat

def proxy_reweighted_wnnp(Y, C, epsilon):
    U, S, V = torch.linalg.svd(Y)
    sigma_y = torch.eye(3).to(device)*(S.unsqueeze(-1))
    C1 = sigma_y - epsilon
    C2 = (sigma_y + epsilon)**2 - 4 * C
    #print(C1)
    #print(C2)

    idx = torch.where(C2 >= 0, 1,0)
    psv = (C1*idx + torch.sqrt(C2*idx))/2
    #print(psv)

    #newV = V.permute(0,2,1)
    #newV = newV[:,idx]
    #newV = newV.permute(0,2,1)
    #x_hat = torch.matmul(torch.matmul(u[:, idx],torch.eye(3).to(device)*psv.unsqueeze(-1)), newv)
    x_hat = torch.matmul(torch.matmul(U,psv), V)
    return x_hat

def rpca(hessian_matrix):
    S = torch.zeros(hessian_matrix.shape)
    Y = torch.zeros(hessian_matrix.shape)

    l1norm = torch.linalg.norm(hessian_matrix, ord=1, dim=(1,2)).mean()
    mu =  9 / l1norm
    muinv = 1/ mu
    lambdae = 1 / torch.sqrt(3)
    
    Sk = S.clone()
    Yk = Y.clone()
    Lk = torch.zeros(hessian_matrix.shape)

    tol = 1e-7*torch.linalg.norm(hessian_matrix, ord='fro', dim=(1,2)).mean()

    err = 1e7
    while err > tol:
        Lk = svd_threshold(hessian_matrix - Sk + muinv*Yk, muinv)
        Sk = shrink(hessian_matrix - Lk + muinv*Yk, muinv*lambdae)
        Yk = Yk + mu * (hessian_matrix - Lk - Sk)
        err = torch.linalg.norm(hessian_matrix - Lk - Sk)
        iter = iter + 1
        if iter % 5 == 0:
            print(err)

    return Lk, Sk


def svd_threshold(M, tau):
    U, S, V = torch.linalg.svd(M)
    newS = shrink(S, tau)
    newM = torch.matmul(torch.matmul(U, torch.diag(newS)), V)
    return newM

def shrink(M, tau):
    newS = torch.max(torch.tensor(0), torch.abs(M) - tau)
