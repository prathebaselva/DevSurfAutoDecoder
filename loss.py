import math
import numpy as np
import random
import torch
from customsvd_py import *
import torch.backends.cudnn as cudnn
from curvature import *
from wnnm import *

deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)


def datafidelity_lossnormal(predicted_sdf, predicted_gradient, gt_sdf_tensor, surface_normals, surfaceP_indices, args):

    pred_sdf = predicted_sdf.clone()
    numpos = len(torch.where(pred_sdf > 0)[0])
    numneg = len(torch.where(pred_sdf < 0)[0])
    dataloss = args.data_delta * torch.nn.functional.l1_loss(predicted_sdf, gt_sdf_tensor , reduction=args.data_reduction) 

    print(len(surfaceP_indices))
    normal_reg_loss = args.normal_delta * ( 1- torch.nn.functional.cosine_similarity(predicted_gradient[surfaceP_indices], surface_normals[surfaceP_indices], dim=-1)).mean()

    loss = dataloss + normal_reg_loss
    return loss


def datafidelity_loss(predicted_sdf, gt_sdf_tensor, latent_codes, epoch, args):

    dataloss = args.data_delta * torch.nn.functional.l1_loss(predicted_sdf, gt_sdf_tensor , reduction=args.data_reduction)
    codeloss = args.code_delta * (torch.norm(latent_codes, dim=1)).mean()
    loss = dataloss + codeloss

    return loss

def datafidelity_testloss(predicted_sdf, gt_sdf_tensor, latent_codes, epoch, args, gradient=None):
    dataloss = args.data_delta * torch.nn.functional.l1_loss(predicted_sdf, gt_sdf_tensor , reduction=args.data_reduction)
    #codeloss = args.code_delta* (torch.norm(latent_codes, dim=1)).mean()
    codeloss =  args.code_delta* torch.norm(latent_codes)
    if args.losstype == 'dataeikonal':
        eikonal_loss = args.eikonal_delta * ((gradient.norm(dim=-1)-1)**2).mean()
        loss = dataloss + codeloss + eikonal_loss
    else:
        loss = dataloss + codeloss

    return loss

def implicit_wnnm_loss(orig_hessian_matrix, X_hat, args):
    hess_regularizer = args.hess_delta * torch.abs(orig_hessian_matrix - X_hat.detach()).mean()
    return hess_regularizer

def implicit_loss(epoch, iteration, gradient, hessian_matrix, surface_normals, args, valid_indices, predicted_sdf):
    n = len(gradient)
    predicted_sdf = predicted_sdf.view(predicted_sdf.size(0)) 
    pred_sdf = torch.abs(predicted_sdf.view(predicted_sdf.size(0)))
    #maxsdf = torch.max(pred_sdf)
#    if iteration: # and ( args.losstype == 'svd5' or args.losstype == 'svd6' or args):
#        histogram = np.histogram(list(predicted_sdf[valid_indices].clone().detach().cpu().numpy()),bins=100)
#        print("mcube valid surface histogram =",histogram)
#        histogram = np.histogram(list(np.abs(predicted_sdf[valid_indices].clone().detach().cpu().numpy())),bins=100)
#        print("mcube abs valid surface histogram =",histogram)


    hess_regularizer = torch.tensor(0).to(device)
    sdfloss = torch.tensor(0).to(device)
    SVD = torch.tensor(0)
#    if iteration:
#        SVD = (torch.linalg.svd(hessian_matrix)[1]).sum(dim=1)
#        #eigen = torch.real(torch.linalg.eigvals(hessian_matrix)).sum(dim=1)
#        #meanCurv = (meanCurvature(gradient, hessian_matrix))
#        rp = np.random.randint(n, size=10)
#        #for k in zip(SVD[rp], eigen[rp], gradient[rp].norm(dim=-1), meanCurv[rp], predicted_sdf[rp]):
#        #    print("{} {} {} {} {}".format(k[0], k[1], k[2], k[3], k[4]))
#        for k in zip(SVD[rp], gradient[rp].norm(dim=-1), predicted_sdf[rp]):
#            print("{} {} {}".format(k[0], k[1], k[2] ))


    if args.hess_delta:
        hess_regularizer = torch.tensor(2e20).to(device)
        #U1,SVD1,V1 = customsvd(hessian_matrix) # (torch.linalg.svd(hessian_matrix))
        U1,SVD1,V1 = (torch.linalg.svd(hessian_matrix))
        SVD = (SVD1).sum(dim=1)


        if args.losstype == 'svd':
            hess_regularizer = args.hess_delta * SVD.mean()
        if args.losstype == 'psum':
            hess_regularizer = args.hess_delta * SVD1[:,2:].mean()
        if args.losstype == 'svd3':
            hess_regularizer = args.hess_delta * SVD1[:,2:].mean()
        if args.losstype == 'detsvd3':
            hess_regularizer = args.hess_delta * SVD1[:,2:].mean()
        if args.losstype == 'invsvd':
            hess_regularizer = args.hess_delta * (1/(1e-10+SVD)).mean()

        if args.losstype == 'logdet':
            sign, detvalue = torch.linalg.slogdet(hessian_matrix)
            hess_regularizer = args.hess_delta * torch.abs(detvalue).mean()

        if args.losstype == 'logdetT':
            detvalue = torch.logdet(torch.bmm(hessian_matrix.permute(0,2,1), hessian_matrix) + torch.eye(3).to(device).repeat(n,1,1))
            hess_regularizer = args.hess_delta * torch.abs(detvalue).mean() 
        if args.losstype == 'eikonal':
            hess_regularizer = args.hess_delta * torch.abs(gradient.norm(dim=-1)-1).mean()

#    eikonalloss = torch.tensor(0).to(device)
#    if args.imp_eikonal_delta:
        #eikonalloss = args.imp_eikonal_delta * torch.abs(gradient.norm(dim=-1) -1e-5).mean()
 #       eikonalloss = args.imp_eikonal_delta * torch.abs(gradient.norm(dim=-1) -1).mean()

#    if iteration:
#        print("hessloss =",hess_regularizer)
#        print("sdfloss = ", sdfloss)
#        print("eikonalloss = ", eikonalloss)
#
#    if torch.isnan(hess_regularizer):
#        print("valid indices = ",len(valid_indices))
#    if torch.isnan(eikonalloss):
#        print("eikonalloss ", gradient)
    #dataloss = 0 
    #dataloss = 0 
    #loss = hess_regularizer +  eikonalloss
    loss = hess_regularizer
    return loss 
