import torch.utils.data as data
import numpy as np
import math
import torch
import os
from scipy.spatial import cKDTree
from numpy.random import rand, seed, shuffle
from utils import *



def getSignedsdf(points, surf_points, surf_normals, nearest_index, sample_variance):
    gt_sdf = []
    sign = []
    for i in range(11):
        ray_vec = points - surf_points[nearest_index[:,i]]
        ray_vec_len = ray_vec.norm(dim=-1)
        index_normal = surf_normals[nearest_index[:,i]]
        if i == 0:
            gt_sdf = ray_vec_len
            index = torch.where(ray_vec_len < sample_variance)[0]
            dot = torch.bmm(index_normal[index].unsqueeze(dim=1), ray_vec[index].unsqueeze(dim=-1)).squeeze()
            gt_sdf[index] = torch.abs(dot) # torch.abs(index_normal[index].dot(ray_vec[index]))
        ray_vec = torch.nn.functional.normalize(ray_vec, dim=-1) 
        sign.append(-torch.sign(torch.bmm(index_normal.unsqueeze(dim=1), (ray_vec.unsqueeze(dim=-1))).squeeze()))

    sign = torch.stack(sign).transpose(-1,0)
    pos_sign = torch.count_nonzero(sign>0, dim=-1) 
    neg_sign = torch.count_nonzero(sign<0, dim=-1) 
    sign = (pos_sign > neg_sign).int()
    sign[torch.where(sign == 0)] = -1
    gt_sdf =(sign*gt_sdf)
    return gt_sdf


def getUniformRandomPointsInBBWithUnsignedSdf(kdtree, number_rand_points, sample_variance, surf_points, surf_normals):
    rand_points = np.random.uniform(-1,1,(number_rand_points,3))
    rand_points = torch.tensor(rand_points)

    gt_rand_sdf, nearest_index = kdtree.query(rand_points,k=1)
    ray_vec = rand_points - surf_points[nearest_index]
    ray_vec = torch.nn.functional.normalize(ray_vec, dim=-1) 
    rand_normals = ray_vec # torch.ones((number_rand_points, 3)) * -1

    return rand_points, rand_normals, torch.tensor(gt_rand_sdf).unsqueeze(dim=-1)

def getUniformRandomPointsInBBWithSignedSdf(kdtree, number_rand_points, sample_variance, surf_points, surf_normals):
    #rand_points = rand(number_rand_points, 3) 
    #for i in range(3):
    #    rand_points[:,i] = (rand_points[:,i] * (max_dimensions[i]-min_dimensions[i]))+min_dimensions[i]
    rand_points = np.random.uniform(-1,1,(number_rand_points,3))
    rand_points = torch.tensor(rand_points)

    gt_rand_sdf, nearest_index = kdtree.query(rand_points,k=1)
    ray_vec = rand_points - surf_points[nearest_index]
    ray_vec_len = ray_vec.norm(dim=-1)
    #index_normal = torch.tensor(orig_normals[nearest_index[:,i]])
    #index = torch.where(ray_vec_len < 2*np.sqrt(sample_variance))[0]
    #mask = torch.ones(number_rand_points, dtype=bool)
    #mask[index] = False
    #rand_points = rand_points[mask]
    #number_rand_points = len(rand_points)

    ray_vec = torch.nn.functional.normalize(ray_vec, dim=-1) 
    rand_normals = ray_vec # torch.ones((number_rand_points, 3)) * -1

    _, nearest_index = kdtree.query(rand_points,k=11)

    gt_rand_sdf = getSignedsdf(rand_points, surf_points, surf_normals, nearest_index, sample_variance)
    return rand_points, rand_normals, gt_rand_sdf.unsqueeze(dim=-1)


def getPerturbedPointsAlongNormalWithUnsignedSdf(kdtree, samp_points, samp_normals, number_points, sample_variance, surf_points, surf_normals):
    gt_pert_sdf = torch.tensor(np.random.normal(scale=np.sqrt(sample_variance), size=(number_points, 1)))
    pert_points = samp_points + samp_normals * gt_pert_sdf

    return pert_points, samp_normals, gt_pert_sdf

def getPerturbedPointsAlongNormalWithSignedSdf(kdtree, samp_points, samp_normals, number_points, sample_variance, surf_points, surf_normals):
    gt_pert_var = torch.tensor(np.random.normal(scale=np.sqrt(sample_variance), size=(number_points, 1)))
    pert_points = samp_points + samp_normals * gt_pert_var
    _, nearest_index = kdtree.query(pert_points,k=11)
    gt_pert_sdf = getSignedsdf(pert_points, surf_points, surf_normals, nearest_index, sample_variance)

    return pert_points, samp_normals, gt_pert_sdf.unsqueeze(dim=-1)

#def getPerturbedAndRandomPoints(kdtree, points1, normals1, points2, normals2, number_points, number_rand_points, sample_variance_1, sample_variance_2, orig_points, orig_normals, typemodel, isVal=False):
def getPerturbedAndRandomPoints(kdtree, p_points, p_normals, p_var , number_points, number_rand_points, orig_points, orig_normals, BB=0.5):

    #print("number points = ", number_points)
    #if number_points > 0:
    pert_var = []
    for p,v in zip(p_points, p_var):
        #variance = torch.clamp(torch.tensor(np.random.normal(scale=np.sqrt(v), size=(number_points, 3))), -np.sqrt(v), np.sqrt(v))
        #variance = torch.clamp(torch.tensor(np.random.normal(scale=np.sqrt(v), size=(number_points, 3))), -v, v)
        variance = torch.tensor(np.random.normal(scale=np.sqrt(v), size=(number_points, 3)))
        #print(variance)
        pert_var.append(p + variance)
    pert_var = torch.vstack(pert_var)

    print("number of pert =",len(pert_var))

    rand_points = np.random.uniform(-BB, BB,(number_rand_points,3))
    rand_points = torch.tensor(rand_points)

    pert_normals = [] #torch.cat((normals, normals, rand_normals))
    #print("nearest_index = ", nearest_index)
    pointsToRemove = []
    rand_gt_sdf, nearest_index = kdtree.query(rand_points,k=1)
    ray_vec = rand_points - orig_points[nearest_index]
    ray_vec_len = ray_vec.norm(dim=-1)
    #index_normal = torch.tensor(orig_normals[nearest_index[:,i]])
    #if not typemodel == 'siren':
    #index_normal = orig_normals[nearest_index]
    #index = torch.where(ray_vec_len < (p_var[0]))[0]
    #mask = torch.ones(number_rand_points, dtype=bool)
    #mask[index] = False
    #rand_points = rand_points[mask]
    #rand_gt_sdf = rand_gt_sdf[mask]
    #print(pert_points.shape)
    #print(len(rand_points))
        

    #pert_points = torch.cat((pert_var1, pert_var2, rand_points))
    #pert_points = torch.cat((pert_var1, pert_var2, pert_var3, pert_var4, rand_points))
    #pert_points = torch.cat((pert_var, rand_points))
    pert_points = pert_var # torch.cat((pert_var, rand_points))
    gt_sdf, nearest_index = kdtree.query(pert_points,k=11)
    sign = []
    for i in range(11):
        ray_vec = pert_points - orig_points[nearest_index[:,i]]
        #ray_vec = orig_points[nearest_index[:,i]] - pert_points
        ray_vec_len = ray_vec.norm(dim=-1)
        #index_normal = torch.tensor(orig_normals[nearest_index[:,i]])
        index_normal = orig_normals[nearest_index[:,i]]
        if i == 0:
            gt_sdf = ray_vec_len
            #index = torch.where(ray_vec_len < sample_variance_2)[0]
            #index = torch.where(ray_vec_len < 0.00025)[0]
            #dot = torch.bmm(index_normal[index].unsqueeze(dim=1), ray_vec[index].unsqueeze(dim=-1)).squeeze()
            #gt_sdf[index] = torch.abs(dot) # torch.abs(index_normal[index].dot(ray_vec[index]))
        ray_vec = torch.nn.functional.normalize(ray_vec, dim=-1) 
        if i == 0:
            pert_normals.append(ray_vec)
        sign.append(torch.sign(torch.bmm(index_normal.unsqueeze(dim=1), (ray_vec.unsqueeze(dim=-1))).squeeze()))

    sign = torch.stack(sign).transpose(-1,0)
    pos_sign = torch.count_nonzero(sign>0, dim=-1) 
    neg_sign = torch.count_nonzero(sign<0, dim=-1) 
    sign = (pos_sign > neg_sign).int()
    sign[torch.where(sign == 0)] = -1
    gt_sdf =(sign*gt_sdf)

    print("gtsdf shape = ", gt_sdf.shape)

    gtnegpos = (torch.where(gt_sdf < 0)[0])
    gtpospos = (torch.where(gt_sdf > 0)[0])
    medianneggt = torch.median(torch.abs(gt_sdf[gtnegpos]))
    medianposgt = torch.median(torch.abs(gt_sdf[gtpospos]))
    print("medianneggt = ", medianneggt)
    print("maxneggt = ", torch.max(torch.abs(gt_sdf[gtnegpos])))
    print("medianposgt = ", medianposgt)
    print("maxposgt = ", torch.max(torch.abs(gt_sdf[gtpospos])))
    maxnegv = max(3*p_var[0], 3*medianneggt) 
    nindex = torch.where(torch.abs(gt_sdf[gtnegpos]) > maxnegv)[0]
    maxposv = max(3*p_var[0], 3*medianposgt) 
    pindex = torch.where(torch.abs(gt_sdf[gtpospos]) > maxposv)[0]

    allindex = gtnegpos[nindex]
    #allindex = torch.hstack((gtnegpos[nindex], gtpospos[pindex]))
    print("allindex = ", len(allindex))
    if len(allindex) > 0:
        #mask[allindex] = False
        p_kdtree = cKDTree(pert_points)
        _, nearest_index = p_kdtree.query(pert_points[allindex],k=301)
        print("nearest index = ", nearest_index.shape)
        #gt_near = gt_sdf[:, np.newaxis][nearest_index].squeeze()
        #sign_nearest_index = torch.sign(gt_near)
        sign_nearest_index = []
        for i in range(len(allindex)):
            sign_nearest_index.append(torch.sign(gt_sdf[nearest_index[i]]))
        sign_nearest_index = torch.stack(sign_nearest_index).squeeze()
        pos_sign = torch.count_nonzero(sign_nearest_index>0, dim=-1) 
        neg_sign = torch.count_nonzero(sign_nearest_index<0, dim=-1) 
        sign = (pos_sign > neg_sign).int()
        sign[torch.where(sign == 0)] = -1
        gt_sdf[allindex] =(sign*abs(gt_sdf[allindex]))
        print("nearest index shape = ", sign_nearest_index.shape)
    
    
    p_kdtree = cKDTree(pert_points)
    _,nearest_index = p_kdtree.query(rand_points,k=101)
    sign_nearest_index = torch.sign(gt_sdf[nearest_index])
    print(sign_nearest_index)
    print("nearest index shape = ", sign_nearest_index.shape)
    pos_sign = torch.count_nonzero(sign_nearest_index>0, dim=-1) 
    neg_sign = torch.count_nonzero(sign_nearest_index<0, dim=-1) 
    sign = (pos_sign > neg_sign).int()
    sign[torch.where(sign == 0)] = -1
    
    rand_gt_sdf =(sign*abs(rand_gt_sdf))
    pert_points = torch.cat((pert_var, rand_points))
    gt_sdf = torch.cat((gt_sdf, rand_gt_sdf))
   
    #print(len(pert_points))
    #pert_points = pert_points[mask]
    #print(len(pert_points))
    #gt_sdf = gt_sdf[mask]

    pert_normals = torch.stack(pert_normals).squeeze()
    #print(pert_normals.squeeze().shape)
    
    #pert_normals = pert_normals[mask]
    return pert_points, pert_normals, gt_sdf.unsqueeze(dim=-1)



def getSignedPerturbedAndUnsignedRandomPoints(kdtree, points1, normals1, points2, normals2, number_points, number_rand_points, sample_variance_1, sample_variance_2, orig_points, orig_normals, typemodel, isVal=False):

    pert_var1 = points1 + torch.tensor(np.random.normal(scale=np.sqrt(sample_variance_1), size=(number_points, 3)))
    pert_var2 = points2 + torch.tensor(np.random.normal(scale=np.sqrt(sample_variance_2), size=(number_points, 3)))

    rand_points = np.random.uniform(-1.02,1.02,(number_rand_points,3))
    rand_points = torch.tensor(rand_points)

    pert_normals = [] #torch.cat((normals, normals, rand_normals))
    pointsToRemove = []
    gt_sdf, nearest_index = kdtree.query(rand_points,k=1)
    rand_ray_vec = rand_points - orig_points[nearest_index]
    rand_ray_vec_len = rand_ray_vec.norm(dim=-1)
    #index_normal = torch.tensor(orig_normals[nearest_index[:,i]])

    pert_points = torch.cat((pert_var1, pert_var2))
    gt_sdf, nearest_index = kdtree.query(pert_points,k=11)
    sign = []
    for i in range(11):
        ray_vec = pert_points - orig_points[nearest_index[:,i]]
        ray_vec_len = ray_vec.norm(dim=-1)
        #index_normal = torch.tensor(orig_normals[nearest_index[:,i]])
        index_normal = orig_normals[nearest_index[:,i]]
        if i == 0:
            gt_sdf = ray_vec_len
            index = torch.where(ray_vec_len < sample_variance_2)[0]
            dot = torch.bmm(index_normal[index].unsqueeze(dim=1), ray_vec[index].unsqueeze(dim=-1)).squeeze()
            gt_sdf[index] = torch.abs(dot) # torch.abs(index_normal[index].dot(ray_vec[index]))
        ray_vec = torch.nn.functional.normalize(ray_vec, dim=-1) 
        if i == 0:
            pert_normals.append(ray_vec)
        sign.append(torch.sign(torch.bmm(index_normal.unsqueeze(dim=1), (ray_vec.unsqueeze(dim=-1))).squeeze()))

    sign = torch.stack(sign).transpose(-1,0)
    pos_sign = torch.count_nonzero(sign>0, dim=-1) 
    neg_sign = torch.count_nonzero(sign<0, dim=-1) 
    sign = (pos_sign > neg_sign).int()
    sign[torch.where(sign == 0)] = -1
    gt_sdf =(sign*gt_sdf)
    gt_sdf = torch.hstack((gt_sdf, rand_ray_vec_len))

    pert_normals = torch.vstack((torch.stack(pert_normals).squeeze(), torch.tensor(rand_ray_vec)))
    pert_points = torch.cat((pert_points, rand_points)) 
    #print(pert_normals.squeeze().shape)
    return pert_points, pert_normals.squeeze(), gt_sdf.unsqueeze(dim=-1)


