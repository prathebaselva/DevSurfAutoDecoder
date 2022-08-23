import torch.utils.data as data
import numpy as np
import math
import torch
import os
from scipy.spatial import cKDTree
from numpy.random import rand, seed, shuffle
from utils import *
from gtSDF import *


class DeepSdfDataset(data.Dataset):
    def __init__(self, data , phase='train', args=None):
        self.phase = phase
        self.numfiles = len(data)
        if self.phase == 'test':
            self.batchsize = args.test_batch
            self.samples_xyz = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
            self.number_samples = self.samples_xyz.shape[0]
            self.xyz = self.samples_xyz
            self.number_batches = math.ceil(self.number_samples * 1.0 / self.batchsize)

        else:
            self.points = [[] for i in range(len(data))]
            self.normals = [[] for i in range(len(data))]
            self.xyz = [[] for i in range(len(data))]
            self.normals_xyz = [[] for i in range(len(data))]
            self.gt_sdf = [[] for i in range(len(data))]
            self.number_samples = [[] for i in range(len(data))]
            self.number_batches = [[] for i in range(len(data))]
            self.surf_startindex = [0 for i in range(len(data))]
            self.rand_startindex = [0 for i in range(len(data))]
            

            for index,pointnormal in enumerate(data): 
                self.points[index] = pointnormal[:,0:3]
                self.normals[index] = pointnormal[:,3:]
                self.points[index] = torch.tensor(self.points)
                self.normals[index] = torch.tensor(self.normals)
                self.batchsize = args.train_batch

                self.sample_variance_1 = args.sample_variance_1
                self.sample_variance_2 = args.sample_variance_2
                self.number_points = self.points.shape[0]
                kdtree = cKDTree(self.points[index])
              
                num_p = 1 
                pert_number_points = num_p*int(len(self.points[index])/2)
                #pert_number_points = num_p*int(len(self.points)*2)
                if args.typemodel == 'siren':
                    pert_number_points = 0
                #rand_number_points = 2*int(len(self.points))
                rand_number_points = num_p*int(args.randx*len(self.points[index]))

                samples_indices1 = np.random.randint(len(self.points[index]), size=pert_number_points,)
                points1 = self.points[samples_indices1, :]
                normals1 = self.normals[samples_indices1, :]

                samples_indices2 = np.random.randint(len(self.points[index]), size=pert_number_points,)
                points2 = self.points[samples_indices2, :]
                normals2 = self.normals[samples_indices2, :]


                if args.sdfnorm == 1:   
                    if args.isclosedmesh:
                        #self.pert_xyz, self.pert_normals, self.pert_gt_sdf = getPerturbedAndRandomPoints(kdtree, points1, normals1, points2, normals2, pert_number_points, rand_number_points, self.sample_variance_1, self.sample_variance_2, self.points, self.normals, args.typemodel)
                        p_points = [points1, points2]
                        p_normals = [normals1, normals2]
                        p_var = [self.sample_variance1, self.sample_variance2] 
                        self.pert_xyz, self.pert_normals, self.pert_gt_sdf = getPerturbedAndRandomPoints(kdtree, p_points, p_normals, p_var,  pert_number_points, rand_number_points, self.points[index], self.normals[index], args.typemodel)
                    else:
                        self.pert_xyz, self.pert_normals, self.pert_gt_sdf = getSignedPerturbedAndUnsignedRandomPoints(kdtree, points1, normals1, points2, normals2, pert_number_points, rand_number_points, self.sample_variance_1, self.sample_variance_2, self.points, self.normals, args.typemodel)


                self.surf_startindex[index] = len(self.pert_xyz)
                self.rand_startindex[index] = 2*pert_number_points
                p  = torch.vstack([self.points[index] for i in range(num_p)])
                n  = torch.vstack([self.normals[index] for i in range(num_p)])
                self.xyz[index] = torch.vstack((self.pert_xyz, p))
                self.normals_xyz[index] = torch.vstack((self.pert_normals, n))
                self.gt_sdf[index] = torch.vstack((self.pert_gt_sdf, torch.zeros(num_p*len(self.points[index]),1)))
                self.number_samples[index] = len(self.xyz)

                self.number_batches[index] = math.ceil(len(self.xyz) * 1.0 / self.batchsize)
#                if phase == 'val':
#                    posgt = torch.where(self.gt_sdf > 0)[0] 
#                    neggt = torch.where(self.gt_sdf < 0)[0] 
#                    zerogt = torch.where(self.gt_sdf == 0)[0] 
#                    convertToPLY(self.xyz[posgt], self.normals_xyz[posgt],  self.gt_sdf[posgt].squeeze().numpy(),True, 'posgt_'+args.save_file_name)
#                    convertToPLY(self.xyz[neggt], self.normals_xyz[neggt], self.gt_sdf[neggt].squeeze().numpy(),True, 'neggt_'+args.save_file_name)
#                    convertToPLY(self.xyz[zerogt], self.normals_xyz[zerogt], self.gt_sdf[zerogt].squeeze().numpy(),True, 'zerogt_'+args.save_file_name)
                    #saveInnpy(self.xyz[posgt], self.normals_xyz[posgt], self.gt_sdf[posgt], True, 'posgt')
                    #saveInnpy(self.xyz[neggt], self.normals_xyz[neggt], self.gt_sdf[neggt], True, 'neggt')
               


    def setBatchSize(self, batchsize):
        self.batchsize = batchsize
   

    def __getitem__(self, fileindex):
        xyz = self.xyz[fileindex]
        gt_sdf = self.gt_sdf[fileindex]
        return np.hstack((xyz, gt_sdf))
        

    def getpoints(self, fileindex, idx):
        start_idx = idx * self.batchsize
        end_idx = min(start_idx + self.batchsize, self.number_samples[fileindex])  # exclusive
        #print("number samples = ",self.number_samples)
        this_bs = end_idx - start_idx

        indices = np.random.randint(self.number_samples[fileindex], size=(this_bs, ))
        xyz = self.xyz[fileindex][indices, :]
        gt_sdf = self.gt_sdf[fileindex][indices,:]
        normals_xyz = self.normals_xyz[fileindex][indices, :]
        #surface_indices = indices[indices >= self.pertindex]
        surface_indices = np.where(indices >= self.surf_startindex)[0]
        rand_indices = np.where((indices >= self.rand_startindex) & (indices < self.surf_startindex))[0]
        pert_indices = np.where((indices < self.rand_startindex))[0]
        #print(surface_indices)

        indices = np.random.randint(len(self.points[fileindex]), size=(this_bs, ))
        surface_xyz = self.points[fileindex][indices, :]
        surface_normals = self.normals[fileindex][indices, :]
        return {'xyz': torch.FloatTensor(xyz.float()), 'gt_sdf': torch.FloatTensor(gt_sdf.float()), 'normals_xyz': torch.FloatTensor(normals_xyz.float()), 'surface_xyz':torch.FloatTensor(surface_xyz.float()), 'surface_normals':torch.FloatTensor(surface_normals.float()), 'surface_indices':surface_indices, 'rand_indices':rand_indices, 'pert_indices':pert_indices, }
