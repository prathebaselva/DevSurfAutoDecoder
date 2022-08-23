import torch.utils.data as data
import numpy as np
import math
import torch
import os
from scipy.spatial import cKDTree
from numpy.random import rand, seed, shuffle
from utils import *
from deepsdf_gtSDF import *


class DeepSdfDataset(data.Dataset):
    def __init__(self, data , phase='train', args=None):
        self.phase = phase
        self.numfiles = len(data)
        if self.phase == 'evaluate':
            self.phase = phase
            self.max_dimensions = np.ones((3, )) * 1.5
            self.min_dimensions = -np.ones((3, )) * 1.5
            seed()
            bounding_box_dimensions = self.max_dimensions - self.min_dimensions  # compute the bounding box dimensions of the point cloud
            #print("bounding box dim = ",bounding_box_dimensions)
            #grid_spacing = max(bounding_box_dimensions) / (args.grid_N - 9)  # each cell in the grid will have the same size
            #dim = np.arange(self.min_dimensions[0] - grid_spacing*4, self.max_dimensions[0] + grid_spacing*4, grid_spacing)
            grid_spacing = max(bounding_box_dimensions) / args.grid_N  # each cell in the grid will have the same size
            dim = np.arange(self.min_dimensions[0] - grid_spacing, self.max_dimensions[0] + grid_spacing, grid_spacing)
            X, Y, Z = np.meshgrid(list(dim), list(dim), list(dim))

            self.grid_shape = X.shape
            self.batchsize = args.test_batch
            self.samples_xyz = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
            self.number_samples = self.samples_xyz.shape[0]
            self.xyz = self.samples_xyz
            self.posxyz = self.samples_xyz
            self.number_batches = math.ceil(self.number_samples * 1.0 / self.batchsize)

        else:
            self.subsample = args.pershapesamples
            self.points = [[] for i in range(len(data))]
            self.normals = [[] for i in range(len(data))]
            self.xyz = [[] for i in range(len(data))]
            self.posxyz = [[] for i in range(len(data))]
            self.negxyz = [[] for i in range(len(data))]
            self.normals = [[] for i in range(len(data))]
            self.posgtsdf = [[] for i in range(len(data))]
            self.neggtsdf = [[] for i in range(len(data))]
            self.gt_sdf = [[] for i in range(len(data))]
            self.number_samples = [[] for i in range(len(data))]
            self.number_batches = [[] for i in range(len(data))]
            self.surf_startindex = [-1 for i in range(len(data))]
            self.rand_startindex = [-1 for i in range(len(data))]

            self.batchsize = args.train_batch
            for index,pointnormal in enumerate(data): 
                if self.loadData(index, args):
                    continue
                self.points[index] = pointnormal[:,0:3]
                self.normals[index] = pointnormal[:,3:]
                self.points[index] = torch.tensor(self.points[index])
                self.normals[index] = torch.tensor(self.normals[index])

                self.sample_variance_1 = args.sample_variance_1
                self.sample_variance_2 = args.sample_variance_2
                self.number_points = self.points[index].shape[0]
                kdtree = cKDTree(self.points[index])
              
                num_p = 1 
                pert_number_points = num_p*int(len(self.points[index])/2)
                #pert_number_points = num_p*int(len(self.points)*2)
                rand_number_points = num_p*int(args.randx*len(self.points[index]))

                samples_indices1 = np.random.randint(len(self.points[index]), size=pert_number_points,)
                points1 = self.points[index][samples_indices1, :]
                normals1 = self.normals[index][samples_indices1, :]

                samples_indices2 = np.random.randint(len(self.points[index]), size=pert_number_points,)
                points2 = self.points[index][samples_indices2, :]
                normals2 = self.normals[index][samples_indices2, :]


                if args.sdfnorm == 1:   
                    if args.isclosedmesh:
                        #self.pert_xyz, self.pert_normals, self.pert_gt_sdf = getPerturbedAndRandomPoints(kdtree, points1, normals1, points2, normals2, pert_number_points, rand_number_points, self.sample_variance_1, self.sample_variance_2, self.points, self.normals, args.typemodel)
                        p_points = [points1, points2]
                        p_normals = [normals1, normals2]
                        p_var = [self.sample_variance_1, self.sample_variance_2] 
                        self.pert_xyz, self.pert_normals, self.pert_gt_sdf = getPerturbedAndRandomPoints(kdtree, p_points, p_normals, p_var,  pert_number_points, rand_number_points, self.points[index], self.normals[index], args.typemodel)
                    else:
                        self.pert_xyz, self.pert_normals, self.pert_gt_sdf = getSignedPerturbedAndUnsignedRandomPoints(kdtree, points1, normals1, points2, normals2, pert_number_points, rand_number_points, self.sample_variance_1, self.sample_variance_2, self.points, self.normals, args.typemodel)


                self.surf_startindex[index] = len(self.pert_xyz)
                self.rand_startindex[index] = 2*pert_number_points
                #p  = torch.vstack([self.points[index] for i in range(num_p)])
                #n  = torch.vstack([self.normals[index] for i in range(num_p)])
                #self.xyz[index] = torch.vstack((self.pert_xyz, p))
                #self.normals[index] = torch.vstack((self.pert_normals, n))
                #self.gt_sdf[index] = torch.vstack((self.pert_gt_sdf, torch.zeros(num_p*len(self.points[index]),1)))
                self.xyz[index] = self.pert_xyz
                self.gt_sdf[index] = self.pert_gt_sdf 
                self.number_samples[index] = len(self.xyz)
                self.number_batches[index] = math.ceil(len(self.xyz[index]) * 1.0 / self.batchsize)
                self.saveData(index,args)

#                if phase == 'val':
#                    posgt = torch.where(self.gt_sdf > 0)[0] 
#                    neggt = torch.where(self.gt_sdf < 0)[0] 
#                    zerogt = torch.where(self.gt_sdf == 0)[0] 
#                    convertToPLY(self.xyz[posgt], self.normals[posgt],  self.gt_sdf[posgt].squeeze().numpy(),True, 'posgt_'+args.save_file_name)
#                    convertToPLY(self.xyz[neggt], self.normals[neggt], self.gt_sdf[neggt].squeeze().numpy(),True, 'neggt_'+args.save_file_name)
#                    convertToPLY(self.xyz[zerogt], self.normals[zerogt], self.gt_sdf[zerogt].squeeze().numpy(),True, 'zerogt_'+args.save_file_name)
                    #saveInnpy(self.xyz[posgt], self.normals[posgt], self.gt_sdf[posgt], True, 'posgt')
                    #saveInnpy(self.xyz[neggt], self.normals[neggt], self.gt_sdf[neggt], True, 'neggt')
               

    def loadData(self, index, args):
        fname = str(index)
        posxyznpy = os.path.join(args.datadir, fname+'_'+self.phase+'_posxyz.npy') 
        negxyznpy = os.path.join(args.datadir, fname+'_'+self.phase+'_negxyz.npy') 
        #normalnpy = os.path.join(args.datadir, fname+'_'+self.phase+'_normals.npy')
        posgtnpy = os.path.join(args.datadir,fname+'_'+self.phase+'_posgtsdf.npy')
        neggtnpy = os.path.join(args.datadir,fname+'_'+self.phase+'_neggtsdf.npy')
        if os.path.exists(posxyznpy) and os.path.exists(negxyznpy) and os.path.exists(posgtnpy) and os.path.exists(neggtnpy):
            self.posxyz[index] = np.load(posxyznpy)
            self.negxyz[index] = np.load(negxyznpy)
            #self.normals[index] = np.load(normalnpy)
            self.posgtsdf[index] = np.load(posgtnpy)
            self.neggtsdf[index] = np.load(neggtnpy)
            self.number_samples[index] = len(self.posxyz[index])+len(self.negxyz[index])
            self.number_batches[index] = math.ceil(self.number_samples[index]*1.0 / self.batchsize)
            return 1
        else:
            return 0

    def saveData(self, index, args):
        fname = str(index)
        posindex = torch.where(self.gt_sdf[index]>=0)[0]
        negindex = torch.where(self.gt_sdf[index]<0)[0]
        self.posxyz[index] = self.xyz[index][posindex]
        self.negxyz[index] = self.xyz[index][negindex]
        self.posgtsdf[index] = self.gt_sdf[index][posindex]
        self.neggtsdf[index] = self.gt_sdf[index][negindex]
 
        np.save(os.path.join(args.datadir, fname+'_'+self.phase+'_posxyz.npy'), self.xyz[index][posindex])
        np.save(os.path.join(args.datadir, fname+'_'+self.phase+'_negxyz.npy'), self.xyz[index][negindex])
        #np.save(os.path.join(args.datadir,fname+'_'+self.phase+'_normals.npy'), self.normals[index])
        np.save(os.path.join(args.datadir, fname+'_'+self.phase+'_posgtsdf.npy'), self.gt_sdf[index][posindex])
        np.save(os.path.join(args.datadir, fname+'_'+self.phase+'_neggtsdf.npy'), self.gt_sdf[index][negindex])

    def setBatchSize(self, batchsize):
        self.batchsize = batchsize

    def __len__(self):
        return len(self.posxyz)

    def __getitem__(self, fileindex):
        posxyz = self.posxyz[fileindex]
        negxyz = self.negxyz[fileindex]
        posgtsdf = self.posgtsdf[fileindex]
        neggtsdf = self.neggtsdf[fileindex]
        half = int(self.subsample / 2)
        posindex = np.random.randint(len(posxyz), size=(half,))
        negindex = np.random.randint(len(negxyz), size=(self.subsample-half,))
        return fileindex,np.vstack((np.hstack((posxyz[posindex], posgtsdf[posindex])), np.hstack((negxyz[negindex], neggtsdf[negindex]))))

    def getpoints(self, fileindex, idx):
        start_idx = idx * self.batchsize
        end_idx = min(start_idx + self.batchsize, self.number_samples[fileindex])  # exclusive
        #print("number samples = ",self.number_samples)
        this_bs = end_idx - start_idx

        indices = np.random.randint(self.number_samples[fileindex], size=(this_bs, ))
        xyz = self.xyz[fileindex][indices, :]
        gt_sdf = self.gt_sdf[fileindex][indices,:]
        normals = self.normals[fileindex][indices, :]
        #surface_indices = indices[indices >= self.pertindex]
        surface_indices = np.where(indices >= self.surf_startindex)[0]
        rand_indices = np.where((indices >= self.rand_startindex) & (indices < self.surf_startindex))[0]
        pert_indices = np.where((indices < self.rand_startindex))[0]
        #print(surface_indices)

        indices = np.random.randint(len(self.points[fileindex]), size=(this_bs, ))
        surface_xyz = self.points[fileindex][indices, :]
        surface_normals = self.normals[fileindex][indices, :]
        return {'xyz': torch.FloatTensor(xyz.float()), 'gt_sdf': torch.FloatTensor(gt_sdf.float()), 'normals': torch.FloatTensor(normals.float()), 'surface_xyz':torch.FloatTensor(surface_xyz.float()), 'surface_normals':torch.FloatTensor(surface_normals.float()), 'surface_indices':surface_indices, 'rand_indices':rand_indices, 'pert_indices':pert_indices, }
