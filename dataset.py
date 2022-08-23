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
    def __init__(self, data=None, fnames=None, phase='train', args=None):
            self.phase = phase
            self.numfiles = len(data)
            print("len data = ", len(data))
            print("len fnames = ", len(fnames))
            self.subsample = args.subsample
               
            self.xyz = [[] for i in range(self.numfiles)]
            self.zeroxyz = [[] for i in range(self.numfiles)]
            self.posxyz = [[] for i in range(self.numfiles)]
            self.negxyz = [[] for i in range(self.numfiles)]
            self.gt_sdf = [[] for i in range(self.numfiles)]
            self.zerogtsdf = [[] for i in range(self.numfiles)]
            self.posgtsdf = [[] for i in range(self.numfiles)]
            self.neggtsdf = [[] for i in range(self.numfiles)]
            self.number_samples = [0 for i in range(self.numfiles)]
            self.number_batches = [0 for i in range(self.numfiles)]
            self.indexfnamedict = fnames
            for index,fname in fnames.items():
                self.points = data[index][:,0:3]
                #self.points -=np.mean(points, axis=0, keepdims=True)
                self.normals = data[index][:,3:]
                self.points = torch.tensor(self.points)
                self.normals = torch.tensor(self.normals)
                self.batchsize = args.train_batch
                num_p = 1 
                self.zeroxyz[index]  = self.points
                self.zerogtsdf[index] = torch.zeros(len(self.points), 1)
                if not self.loadData(index, fname, args):
                    self.sample_variance_1 = args.sample_variance_1
                    self.sample_variance_2 = args.sample_variance_2
                    self.number_points = self.points.shape[0]
                    kdtree = cKDTree(self.points)
                  
                    pert_number_points = num_p*int(len(self.points)/2)
                    #print("pert number points = ", pert_number_points)
                    #pert_number_points = num_p*int(len(self.points)*2)
                    if args.typemodel == 'siren':
                        pert_number_points = 0
                    #rand_number_points = 2*int(len(self.points))
                    rand_number_points = num_p*int(args.randx*len(self.points))

                    samples_indices1 = np.random.randint(len(self.points), size=pert_number_points,)
                    points1 = self.points[samples_indices1, :]
                    normals1 = self.normals[samples_indices1, :]

                    samples_indices2 = np.random.randint(len(self.points), size=pert_number_points,)
                    points2 = self.points[samples_indices2, :]
                    normals2 = self.normals[samples_indices2, :]


                    #if args.sdfnorm == 1:   
                    #    if args.isclosedmesh:
                            #self.pert_xyz, self.pert_normals, self.pert_gt_sdf = getPerturbedAndRandomPoints(kdtree, points1, normals1, points2, normals2, pert_number_points, rand_number_points, self.sample_variance_1, self.sample_variance_2, self.points, self.normals, args.typemodel)
                    p_points = [points1, points2] #points3, points4]
                    p_normals = [normals1, normals2]# normals3, normals4]
                    p_var = [0.002, 0.00025]
                    self.pert_xyz, self.pert_normals, self.pert_gt_sdf = getPerturbedAndRandomPoints(kdtree, p_points, p_normals, p_var,  pert_number_points, rand_number_points, self.points, self.normals, float(args.BB))
                     #   else:
                     #       self.pert_xyz, self.pert_normals, self.pert_gt_sdf = getSignedPerturbedAndUnsignedRandomPoints(kdtree, points1, normals1, points2, normals2, pert_number_points, rand_number_points, self.sample_variance_1, self.sample_variance_2, self.points, self.normals, args.typemodel, float(args.BB))

                    self.surf_startindex = len(self.pert_xyz)
                    self.rand_startindex = 2*pert_number_points
                    #print("rand_start index = ",self.rand_startindex)

                    #print(len(samples_indices))
                    #self.xyz = torch.vstack((self.pert_xyz, self.points))
                    #self.normals_xyz = torch.vstack((self.pert_normals, self.normals))
                    #self.gt_sdf = torch.vstack((self.pert_gt_sdf, torch.zeros(len(self.points),1)))
                    #p  = torch.vstack([self.points for i in range(num_p)])
                    #n  = torch.vstack([self.normals for i in range(num_p)])
                    #self.xyz[index] = torch.vstack((self.pert_xyz, p))
                    #self.normals_xyz = torch.vstack((self.pert_normals, n))
                    #self.gt_sdf[index] = torch.vstack((self.pert_gt_sdf, torch.zeros(num_p*len(self.points),1)))
                    #posindex = torch.where(self.gt_sdf[index] > 0)[0]
                    #zeroindex = torch.where(self.gt_sdf[index] == 0)[0]
                    #negindex = torch.where(self.gt_sdf[index] < 0)[0]
                    #self.posxyz[index] = self.xyz[index][posindex]
                    #self.negxyz[index] = self.xyz[index][negindex]
                    #self.zeroxyz[index] = self.xyz[index][zeroindex]
                    #self.posgtsdf[index] = self.gt_sdf[index][posindex]
                    #self.neggtsdf[index] = self.gt_sdf[index][negindex]
                    #self.zerogtsdf[index] = self.gt_sdf[index][zeroindex]
                    self.xyz[index] = self.pert_xyz
                    self.normals_xyz = self.pert_normals
                    self.gt_sdf[index] = self.pert_gt_sdf 
                    posindex = torch.where(self.gt_sdf[index] >= 0)[0]
                    negindex = torch.where(self.gt_sdf[index] < 0)[0]
                    self.posxyz[index] = self.xyz[index][posindex]
                    self.negxyz[index] = self.xyz[index][negindex]
                    self.posgtsdf[index] = self.gt_sdf[index][posindex]
                    self.neggtsdf[index] = self.gt_sdf[index][negindex]
                    self.saveData(index,fname,args)
                
                #self.xyz = torch.vstack((self.pert_xyz, self.points, self.points, self.points))
                #self.normals_xyz = torch.vstack((self.pert_normals, self.normals, self.normals, self.normals))
                #self.gt_sdf = torch.vstack((self.pert_gt_sdf, torch.zeros(3*len(self.points),1)))
                #print(len(self.xyz), flush=True)
                #print(index)
                #print(fname)
                #print(len(self.xyz[index]))
                #print(self.number_samples) 
                #self.number_samples[index] = len(self.posxyz[index]) + len(self.negxyz[index]) + len(self.points)
                self.number_samples[index] = len(self.posxyz[index]) + len(self.negxyz[index])
                #print("number samples = ", self.number_samples[index])
                #print("number pos neg samples = ", len(self.posxyz[index]) + len(self.negxyz[index]), flush=True)

                self.number_batches[index] = math.ceil(self.number_samples[index] * 1.0 / self.batchsize)

#    def loadOnSurfData(self, index, fname, args):
#        #if os.path.exists(fname+'_'+self.phase+'_xyz.pt') and os.path.exists(fname+'_'+self.phase+'_normals_xyz.pt') and os.path.exists(fname+'_'+self.phase+'_gt_sdf.pt') and os.path.exists(fname+'_'+self.phase+'_pert_xyz_len.pt'):
#        if os.path.exists(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_zeroxyz.pt')) and os.path.exists(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_zerogt.pt')): 
#            self.zeroxyz[index] = torch.load(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_zeroxyz.pt'))
#            self.zerogtsdf[index] = torch.load(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_zerogt.pt'))
        
    def loadData(self, index, fname, args):
        if os.path.exists(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_posxyz.pt')) and os.path.exists(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_posgt.pt')) and os.path.exists(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_negxyz.pt')) and os.path.exists(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_neggt.pt')): 
            print("loading data")
            #self.xyz[index] = torch.load(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_xyz.pt'))
            #self.zeroxyz[index] = torch.load(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_zeroxyz.pt'))
            #print(index)
            self.posxyz[index] = torch.load(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_posxyz.pt'))
            self.negxyz[index] = torch.load(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_negxyz.pt'))
            #self.normals_xyz = torch.load(fname+'_'+self.phase+'_normals_xyz.pt')
            #self.gt_sdf[index] = torch.load(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_gt_sdf.pt'))
            #self.zerogtsdf[index] = torch.load(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_zerogt.pt'))
            self.posgtsdf[index] = torch.load(os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_posgt.pt'))
            self.neggtsdf[index] = torch.load(os.path.join(args.sdfdir,str(index)+'_'+fname+'_'+self.phase+'_neggt.pt'))
            
            #self.surf_startindex = int(torch.load(fname+'_'+self.phase+'_pert_xyz_len.pt'))
            self.number_samples[index] = len(self.xyz[index])

            if (len(self.posxyz[index]) == len(self.posgtsdf[index])) and (len(self.negxyz[index]) == len(self.neggtsdf[index])):
                #print(str(index)+'_'+fname+'_'+self.phase+'data present', flush=True)
                return 1
            else:
                print(str(index)+'_'+fname+'_'+self.phase+'data not present', flush=True)
                return 0
            #pert_number_points = int(len(self.points)/2)
            #self.rand_startindex = 2*pert_number_points
        else:
            print(str(index)+'_'+fname+'_'+self.phase+'datanot present 1', flush=True)
            return 0

    def saveOnSurfData(self, index, fname, args):
        torch.save(self.zeroxyz[index], os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_zeroxyz.pt'))
        torch.save(self.zerogtsdf[index], os.path.join(args.sdfdir,str(index)+'_'+fname+'_'+self.phase+'_zerogt.pt'))

    def saveData(self, index, fname, args):
        print("saving data "+str(index)+'_'+fname, flush=True)
        #torch.save(self.xyz[index], os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_xyz.pt'))
        #torch.save(self.zeroxyz[index], os.path.join(args.sdfdir,str(index)+'_'+fname+'_'+self.phase+'_zeroxyz.pt'))
        torch.save(self.posxyz[index], os.path.join(args.sdfdir,str(index)+'_'+fname+'_'+self.phase+'_posxyz.pt'))
        torch.save(self.negxyz[index], os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_negxyz.pt'))
        #torch.save(self.normals_xyz, fname+'_'+self.phase+'_normals_xyz.pt')
        #torch.save(self.gt_sdf[index], os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_gt_sdf.pt'))
        #torch.save(self.zerogtsdf[index], os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_zerogt.pt'))
        torch.save(self.posgtsdf[index], os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_posgt.pt'))
        torch.save(self.neggtsdf[index], os.path.join(args.sdfdir, str(index)+'_'+fname+'_'+self.phase+'_neggt.pt'))
        #torch.save(torch.tensor(len(self.pert_xyz)), fname+'_'+self.phase+'_pert_xyz_len.pt')
        #self.number_samples[index] = len(self.xyz[index])
        #pert_number_points[index] = int(len(self.points)/2)
        #self.rand_startindex = 2*pert_number_points
        if self.phase == 'test':
            convertToPLY(self.negxyz[index], None, self.neggtsdf[index], True, fname)
            convertToPLY(self.posxyz[index], None, self.posgtsdf[index], False, fname)
            #convertToPLY(self.negxyz[index], None, self.neggtsdf[index], False, fname)

    def getindexfname(self):
        return self.indexfnamedict

    def getInputDim(self):
        inputmax = torch.max(self.points, dim=0)
        inputmin = torch.min(self.points, dim=0)
        inputmaxall = torch.max(self.xyz, dim=0)
        inputminall = torch.min(self.xyz, dim=0)
        return inputmax, inputmin, inputmaxall, inputminall

    def setBatchSize(self, batchsize):
        self.batchsize = batchsize
   
    def __len__(self):
        return self.numfiles
        #self.number_batches = math.ceil(len(self.xyz) * 1.0 / self.batchsize)
        #return self.number_batches

    def __getitem__(self, fileindex):
        #print(fileindex, flush=True)
        posxyz = self.posxyz[fileindex]
        negxyz = self.negxyz[fileindex]
        #zeroxyz = self.zeroxyz[fileindex]
        posgtsdf = self.posgtsdf[fileindex]
        neggtsdf = self.neggtsdf[fileindex]
        #zerogtsdf = self.zerogtsdf[fileindex]
        #onethird = int(self.subsample / 3)
        half = int(self.subsample / 2)
        
        posindex = np.random.randint(len(posxyz), size=(half,))
        negindex = np.random.randint(len(negxyz), size=(self.subsample-half,))
        #posindex = np.random.randint(len(posxyz), size=(onethird,))
        #negindex = np.random.randint(len(negxyz), size=(onethird,))
        #zeroindex = np.random.randint(len(negxyz), size=(self.subsample-2*onethird,))
        return fileindex,torch.tensor(np.vstack((np.hstack((posxyz[posindex], posgtsdf[posindex])), np.hstack((negxyz[negindex], neggtsdf[negindex])))))
        #return fileindex,torch.tensor(np.vstack((np.hstack((zeroxyz[zeroindex], zerogtsdf[zeroindex])), np.hstack((posxyz[posindex], posgtsdf[posindex])), np.hstack((negxyz[negindex], neggtsdf[negindex])))))

class gridData(data.Dataset):
    def __init__(self, args=None):
        self.max_dimensions = np.ones((3, )) * float(args.BB)
        self.min_dimensions = -np.ones((3, )) * float(args.BB)

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
        self.number_batches = math.ceil(self.number_samples * 1.0 / self.batchsize)

    def setBatchSize(self, batchsize):
        self.batchsize = batchsize
   
    def __len__(self):
        self.number_batches = math.ceil(len(self.xyz) * 1.0 / self.batchsize)
        return self.number_batches


    def __getitem__(self, idx):
        start_idx = idx * self.batchsize
        end_idx = min(start_idx + self.batchsize, self.number_samples)  # exclusive
        #print("number samples = ",self.number_samples)
        this_bs = end_idx - start_idx
        end_idx = min(start_idx + self.batchsize, self.number_samples)  # exclusive
        xyz = torch.tensor(self.samples_xyz[start_idx:end_idx, :])

        return torch.FloatTensor(xyz.float())
