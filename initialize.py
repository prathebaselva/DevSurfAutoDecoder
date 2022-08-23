import numpy as np
import os
from utils import normalize_pts_withdia, normalize_normals
from dataset import  DeepSdfDataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

def getPointNormal(filename, onsurfdir):
    numpy_file = os.path.join(onsurfdir, filename+'.npy')
    if os.path.exists(numpy_file):
        pointnormal = np.load(numpy_file)
    else:
        input_point_cloud = np.loadtxt(os.path.join(onsurfdir,filename+'.pts'))
        points = normalize_pts_withdia(input_point_cloud[:, 0:3])
        normals = normalize_normals(input_point_cloud[:, 3:7])
        pointnormal = np.hstack((points, normals))
        np.save(os.path.join(onsurfdir, filename+'.npy'),pointnormal) 
    return pointnormal

def getDeepsdfDatafrompath(filepath, onsurfdir):
    filelist = open(filepath, 'r')
    data = []
    fnames = {}
    for index, filename in enumerate(filelist):
        filename = filename.strip()
        fnames[index] = filename
        pointnormal = getPointNormal(filename, onsurfdir)
        data.append(pointnormal)
    return data, fnames

def getDeepsdfData(filename, onsurfdir):
    filename = filename.strip()
    pointnormal = getPointNormal(filename, onsurfdir)
    fnames = {0:filename}
    return [pointnormal], fnames

def initDeepsdfDataSet(args):
    alldata, fnames = getDeepsdfDatafrompath(args.trainfilepath, args.onsurfdir)
    traindata = [[] for i in range(len(alldata))]
    valdata = [[] for i in range(len(alldata))]
    for index in range(len(alldata)):
        n_points = len(alldata[index])
        n_points_train = int(args.train_split_ratio * n_points)
        full_indices = np.arange(n_points)
        np.random.shuffle(full_indices)
        train_indices = full_indices[:n_points_train]
        val_indices = full_indices[n_points_train:]
        traindata[index] = alldata[index][train_indices]
        valdata[index] = alldata[index][val_indices] 
    
    train_dataset = DeepSdfDataset(data=traindata,fnames=fnames, phase='train', args=args)
    val_dataset = DeepSdfDataset(data=valdata,fnames=fnames, phase='val', args=args)
    return train_dataset, val_dataset

#def initDeepsdfTestDataSet(args):
#    testdata, fnames = getDeepsdfData(args.testfilepath, args.testdir)
#    test_dataset = DeepSdfDataset(data=testdata,fnames=fnames, phase='test', args=args)
#    return test_dataset

def initDeepsdfTestDataSet(args, filename):
    testdata, fnames = getDeepsdfData(filename, args.testdir)
    test_dataset = DeepSdfDataset(data=testdata,fnames=fnames, phase='test', args=args)
    return test_dataset

def initDeepsdfDataSetHessreg(indexes, args):
    alldata, fnames = getDeepsdfData(args.trainfilepath,  args.onsurfdir, args.traindir)
    traindata = [[] for i in range(len(indexes))]
    newfnames = {}
    for index in indexes:
        newfnames[index] = fnames[index] 
        #traindata[index] = alldata[indexes[index]]
    print(newfnames)
    train_dataset = DeepSdfDataset(data=alldata,fnames=newfnames, phase='train', args=args)
    return train_dataset

def initDataSet(args):
    input_point_cloud = np.loadtxt(args.input_pts)
    training_points = normalize_pts_withdia(input_point_cloud[:, :3])
    #training_points = normalize_pts(input_point_cloud[:, :3])
    isnormal = False
    if len(input_point_cloud[0] > 3):
        isnormal = True
        training_normals = normalize_normals(input_point_cloud[:, 3:])
    n_points = training_points.shape[0]
    print("=> Number of points in input point cloud: %d" % n_points)
    n_points_train = int(args.train_split_ratio * n_points)
    full_indices = np.arange(n_points)
    np.random.shuffle(full_indices)
    if args.N_samples == 16384:
        full_indices = full_indices[0:16384]
        n_points_train = int(args.train_split_ratio * 16384)
    train_indices = full_indices[:n_points_train]
    val_indices = full_indices[n_points_train:]
    #test_indices = full_indices[n_points_train+n_points_val:]
    #with torch.autograd.detect_anomaly():
    if isnormal:
        train_dataset = SdfDataset(points=training_points[train_indices], normals=training_normals[train_indices], args=args)
        val_dataset = SdfDataset(points=training_points[val_indices], normals=training_normals[val_indices], phase='val', args=args)
    else:
        train_dataset = SdfDataset(points=training_points[train_indices], normals=None, args=args)
        val_dataset = SdfDataset(points=training_points[val_indices], normals=None, phase='val', args=args)
    print("number of train points - ", len(train_dataset))
    print("number of val points - ", len(val_dataset))

    maxdim, mindim, maxdimall, mindimall = val_dataset.getInputDim()
    print("maxdim = {}, mindim = {}".format(maxdim, mindim))
    print("maxdimall = {}, mindimall = {}".format(maxdimall, mindimall), flush=True)
    return train_dataset, val_dataset


def initlatentOptimizer(latent, args):
    if args.optimizer == 'adam':
        optimizer = Adam([latent], lr=args.latlr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = SGD([latent], lr=args.latlr, weight_decay=args.weight_decay, momentum=0.9)
    #print("=> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    return optimizer

def initlatcodeandmodelOptimizer(model, latent, args):
    if args.optimizer == 'adam':
        optimizer = Adam([{"params":filter(lambda p: p.requires_grad, model.parameters()), "lr": args.lr}, {"params":[latent], "lr":args.latlr}], weight_decay=args.weight_decay)
    return optimizer

def initOptimizer(model, latent, args):
    if args.optimizer == 'adam':
        optimizer = Adam([{"params":filter(lambda p: p.requires_grad, model.parameters()), "lr": args.lr}, {"params":latent.parameters(), "lr":args.latlr}], weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    print("=> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    return optimizer

def initScheduler(optimizer, args):
    if args.scheduler == 'reducelr':
        scheduler = ReduceLROnPlateau(optimizer, factor=args.factor,patience=15,mode='min',threshold=1e-4, eps=0, min_lr=0)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000*len(train_dataset), T_mult=2)
    return scheduler

