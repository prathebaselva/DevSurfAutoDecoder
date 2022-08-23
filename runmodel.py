#import open3d
import torch
import torch.backends.cudnn as cudnn
from loss import datafidelity_loss
from torch.autograd import Variable
from numpy import arange
import random

outfolder = 'output/'

deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)


def runtrainmodel(dataset, latent, model, optimizer, epoch, args, isTrain=True):
    loss_sum = 0.0
    loss_count = 0.0
    indexcount = 0
    #print("num_batch=",num_batch)
    surfaceP_points = []
    surfaceP_points_gradients = []
    surfaceP_points_svd = []

    interval = (epoch % 1 ==0)

    numiter = 100 #1500
    for iteration in range(numiter):
        randindex = arange(len(dataset))
        random.shuffle(randindex)
        for index in randindex:
            fileindex, data = dataset[index] 
            optimizer.zero_grad()
            sampled_points = data[:,0:3].to(device) # data['xyz'].to(device)

            this_bs =  sampled_points.shape[0]

            lat_vec = latent(torch.tensor(index)).to(device)
            lat_vec = lat_vec.expand(this_bs,-1)
            sampled_points = torch.cat([lat_vec, sampled_points],dim=1).to(device)
            if not isTrain:
                with torch.no_grad():
                    predicted_sdf = model(sampled_points)
            else:
                predicted_sdf = model(sampled_points)
            gt_sdf_tensor = torch.clamp(data[:,3:].to(device), -args.clamping_distance, args.clamping_distance)

            loss = datafidelity_loss(predicted_sdf, gt_sdf_tensor, lat_vec, epoch, args)

            if isTrain:
                loss.backward()
                #if args.clip > 0:
                #    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    #            if printinterval:
    #                total_norm = 0
    #                for name,p in model.named_parameters():
    #                    if p.requires_grad:
    #                        print("name = ", name)
    #                        param_norm = p.grad.data.norm(2)
    #                        print(param_norm)
    #                        total_norm += param_norm.item() ** 2
    #                total_norm = total_norm **(1./2)
    #                print("total norm = ", total_norm)
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip, norm_type=2)
          
            loss_sum += loss.item() * this_bs
                
            loss_count += this_bs
            if isTrain:
                optimizer.step()

        if loss_count == 0:
            return 2e20, -512
        return loss_sum, loss_count



