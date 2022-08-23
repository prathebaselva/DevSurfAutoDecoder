#import open3d
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
import time
from trainhelper import *
from loadmodel import *
from utils import *
from dataset import *
from runmodel import *
from initialize import *
from test import *
from inference import *

#outfolder = '/mnt/nfs/work1/kalo/pselvaraju/DevelopSurf/output/'
#outfolder = '/mnt/nfs/work1/kalo/gopalsharma/pselvaraju/DevelopSurf/output/'
#    if args.evaluate:
#        alldata, fnames = getDeepsdfData(args.trainfilepath, args.traindir)
#        testdata, testfnames = getDeepsdfData(args.testfilepath, args.testdir)
#        cham = []
#        chamfile = {}
#        for index, fname in enumerate(fnames):
#            c = getChamferDist(alldata[index][:,0:3], testdata[0][:,0:3])
#            chamfile[fname] = c
#            cham.append(c)
#            print("{} {}".format(fname, c), flush=True)
#        print(chamfile)
#        model, latent = loadEvaluateModel(model, args)
#        if model is None:
#            return
#        model.to(device)
#        print("loaded evaluation model")
#        validate_dataset = gridData(phase='test', args=args)
#        #for i in range(10):
#        getSurfacePoints(validate_dataset, latent(torch.tensor(1)), model, 0, 300000, args, prefname="test"+str(1))
#        return
outfolder = 'output/'

deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)
   

def train(dataset, latent,  model, optimizer, epoch, args):
    model.train()  # switch to train mode
    loss_sum, loss_count = runtrainmodel(dataset, latent,  model, optimizer, epoch, args, True)
    loss = loss_sum / loss_count
    #loss = getLoss(dataloader, latent,  model, optimizer, epoch, args, mcube_points)
    return loss

# validation function
def val(dataset, latent,  model, optimizer, epoch, args):
    model.eval()  # switch to test mode 
    loss_sum, loss_count = runtrainmodel(dataset, latent,  model, optimizer, epoch, args, False)
    loss = loss_sum / loss_count
    return loss 


def trainModel(args):
    best_loss = 2e20
    best_epoch = -1
    check_latent = None
    # create checkpoint folder
    if not isdir(args.checkpoint_folder):
        print("Creating new checkpoint folder " + args.checkpoint_folder)
        mkdir_p(args.checkpoint_folder)

    model = initModel(args)
    cudnn.benchmark = True
    
    if args.trainreconstruct:
        reconstructTrain(args)     
        return

    if args.reconstruct:
        reconstruct(model, args)     
        return

    if args.reconstructall:
        reconstructall(args)     
        #reconstructallForChamfer(args)
        return

    if args.inference:
        testfiles = open(args.testfilepath, 'r')
        args.resamp = 2
        args.lr = 1e-06
        args.latlr = 1e-05
        for fname in testfiles:
            fname = fname.strip()
            test_dataset = initDeepsdfTestDataSet(args, fname)
            latCodeOptimization(args, fname)
            args.hess_delta = 0
            latCodeandReg0ModelOptimization(args,fname, str(args.hess_delta))
            args.losstype = 'svd3'
            args.hess_delta = 1e1
            latCodeandModelOptimization(args,fname, str(args.hess_delta),'06_05')
            #args.losstype = 'svd'
            #args.hess_delta = 1
            #latCodeandModelOptimization(args,fname, str(args.hess_delta),'06_05')
            #args.losstype = 'logdet'
            #args.hess_delta = 1
            #latCodeandModelOptimization(args,fname, str(args.hess_delta),'06_05')
        return
#        for fname in testfiles:
#            fname = fname.strip()
#            test_dataset = initDeepsdfTestDataSet(args, fname)
#        return
#        for fname in testfiles:
#            fname = fname.strip()
#        
#            test_dataset = initDeepsdfTestDataSet(args, fname)
#            latCodeOptimization(args, fname)
#            args.latlr = 1e-05
#            test_dataset = initDeepsdfTestDataSet(args, fname)
#            latCodeOptimization(args, fname)
#            #args.hess_delta = 1e1
#            #args.resamp = 4
#            #hess = args.hess_delta
#            args.hess_delta = 0
#            args.latlr = 1e-07
#            latCodeandReg0ModelOptimization(args,fname, str(args.hess_delta))
            #args.hess_delta = 1e1
            #args.resamp = 4
            #hess = args.hess_delta
            #args.hess_delta = 0
        args.resamp = 2
        args.lr = 1e-07
        args.latlr = 1e-08
        for fname in testfiles:
            fname = fname.strip()
            args.losstype = 'svd3'
            args.hess_delta = 5e1
            latCodeandModelOptimization(args,fname, str(args.hess_delta), '07_08')
            args.hess_delta = 1e1
            latCodeandModelOptimization(args,fname, str(args.hess_delta), '07_08')
        return
        for fname in testfiles:
            fname = fname.strip()
            args.losstype = 'svd3'
            args.hess_delta = 1e2
            latCodeandModelOptimization(args,fname, str(args.hess_delta))
        return

    if args.reinference:
        testfiles = open(args.testfilepath, 'r')
        args.resamp = 2
        args.lr = 1e-06
        args.latlr = 1e-05
        for fname in testfiles:
            fname = fname.strip()
            test_dataset = initDeepsdfTestDataSet(args, fname)
            latCodeOptimization(args, fname)
#            args.hess_delta = 0
#            args.eikonal_delta = 1e1
#            args.losstype = 'dataeikonal'
#            args.resamp = 4
#            latCodeandReg0ModelOptimization(args,fname, str(args.eikonal_delta),'06_05')
        return
        for fname in testfiles:
            fname = fname.strip()
            test_dataset = initDeepsdfTestDataSet(args, fname)
            #latCodeOptimization(args, fname)
            #args.hess_delta = 0
            #args.resamp = 4
            #latCodeandReg0ModelOptimization(args,fname, str(args.hess_delta),'06_05')
            args.resamp = 2
            #args.losstype = 'svd'
            #args.hess_delta = 1
            #latCodeandModelOptimization(args,fname, str(args.hess_delta),'06_05')
            args.losstype = 'logdetT'
            args.hess_delta = 1
            latCodeandModelOptimization(args,fname, str(args.hess_delta),'06_05')
#            latCodeandHessOptimization(args, fname, str(args.hess_delta), '06_05')
        return

#def latCodeandHessOptimization(args):
        totaltime = 0
        numfile = 0
        for fname in testfiles:
            fname = fname.strip()
            test_dataset = initDeepsdfTestDataSet(args, fname)
            #latCodeOptimization(args, fname)
            #args.hess_delta = 0
            #args.resamp = 4
            #latCodeandReg0ModelOptimization(args,fname, str(args.hess_delta))
            args.resamp = 2
            args.losstype = 'psum'
            args.hess_delta = 1e1
            start = time.time()
            latCodeandModelOptimization(args,fname, str(args.hess_delta),'06_05')
            end = time.time()
            print("Time taken = ", end - start)
            totaltime += (end-start)
            numfile += 1
#            args.losstype = 'logdetT'
#            args.hess_delta = 1e1
#            start = time.time()
#            latCodeandModelOptimization(args,fname, str(args.hess_delta),'06_05')
#            end = time.time()
#            print("Time taken = logdetT ", end - start)
#            args.losstype = 'eikonal'
#            args.hess_delta = 1e1
#            start = time.time()
#            latCodeandModelOptimization(args,fname, str(args.hess_delta),'06_05')
#            end = time.time()
#            print("Time taken = eikonal", end - start)
        print("avg time = ", totaltime/numfile)
        return

        for fname in testfiles:
            fname = fname.strip()
            test_dataset = initDeepsdfTestDataSet(args, fname)
            #latCodeOptimization(args, fname)
            #args.hess_delta = 0
            #args.resamp = 4
            #latCodeandReg0ModelOptimization(args,fname, str(args.hess_delta))
            args.resamp = 2
            args.losstype = 'svd3'
            args.hess_delta = 1e1
            start = time.time()
            latCodeandModelOptimization(args,fname, str(args.hess_delta),'06_05')
            end = time.time()
            print("Time taken = ", end - start)
            args.losstype = 'svd'
            args.hess_delta = 1
            latCodeandModelOptimization(args,fname, str(args.hess_delta),'06_05')
            #args.losstype = 'logdet'
            #args.hess_delta = 1
            #latCodeandModelOptimization(args,fname, str(args.hess_delta),'06_05')
        return
#        args.lr = 1e-06
#        args.latlr = 1e-04

        for fname in testfiles:
            fname = fname.strip()
            args.losstype = 'logdet'
            args.hess_delta = 1
            latCodeandModelOptimization(args,fname, str(args.hess_delta),'06_05')
        return

        args.resamp = 2
        args.lr = 1e-07
        args.latlr = 1e-08
        for fname in testfiles:
            fname = fname.strip()
            args.losstype = 'svd'
            args.hess_delta = 1
            latCodeandModelOptimization(args,fname, str(args.hess_delta), '07_08')
            args.losstype = 'logdet'
            args.hess_delta = 0.1
            latCodeandModelOptimization(args,fname, str(args.hess_delta),'07_08')
        return
        args.lr = 1e-06
        args.latlr = 1e-07
        for fname in testfiles:
            fname = fname.strip()
            args.losstype = 'svd3'
            args.hess_delta = 1e2
            latCodeandModelOptimization(args,fname, str(args.hess_delta), '06_07')
            args.hess_delta = 5e1
            latCodeandModelOptimization(args,fname, str(args.hess_delta), '06_07')
        return
        args.resamp = 2
        args.lr = 1e-07
        args.latlr = 1e-08
        for fname in testfiles:
            fname = fname.strip()
            args.losstype = 'svd3'
            args.hess_delta = 1e2
            latCodeandModelOptimization(args,fname, str(args.hess_delta))
        return
        #args.lr = 1e-06
        #args.latlr = 1e-07
        #for fname in testfiles:
        #    fname = fname.strip()
        #    args.losstype = 'svd3'
        #    args.hess_delta = 1e1
        #    latCodeandModelOptimization(args,fname, str(args.hess_delta))
#        for fname in testfiles:
#            fname = fname.strip()
#            args.losstype = 'svd3'
#            args.hess_delta = 5e1
#            latCodeandModelOptimization(args,fname, str(args.hess_delta))
#            args.hess_delta = 1e1
#            latCodeandModelOptimization(args,fname, str(args.hess_delta))
        for fname in testfiles:
            fname = fname.strip()
            test_dataset = initDeepsdfTestDataSet(args, fname)
            args.losstype = 'svd'
            args.hess_delta = 1
            latCodeandModelOptimization(args,fname, str(args.hess_delta))
            args.hess_delta = 0.1
            latCodeandModelOptimization(args,fname, str(args.hess_delta))
        for fname in testfiles:
            fname = fname.strip()
            test_dataset = initDeepsdfTestDataSet(args, fname)
            args.losstype = 'logdet'
            args.hess_delta = 0.1
            latCodeandModelOptimization(args,fname, str(args.hess_delta))
            args.hess_delta = 1
            latCodeandModelOptimization(args,fname, str(args.hess_delta))
        return

    if args.hessdeltainference:
        testfiles = open(args.testfilepath, 'r')
        for fname in testfiles:
            fname = fname.strip()
            args.hess_delta = 1e2
            args.losstype='svd3'
            latCodeandModelOptimization(args,fname, '1e2')
            #args.hess_delta = 1
            #args.losstype='logdet'
            #latCodeandModelOptimization(args,fname, '1_logdet')
            #args.losstype='svd'
            #latCodeandModelOptimization(args,fname, '1_svd')
        #latCodeandHessOptimization(args)
        #latCodeandModelCombineOptimization(args)
        #latCodeandHessCombineOptimization(args)
        return

   
    train_dataset, val_dataset = initDeepsdfDataSet(args)
    #return
    latent = torch.nn.Embedding(len(train_dataset), args.lat, max_norm=1.0)   
    torch.nn.init.normal_(
         latent.weight.data,
        0.0,
        (1.0) / math.sqrt(args.lat),
    )

    if args.use_checkpoint_model:
        check_model, check_latent, check_best_loss, args = loadCheckpointModel(model, args)
       
        if check_model:
            best_loss = check_best_loss
            model = check_model
            latent = check_latent  
            latent.requires_grad = True
        if check_model is None:
            print("no checkpoint model exists. Start to training from scratch")
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)       
 
    model.to(device)
    optimizer = initOptimizer(model, latent, args)
    scheduler = initScheduler(optimizer, args)


    all_loss_train = []
    all_loss_val = []
    diff_epoch = 0
    curr_lr = args.lr

    grid_uniformSamples = gridData(args)

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(train_dataset, latent, model, optimizer, epoch, args)
        val_loss = val(val_dataset, latent, model, optimizer, epoch, args)
        all_loss_train.append(train_loss)
        all_loss_val.append(val_loss)
        np_all_loss_train = np.ma.masked_where(np.array(all_loss_train) >= 2e9, np.array(all_loss_train))
        np_all_loss_val = np.ma.masked_where(np.array(all_loss_val) >= 2e9, np.array(all_loss_val))

        plotloss(outfolder, epoch, args.save_file_name, np_all_loss_train, np_all_loss_val)

        is_best_loss = abs(val_loss) < best_loss
        #if is_best_loss and epoch > 50:
        #    rlatent = torch.tensor(np.random.randint(0,len(train_dataset)))
        #    print("rlatent =",rlatent)
        #    getSurfaceSamplePoints(grid_uniformSamples, latent(rlatent), model, epoch, 300000, args)

        if is_best_loss:
            loss_epoch = 0
            best_loss = val_loss
            best_epoch = epoch
            numhighloss = 0
            print("Best Epoch::Loss = {}::{}".format(epoch, val_loss))
            
            for param_group in optimizer.param_groups:
                print("LR step :: ",param_group['lr'])
                curr_lr = param_group['lr']
            save_reg0checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict(), "train_latent":latent}, is_best_loss, checkpoint_folder=args.checkpoint_folder)

        scheduler.step(val_loss)

        save_curr_checkpoint({"epoch": epoch , "lr": curr_lr, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict(), "train_latent":latent}, checkpoint_folder=args.checkpoint_folder)
        if epoch % 10 == 0:
            for param_group in optimizer.param_groups:
                print("LR step :: ",param_group['lr'])
        #print(f"Epoch{epoch:d}. train_loss: {train_loss:.8f}. Best Epoch: {best_epoch:d}. Best val loss: {best_loss:.8f}.",flush=True)
        print(f"Epoch{epoch:d}. train_loss: {train_loss:.8f}. val_loss: {val_loss:.8f}. Best Epoch: {best_epoch:d}. Best val loss: {best_loss:.8f}.",flush=True)

