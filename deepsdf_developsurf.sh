#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=deepsdf_developsurf_griffin_250k_06_reg1_de1e1_dh1e1_dm1e1_ds0_dd1e4_clamp0001_allpoints_surfReg
#SBATCH --output=deepsdf_developsurf_griffin_250k_06_reg1_de1e1_dh1e1_dm1e1_ds0_dd1e4_clamp0001_allpoints_surfReg.out  # output file
#SBATCH -e deepsdf_developsurf_griffin_250k_06_reg1_de1e1_dh1e1_dm1e1_ds0_dd1e4_clamp0001_allpoints_surfReg.err        # File to which STDERR will be written
#SBATCH -p titanx-long
#SBATCH --mem=200000
#SBATCH --exclude=node084

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=07-00:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_9.2
module load cuda11/11.2.1

python -m train --input_pts /mnt/nfs/work1/kalo/pselvaraju/DevelopSurf/data/NormalizedPts/griffin_250k_normalized.pts --save_file_name='griffin_250k_06_reg1_de1e1_dh1e1_dm1e1_ds0_dd1e4_clamp0001_allpoints_surfReg' --train_batch=512 --epoch=20000 --lr=1e-06 --checkpoint_folder='checkpoint_griffin_250k_06_reg1_de1e1_dh1e1_dm1e1_ds0_dd1e4_clamp0001_allpoints_surfReg' --N_samples=80 --hess_delta=1e1 --mean_delta=1e1 --eikonal_delta=1e1 --offsurf_delta=0 --data_delta=1e4 --reg 1 --omega 30 --scheduler='reducelr' --clamp 0 --clamping_distance=0.0001 --threshold=0.0001 --use_surface_points=True --use_surface_points_regularizer=True 

sleep 1
exit
