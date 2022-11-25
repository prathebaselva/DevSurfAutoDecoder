#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=sphere_02_reg0_mean1_samp50_512
#SBATCH --output=sphere_02_reg0_mean1_samp50_512.out  # output file
#SBATCH -e sphere_02_reg0_mean1_samp50_512.err        # File to which STDERR will be written
#SBATCH -p titanx-long
#SBATCH --mem=100000

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=02-00:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_9.2
module load cuda11/11.2.1

python -m train --input_pts /mnt/nfs/work1/kalo/pselvaraju/DevelopSurf/data/NormalizedPts/sphere_normalized.pts --save_file_name='sphere_02_reg0_mean1_samp50_512' --train_batch=512 --epoch=100000 --lr=1e-02 --checkpoint_folder='checkpoint_sphere_02_reg0_mean1_samp50_512' --reg 0 --omega 30 --scheduler='reducelr' --N_samples=1 --clamping_distance=0.01 --offsurf_delta=0 --use_surface_points=True --data_reduction='mean' --data_delta=1 --eikonal_delta=0 --optimizer='sgd' --clamp 0


sleep 1
exit
