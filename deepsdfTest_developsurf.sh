#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=deepsdfTest_developsurf_griffin_250k_06_reg1_de0_dh1_dm0_ds0_dd1_clamp01_allpoints_surfReg
#SBATCH --output=deepsdfTest_developsurf_griffin_250k_06_reg1_de0_dh1_dm0_ds0_dd1_clamp01_allpoints_surfReg.out  # output file
#SBATCH -e deepsdfTest_developsurf_griffin_250k_06_reg1_de0_dh1_dm0_ds0_dd1_clamp01_allpoints_surfReg.err        # File to which STDERR will be written
#SBATCH -p 1080ti-short
#SBATCH --mem=200000
#SBATCH --exclude=node084

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=00-00:30         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_9.2
module load cuda11/11.2.1

python -m train -e --input_pts /mnt/nfs/work1/kalo/pselvaraju/DevelopSurf/data/NormalizedPts/griffin_250k_normalized.pts --save_file_name='output/griffin_250k_06_reg1_de0_dh1_dm0_ds0_dd1_clamp01_allpoints_surfReg' --test_batch=512 --grid_N=64 --checkpoint_folder='checkpoint_griffin_250k_06_reg1_de0_dh1_dm0_ds0_dd1_clamp01_allpoints_surfReg' --omega 30 --scheduler='reducelr' --clamp 0


sleep 1
exit
