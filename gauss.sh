#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=noisy_table_2_250k_1e-09_reg1_dh1e2_de0_dm0_dd1e4_pretrained_clamp01_thresh0005_512_pointx_pertx_randx_negexp_nothresh_sortw_1e2_surf1_mcube0_gauss
#SBATCH --output=noisy_table_2_250k_1e-09_reg1_dh1e2_de0_dm0_dd1e4_pretrained_clamp01_thresh0005_512_pointx_pertx_randx_negexp_nothresh_sortw_1e2_surf1_mcube0_gauss.out  # output file
#SBATCH -e noisy_table_2_250k_1e-09_reg1_dh1e2_de0_dm0_dd1e4_pretrained_clamp01_thresh0005_512_pointx_pertx_randx_negexp_nothresh_sortw_1e2_surf1_mcube0_gauss.err        # File to which STDERR will be written
#SBATCH -p titanx-short
#SBATCH --mem=200000
#SBATCH --exclude=node094,node083,node051,node059,node061,node084,node069,node030,node029,node095,node097


#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=00-04:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_9.2
module load cuda11/11.2.1

python -m train --input_pts /mnt/nfs/work1/kalo/pselvaraju/DevelopSurf/data/NormalizedPts/noisy_table_2_250k_normalized.pts --save_file_name='noisy_table_2_250k_1e-09_reg1_dh1e2_de0_dm0_dd1e4_pretrained_clamp01_thresh0005_512_pointx_pertx_randx_negexp_nothresh_sortw_1e2_surf1_mcube0_gauss' --file_name='noisy_table_2_250k' --train_batch=512 --epoch=1 --lr=1e-09 --checkpoint_folder='checkpoint_noisy_table_2_250k_1e-09_reg1_dh1e2_de0_dm0_dd1e4_pretrained_clamp01_thresh0005_512_pointx_pertx_randx_negexp_nothresh_sortw_1e2_surf1_mcube0_gauss' --hess_delta=1e4 --mean_delta=0 --eikonal_delta=0 --normal_delta=0 --data_delta=1e4 --reg 1 --omega 30 --scheduler='reducelr' --use_pretrained_model=True --pretrained_model_folder 'pretrained_model_folder/noisy_table_2/pointx_pertx_randx_ssdf_notnormal/dd1e4/' --clamp 0 --clamping_distance=0.01 --threshold=0.0005 --optimizer='adam' --model='simple_512' --load_training_checkpoint=1 --usedropout=1 --use_model='best' --expsdf=1 --gradNorm=0 --grid_N=256 --mcube=0 --onsurf=1 --exp=-1e2 --gauss=1
sleep 1
exit
