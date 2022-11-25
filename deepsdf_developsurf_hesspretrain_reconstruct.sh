#!/bin/bash
#
#SBATCH --job-name=1e1_reconstruct_pretrain_noisy_0.01_241_250_model980_1e-06_1e-05_reg1_iter1000_sdfdata_adamoptim_lat256_dc1e-04_dd1e4_BB051_rand1x_resamp2_weight_tanh10xxsimple5hnd256lat2
#SBATCH --output=1e1_reconstruct_pretrain_noisy_0.01_241_250_model980_1e-06_1e-05_reg1_iter1000_sdfdata_adamoptim_lat256_dc1e-04_dd1e4_BB051_rand1x_resamp2_weight_tanh10xxsimple5hnd256lat2.out  # output file
#SBATCH -e 1e1_reconstruct_pretrain_noisy_0.01_241_250_model980_1e-06_1e-05_reg1_iter1000_sdfdata_adamoptim_lat256_dc1e-04_dd1e4_BB051_rand1x_resamp2_weight_tanh10xxsimple5hnd256lat2.err        # File to which STDERR will be written
#SBATCH -p gypsum-1080ti-phd
#SBATCH --mem=20G
#SBATCH --exclude=gypsum-gpu083

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=7-00:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_999.2
module load cuda/11.3.1


python -m main -r --latfname='noisy_0.01_rand1x_1e-05_tanh10xxsimple5hnd256lat2' --save_file_name='reconstruct_pretrain_noisy_0.01_241_250_model980_1e-06_1e-05_reg1_iter1000_sdfdata_adamoptim_lat256_dc1e-04_dd1e4_BB051_rand1x_resamp2_weight_tanh10xxsimple5hnd256lat2' --epoch=11000 --latlr=1e-05 --lr=1e-06 --checkpoint_folder='checkpoints/checkpoint_pretrain_noisy_0.01_241_250_model980_1e-06_1e-05_reg1_iter1000_sdfdata_adamoptim_lat256_dc1e-04_dd1e4_BB051_rand1x_resamp2_weight_tanh10xxsimple5hnd256lat2' --reg 1 --hess_delta=1e1 --use_checkpoint_model=1 --use_model='best' --use_pretrained_model=1 --pretrained_model_folder='pretrained_model_folder/model980_BB051_rand1x_weight_tanh10xxsimple5hnd256lat2/' --typemodel='deepsdf' --clamping_distance=0.01 --clamp 0 --clip=0 --randx=1 --data_reduction='mean' --optimizer='adam' --model='simple5hnd256lat2' --activation='tanh' --grid_N=256 --isclosedmesh=1 --normalization='weight' --omega=10 --withomega=1 --trainfilepath='data/train.txt' --traindir='../data/250k_sampled/sdfdata' --subsample=512 --batch_size=1 --lat=256 --testfilepath='data/noisy_0.01_241_250.txt' --testdir='../data/test_data/' --losstype='svd3' --BB=0.51 --resamp=2 --mcube=1


sleep 1
exit
