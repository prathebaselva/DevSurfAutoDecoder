#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=train_modelbuilding_1e-07_reg0_numiter100_lat256_weight_dc1e-04_dd1e4_rand1x_BB05_tanh10xxsimple5hnd256
#SBATCH --output=train_modelbuilding_1e-07_reg0_numiter100_lat256_weight_dc1e-04_dd1e4_rand1x_BB05_tanh10xxsimple5hnd256.out  # output file
#SBATCH -e train_modelbuilding_1e-07_reg0_numiter100_lat256_weight_dc1e-04_dd1e4_rand1x_BB05_tanh10xxsimple5hnd256.err        # File to which STDERR will be written
#SBATCH -p gypsum-1080ti-phd
#SBATCH --mem=30G


#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=00-10:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_99.2
module load cuda11/11.2.1

python -m main -e --save_file_name='train_modelbuilding_1e-07_1e-04_reg0_numiter100_lat256_weight_dc1e-04_dd1e4_rand1x_BB05_tanh10xxsimple5hnd256lat2' --train_batch=512 --epoch=10000 --checkpoint_folder='checkpoints/checkpoint_modelbuilding_1e-07_1e-04_reg0_numiter100_lat256_weight_dc1e-04_dd1e4_rand1x_BB05_tanh10xxsimple5hnd256lat2' --pretrained_model_folder='pretrained_model_folder/modelbuilding_BB05_rand1x_weight_tanh10xxsimple5hnd256lat2/' --use_pretrained_model 1 --reg 0 --code_delta=1e-04 --omega=10 --scheduler='reducelr' --grid_N=256 --use_model='best' --clamping_distance=0.01 --usedropout=1 --isolevel=1 --gauss=1 --activation='tanh' --model='simple5hnd256lat2' --normalization='weight' --withomega=1 --subsample=512 --traindir='../data/250k_sampled/sdfdata' --onsurfdir='../data/250k_sampled/onsurfdata' --testdir='../data/test_data/' --trainfilepath='data/trainbuildings.txt'  --BB=0.5 --randx=1

sleep 1
exit
