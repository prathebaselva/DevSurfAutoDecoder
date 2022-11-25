#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=pretrain_abc7_model980_1e-06_1e-05_reg0_hessreg1_hess980model_BB05_adamoptim_numiter1000_lat256_weight_svd3_dh1e1_dc1e-07_dd1e4_BB05_rand1x_weight_tanh10xxsimple5hnd256lat2
#SBATCH --output=pretrain_abc7_model980_1e-06_1e-05_reg0_hessreg1_hess980model_BB05_adamoptim_numiter1000_lat256_weight_svd3_dh1e1_dc1e-07_dd1e4_BB05_rand1x_weight_tanh10xxsimple5hnd256lat2.out  # output file
#SBATCH -e pretrain_abc7_model980_1e-06_1e-05_reg0_hessreg1_hess980model_BB05_adamoptim_numiter1000_lat256_weight_svd3_dh1e1_dc1e-07_dd1e4_BB05_rand1x_weight_tanh10xxsimple5hnd256lat2.err        # File to which STDERR will be written
#SBATCH -p gypsum-titanx-phd
#SBATCH --mem=70G

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=7-00:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_999.2
module load cuda11/11.2.1

python -m main --save_file_name='pretrain_abc7_model980_1e-06_1e-05_reg0_hessreg1_hess980model_BB05_adamoptim_numiter1000_lat256_weight_svd3_dh1e1_dc1e-07_dd1e4_BB05_rand1x_weight_tanh10xxsimple5hnd256lat2' --epoch=11000 --latlr=1e-05 --lr=1e-06 --checkpoint_folder='checkpoints/checkpoint_pretrain_abc7_model980_1e-06_1e-05_reg0_hessreg1_hess980model_BB05_adamoptim_numiter1000_lat256_weight_svd3_dh1e1_dc1e-07_dd1e4_BB05_rand1x_weight_tanh10xxsimple5hnd256lat2' --reg 0 --hessreg 1 --hess_delta=1e1 --code_delta=1e-07 --data_delta=1e4 --use_checkpoint_model=1 --use_model='best' --use_pretrained_model=1 --pretrained_model_folder='pretrained_model_folder/model980_BB051_rand1x_weight_tanh10xxsimple5hnd256lat2/' --typemodel='deepsdf' --clamping_distance=0.01 --clamp 0 --clip=0 --randx=1 --data_reduction='mean' --optimizer='adam' --model='simple5hnd256lat2' --activation='tanh' --grid_N=256 --isclosedmesh=1 --normalization='weight' --omega=10 --withomega=1 --trainfilepath='data/train.txt' --traindir='../data/250k_sampled/sdfdata' --subsample=512 --batch_size=1 --lat=256 --testfilepath='data/test_abc7.txt' --testdir='../data/noisy_0.01_250k_sampled/' --losstype='svd3' --BB=0.5 --useextdropout=0 --resamp=2 --mcube=1


sleep 1
exit
