#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=pretrain_testall_noisy_0.01_241_250_svd3_model980_1e-06_1e-05_reg1_adamoptim_lat256_dc1e-04_dd1e4_BB051_rand1x_none_sine30xxsimple5hnd256lat2
#SBATCH --output=pretrain_testall_noisy_0.01_241_250_svd3_model980_1e-06_1e-05_reg1_adamoptim_lat256_dc1e-04_dd1e4_BB051_rand1x_none_sine30xxsimple5hnd256lat2.out  # output file
#SBATCH -e pretrain_testall_noisy_0.01_241_250_svd3_model980_1e-06_1e-05_reg1_adamoptim_lat256_dc1e-04_dd1e4_BB051_rand1x_none_sine30xxsimple5hnd256lat2.err        # File to which STDERR will be written
#SBATCH -p gypsum-titanx
#SBATCH --exclude=gypsum-gpu030,gypsum-gpu078
#SBATCH --mem=60G

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=5-00:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_999.2
module load cuda/11.3.1

###python -m main -ra --latfname='rand1x_1e-04_sine30xxsimple5hnd256lat2' --save_file_name='pretrain_testall_noisy_0.01_svd3_model980_1e-06_1e-05_reg1_adamoptim_lat256_dc1e-04_dd1e4_BB051_rand1x_none_sine30xxsimple5hnd256lat2' --epoch=11000 --latlr=1e-04 --lr=1e-06 --checkpoint_folder='alltestcheckpoints/' --reg 1 --hess_delta=1e1 --code_delta=1e-04 --data_delta=1e4 --use_checkpoint_model=1 --use_model='best' --use_pretrained_model=1 --pretrained_model_folder='pretrained_model_folder/model980_BB051_rand1x_none_sine30xxsimple5hnd256lat2/' --typemodel='deepsdf' --clamping_distance=0.01 --clamp 0 --clip=0 --randx=1 --data_reduction='mean' --optimizer='adam' --model='simple5hnd256lat2' --activation='sine' --grid_N=256 --isclosedmesh=1 --normalization='none' --omega=10 --withomega=1 --trainfilepath='data/train.txt' --traindir='../data/250k_sampled/sdfdata' --subsample=512 --batch_size=1 --lat=256 --testfilename='noisy_0.01_svd3' --testfilepath='data/test_noisy_0.01_svd3.txt' --testdir='../data/test_data/' --osstype='svd3' --BB=0.51 --resamp=2 --mcube=1
python -m main -i --latfname='noisy_0.01_rand1x_1e-05_sine30xxsimple5hnd256lat2' --save_file_name='reconstruct_noisy_0.01_241_250_model980_1e-06_1e-05_reg1_06_05_iter1000_sdfdata_adamoptim_lat256_dc1e-04_dd1e4_BB051_rand1x_resamp2_none_sine30xxsimple5hnd256lat2' --epoch=1000 --latlr=1e-05 --lr=1e-06 --checkpoint_folder='checkpoints/checkpoint_pretrain_noisy_0.01_241_250_model980_1e-06_1e-05_reg1_iter1000_sdfdata_adamoptim_lat256_dc1e-04_dd1e4_BB051_rand1x_resamp2_none_sine30xxsimple5hnd256lat2' --reg 1 --hess_delta=1e1 --use_checkpoint_model=1 --use_model='best' --use_pretrained_model=1 --pretrained_model_folder='pretrained_model_folder/model980_BB051_rand1x_none_sine30xxsimple5hnd256lat2/' --typemodel='deepsdf' --clamping_distance=0.01 --clamp 0 --clip=0 --randx=1 --data_reduction='mean' --optimizer='adam' --model='simple5hnd256lat2' --activation='sine' --grid_N=256 --isclosedmesh=1 --normalization='none' --omega=10 --withomega=1 --trainfilepath='data/train.txt' --traindir='../../data/250k_sampled/sdfdata' --subsample=512 --batch_size=1 --lat=256 --testfilepath='data/noisy_0.01_241_250.txt' --testdir='/work/pselvaraju_umass_edu/data/test_data/' --losstype='svd3' --BB=0.51 --resamp=2 --mcube=1 --testfilename='noisy_0.01_241_250_svd3_sine'


sleep 1
exit
