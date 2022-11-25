#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=model980_1e-06_1e-05_reg0_numiter100_lat256_none_dc1e-04_dd1e4_rand1x_BB05_relu1xxsimple5h256lat2
#SBATCH --output=model980_1e-06_1e-05_reg0_numiter100_lat256_none_dc1e-04_dd1e4_rand1x_BB05_relu1xxsimple5h256lat2.out  # output file
#SBATCH -e model980_1e-06_1e-05_reg0_numiter100_lat256_none_dc1e-04_dd1e4_rand1x_BB05_relu1xxsimple5h256lat2.err        # File to which STDERR will be written
#SBATCH -p gypsum-1080ti-phd
#SBATCH --mem=200G

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=7-00:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_999.2
module load cuda11/11.2.1

python -m main --save_file_name='model980_1e-06_1e-05_reg0_numiter100_lat256_none_dc1e-04_dd1e4_rand1x_BB05_relu1xxsimple5h256lat2' --epoch=120000 --lr=1e-06 --latlr=1e-05 --checkpoint_folder='checkpoints/checkpoint_model980_1e-06_1e-05_reg0_numiter100_lat256_none_dc1e-04_dd1e4_rand1x_BB05_relu1xxsimple5h256lat2' --reg 0 --code_delta=1e-04 --data_delta=1e4 --normal_delta=1e2 --eikonal_delta=0 --use_checkpoint_model=0 --use_model='last' --typemodel='deepsdf' --clamping_distance=0.01 --clamp 0 --clip=0 --randx=1 --data_reduction='mean' --optimizer='adam' --model='simple5h256lat2' --activation='relu' --grid_N=256 --isclosedmesh=1 --normalization='none' --omega=1 --withomega=1 --trainfilepath='data/train_980.txt' --traindir='../data/250k_sampled/onsurfdata' --subsample=512 --batch_size=1 --lat=256 --BB=0.5 --sdfdir='../data/250k_sampled/sdfdata'


sleep 1
exit
