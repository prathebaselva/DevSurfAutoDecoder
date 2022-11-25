#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=test_noisy_table_2_250k_1e-13_reg0_d1e8_o0_k5e1_n5e1_clamp01_noeik_sgd_deepsdfequal
#SBATCH --output=test_noisy_table_2_250k_1e-13_reg0_d1e8_o0_k5e1_n5e1_clamp01_noeik_sgd_deepsdfequal.out  # output file
#SBATCH -e test_noisy_table_2_250k_1e-13_reg0_d1e8_o0_k5e1_n5e1_clamp01_noeik_sgd_deepsdfequal.err        # File to which STDERR will be written
#SBATCH -p titanx-short
#SBATCH --mem=100000
#SBATCH --exclude=node051,node083,node094,node029,node084,node097,node030

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=00-01:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_9.2
module load cuda11/11.2.1

python -m train -e --input_pts /mnt/nfs/work1/kalo/pselvaraju/DevelopSurf/data/NormalizedPts/noisy_table_2_250k_normalized.pts --save_file_name='output/noisy_table_2_250k_1e-13_reg0_d1e8_o0_k5e1_n5e1_clamp01_noeik_sgd_deepsdfequal' --train_batch=512 --epoch=10000 --checkpoint_folder='checkpoint_noisy_table_2_250k_1e-13_reg0_d1e8_o0_k5e1_n5e1_clamp01_noeik_sgd_deepsdfequal' --reg 0 --omega 30 --scheduler='reducelr' --clamp 0 --optimizer='sgd' --usedropout=1  --model='simple_512' --grid_N=256 --use_model='best' --clamping_distance=0.01	

sleep 1
exit
