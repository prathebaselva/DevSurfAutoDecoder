#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=test_table_250k_1e-05_reg0_d1e4_o0_k0_n0_clamp01_noeik_adam_512_deepsdfequal_pointx_pertx_randx_unsdf
#SBATCH --output=test_table_250k_1e-05_reg0_d1e4_o0_k0_n0_clamp01_noeik_adam_512_deepsdfequal_pointx_pertx_randx_unsdf.out  # output file
#SBATCH -e test_table_250k_1e-05_reg0_d1e4_o0_k0_n0_clamp01_noeik_adam_512_deepsdfequal_pointx_pertx_randx_unsdf.err        # File to which STDERR will be written
#SBATCH -p titanx-short
#SBATCH --mem=10000
#SBATCH --exclude=node083,node029,node084,node097,node030

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=00-01:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:1
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_9.2
module load cuda11/11.2.1

python -m train -e --input_pts /mnt/nfs/work1/kalo/pselvaraju/DevelopSurf/data/NormalizedPts/table_250k_normalized.pts --save_file_name='output/table_250k_1e-05_reg0_d1e4_o0_k0_n0_clamp01_noeik_adam_512_deepsdfequal_pointx_pertx_randx_unsdf' --train_batch=512 --epoch=10000 --checkpoint_folder='checkpoint_table_250k_1e-05_reg0_d1e4_o0_k0_n0_clamp01_noeik_adam_512_deepsdfequal_pointx_pertx_randx_unsdf' --reg 0 --omega 30 --scheduler='reducelr' --clamp 0 --offsurf_delta=0 --use_surface_points=0  --optimizer='sgd' --usedropout=1  --N_samples=100 --clamping_distance=0.01 --model='simple_512' --grid_N=256 --use_model='best'

sleep 1
exit
