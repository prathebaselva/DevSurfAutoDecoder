#!/bin/sh
#!/bin/bash
#
#SBATCH --job-name=deepsdfTest_griffin
#SBATCH --output=deepsdfTest_griffin.out  # output file
#SBATCH -e deepsdfTest_griffin.err        # File to which STDERR will be written
#SBATCH -p 1080ti-long
#SBATCH --exclude=node128
#SBATCH --mem=200000

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=07-00:00         # Maximum runtime in D-HH:MM
#SBATCH --gres=gpu:4
#SBATCH --mail-user pselvaraju@cs.umass.edu

#module load gcc/7.1.0
#module load python3/3.7.4-191e-4
#module load cuda11/11.2.1
#module load cudnn/7.5-cuda_9.2
module load cuda11/11.2.1

python -m train -e --input_pts data/NormalizedPts/griffin_normalized.pts --save_file_name='griffin_normalized.html' --train_batch=2048 --epoch=500 --lr=1e-8 

sleep 1
exit
