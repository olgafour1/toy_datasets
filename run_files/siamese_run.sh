#!/bin/bash
#SBATCH --job-name=orig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:05:00
#SBATCH --output=/home/ofourkioti/Projects/Neighbor-based Multiple Instance Learning/results/transformer_siamese_k_1_dist_15_data_aug_%j.out
#SBATCH --error=/home/ofourkioti/Projects/Neighbor-based Multiple Instance Learning/results/transformer_siamese_k_1_dist_15_data_aug_%j.err
#SBATCH --partition=gpu


module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate histo_env
cd /home/ofourkioti/Projects/Neighbor-based Multiple Instance Learning/
python run.py --experiment_name k_1 --mode siamese --k 1 --data_path ColonCancer --input_shape 27 27 3 --extention bmp --siamese_weights_path /data/scratch/DBI/DUDBI/DYNCESYS/ofourkioti/colon_weights_15_data_aug  --data colon --siam_pixel_distance 20
