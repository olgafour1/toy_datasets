#!/bin/bash
#SBATCH --job-name=orig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:05:00
#SBATCH --output=/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/toy_datasets/ColonCancer_results/siamese_k_3.out
#SBATCH --error=/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/toy_datasets/ColonCancer_results/siamese_k_3.err
#SBATCH --partition=gpu


module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate alma
cd /home/ofourkioti/Projects/Neighbor-based-Multiple-Instance-Learning/
python run.py --experiment_name  k_3 --mode siamese --k 3 --input_shape 27 27 3 --extention bmp --data colon --prob 0.75 --data_path ColonCancer  --siam_epochs 20

