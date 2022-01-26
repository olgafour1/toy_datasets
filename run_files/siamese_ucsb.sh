#!/bin/bash
#SBATCH --job-name=orig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:05:00
#SBATCH --output=/home/ofourkioti/Projects/Neighbor-based-Multiple-Instance-Learning/ucsb_results/average_layer%j.out
#SBATCH --error=/home/ofourkioti/Projects/Neighbor-based-Multiple-Instance-Learning/ucsb_results/average_layer%j.err
#SBATCH --partition=gpu


module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate histo_env
cd /home/ofourkioti/Projects/Neighbor-based-Multiple-Instance-Learning/
python run.py --arch '{"type": "Conv2D", "channels": 36, "kernel": (4, 4)},{"type": "MaxPooling2D", "pool_size": (2, 2)},{"type": "Conv2D", "channels": 48, "kernel": (3, 3)},{"type": "MaxPooling2D", "pool_size": (2, 2)},{"type": "Flatten"},{"type": "relu", "size": 512},{"type": "Dropout", "rate": 0.2},{"type": "relu", "size": 512},{"type": "Dropout", "rate": 0.2}' --k 5 --folds 4 --data ucsb --input_shape 32 32 3  --mode siamese --data_path Breast_Cancer_Cells --ext tif --experiment_name ucsb_5_siamese --siamese_weights_path /data/scratch/DBI/DUDBI/DYNCESYS/ofourkioti/UcsbWeights


