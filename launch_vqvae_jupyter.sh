# !/bin/bash
# SBATCH -p iaensta
# SBATCH -t 48:10:00 # time limit max 48:00:00
# SBATCH -c 4 # number of cores
# SBATCH --gres=gpu:1 # number of gpus required max 4 (will be limited to 1)
# SBATCH -o ~/vqvae/results.out #where to write sys.out/print
# SBATCH -D ~/vqvae # directory where to start the code
dir=$PWD
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate vqvae
cd $dir
#jupyter run vqvae-lisa-cluster.ipynb > test.log 2>&1
jupyter nbconvert --to notebook --execute vqvae-lisa-cluster.ipynb
