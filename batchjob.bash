#!/bin/bash
#SBATCH --job-name=veahola
#SBATCH --account=project_2013898
#SBATCH --partition=gpumedium
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72  # The product should be 72 if requesting 1 GPU per node
#SBATCH --gres=gpu:gh200:1  # Corresponds to 1 GPU per node

# Set the number of CPU threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Place and bind CPU threads to single CPU cores
# Comment the following lines if binding is not desired
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Run the program
#srun python src/correction_pipeline.py --config src/config/correction/llama3.2_1b.json --seed 44
#srun python src/benchmarking.py --log_dir paper_logs/correction/42
#srun python src/correction_pipeline_mc.py --config src/config/correction/llama3.2_1b.json --seed 44
#srun python src/metrics.py --log_dir paper_logs/detection

#srun  python src/run_baselines.py src/config/baselines/llama3.2_1b.json 42
srun python src/main.py src/config/sup/llama3.2_1b.json 44
#srun python compare_dataframes.py
#srun python src/train.py src/config/sup/llama3.2_1b.json 42
#srun python generate_with_models.py