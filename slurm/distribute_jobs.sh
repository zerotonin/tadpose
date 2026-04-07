#!/bin/bash
#SBATCH --job-name=distribute_gpu_jobs
#SBATCH --account=matal178
#SBATCH --partition=aoraki
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --time=19:59:59

# Load necessary modules or environment (if required)
# module load your_module_here

# Define the target partitions
target_partitions=("aoraki_gpu_A100_80GB" "aoraki_gpu_L40" "aoraki_gpu_A100_40GB" "aoraki_gpu_H100")

# Function to distribute jobs across partitions
distribute_jobs() {
    # Get all your pending jobs in target partitions
    jobs=$(squeue -u $(whoami) --states=PD --format="%.18i %.18P" -h | awk -v partitions="${target_partitions[*]}" '
    BEGIN {split(partitions, p_array, " ")}
    {
        for (i in p_array) {
            if ($2 == p_array[i]) {
                print $1
            }
        }
    }')
    
    count=0
    for job in $jobs; do
        target_partition=${target_partitions[$((count % ${#target_partitions[@]}))]}
        echo "Moving job $job to $target_partition"
        scontrol update jobid=$job partition=$target_partition
        ((count++))
    done
}

# Infinite loop to run the job distribution every 10 minutes
while true; do
    echo "Checking and distributing jobs at $(date)"
    distribute_jobs
    echo "Sleeping for 10 minutes..."
    sleep 600  # Sleep for 10 minutes (600 seconds)
done
