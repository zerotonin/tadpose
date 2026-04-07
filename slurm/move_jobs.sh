#!/bin/bash

# # Define the target partitions
# # partitions=("aoraki_gpu_L40" "aoraki_gpu_A100_40GB" "aoraki_gpu_A100_80GB")
# partitions=("aoraki_gpu_L40")
# # Get all your pending jobs in the current partition
# jobs=$(squeue -u $(whoami) --states=PD -h -o "%.18i" -p aoraki_gpu_A100_80GB)

# # Distribute jobs across the new partitions
# count=0
# for job in $jobs; do
#     target_partition=${partitions[$((count % ${#partitions[@]}))]}
#     echo "Moving job $job to $target_partition"
#     scontrol update jobid=$job partition=$target_partition
#     ((count++))
# done

# Define the target partitions excluding aoraki_gpu_H100
# target_partitions=("aoraki_gpu_A100_80GB" "aoraki_gpu_L40" "aoraki_gpu_A100_40GB" "aoraki_gpu_H100" )
target_partitions=("aoraki_gpu_A100_80GB" "aoraki_gpu_L40" "aoraki_gpu_A100_40GB" "aoraki_gpu_H100")

# Get all your pending jobs in the aoraki_gpu_H100 partition
# jobs=$(squeue -u $(whoami) --states=PD -h -o "%.18i" -p aoraki_gpu_H100)
jobs=$(squeue -u $(whoami) --states=PD -h -o "%.18i") # for all running jobs
# Distribute jobs across the target partitions
count=0
for job in $jobs; do
    target_partition=${target_partitions[$((count % ${#target_partitions[@]}))]}
    echo "Moving job $job to $target_partition"
    scontrol update jobid=$job partition=$target_partition
    ((count++))
done
