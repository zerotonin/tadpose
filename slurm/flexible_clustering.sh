#!/bin/bash

# Import machine-specific paths (TADPOSE_* env vars) from local_paths.json
source "$(dirname "${BASH_SOURCE[0]}")/load_paths.sh" hpc

######## CLUSTERING CONFIG #########
# Order of execution for deletion sizes
declare -a execution_order=(0 50 25 10)
# Combine all k_values
# declare -a k_values_all=( $(seq 2 1 20) 25 30 35 40 45 50 ) 

# declare -a k_values_all=($(seq 20 1 40) 45 50 55 60 65 70 75 80 85 90 95 100)
declare -a k_values_all=($(seq 2 1 20) )


####################################

# User Variables
# 80s Hacker Mode: ON
echo "Initializing job submission protocol..."

# Configuration for partitions
declare -a partitions=("aoraki_gpu_L40" "aoraki_gpu_A100_40GB" "aoraki_gpu_A100_80GB" "aoraki_gpu_H100")
# declare -a partitions=("aoraki_gpu_L40" "aoraki_gpu_A100_40GB" "aoraki_gpu_A100_80GB" )

# declare -a partitions=("aoraki_gpu_L40" "aoraki_gpu_A100_40GB" "aoraki_gpu_H100")
# Adjusted command for generating filenames without requiring CUDA
filename_cmd="${TADPOSE_PYTHON_INTERPRETER} -m tadpose.clustering --print-meta-path"

# Prompt user to select the type of clustering.
# NOTE: feature-set selection (velocity-only, posture-only, weighted, ...)
# is now decided when the z-scored feature matrix is assembled upstream, not
# by separate clustering scripts.  Every option runs the same
# `tadpose.clustering` module on the matrix passed as $data_file; pick the
# option that matches the matrix you built.
echo "Select the type of clustering you want to perform:"
echo "1) Vanilla"
echo "2) Velocity Only"
echo "3) Weighted"
echo "4) Posture Only"
echo "5) Posture Diff and Velocity"
read -p "Enter the number corresponding to your choice: " clustering_choice

# Set the base command based on the user's choice
case $clustering_choice in
    1)
        base_cmd="${TADPOSE_PYTHON_INTERPRETER} -m tadpose.clustering"
        ;;
    2)
        base_cmd="${TADPOSE_PYTHON_INTERPRETER} -m tadpose.clustering"
        ;;
    3)
        base_cmd="${TADPOSE_PYTHON_INTERPRETER} -m tadpose.clustering"
        ;;
    4)
        base_cmd="${TADPOSE_PYTHON_INTERPRETER} -m tadpose.clustering"
        ;;
    5)
        base_cmd="${TADPOSE_PYTHON_INTERPRETER} -m tadpose.clustering"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac


read -p "Enter the full path to the data file: " data_file
read -p "Enter the full path to the result directory: " result_dir
read -p "Enter the tag for clustering: " tag

# SLURM job parameters for readability
slurm_params="--account=${TADPOSE_ACCOUNT} --nodes=1 --ntasks-per-node=1 --gpus-per-task=1 --mem=64GB"

# Initialize job list
declare -a job_list=()

# count=0 # for debugging

# Iterate based on the predefined order to create the job list
for del_size in "${execution_order[@]}"; do
    for del_pos in $(seq 2 2 100); do
        for k in "${k_values_all[@]}"; do
            # Generate filename for checking if the job is already done
            meta_file=$($filename_cmd -sd "$result_dir" -t "$tag" -nc "$k" -ds "$del_size" -dp "$del_pos")
            # Check if the meta file already exists
            if [ ! -f "$meta_file" ]; then
                # Construct job_name with del_size, del_pos, and k
                job_name="clust_${del_size}_${del_pos}_k${k}"
                
                # Add the job details to the job list
                job_list+=("$job_name;$k;$del_size;$del_pos")
            else
                echo "Did not cluster combination k=$k, del_size=$del_size, del_pos=$del_pos because already done."
            fi

            # count=$((count + 1)) # FOr debugging onl y run twice
            # if [ $count -ge 2 ]; then
            #     break
            # fi
        done
    done
done

# Function to get the partition index
get_partition_index() {
    partition_count=${#partitions[@]}
    echo $(( $1 % $partition_count ))
}

# Submit jobs from the job list
job_counter=0
for job in "${job_list[@]}"; do
    IFS=';' read -r job_name k del_size del_pos <<< "$job"
    partition_index=$(get_partition_index $job_counter)
    partition=${partitions[$partition_index]}
    
    # Submit the job
    sbatch $slurm_params --partition=$partition --job-name="$job_name" --wrap="$base_cmd -t $tag -nc $k -ds $del_size -dp $del_pos -df $data_file -sd $result_dir" # -rs 0 
    
    # Increment the job counter
    job_counter=$((job_counter + 1))
done

echo "All jobs submitted."
