#!/bin/bash

# Import machine-specific paths (TADPOSE_* env vars) from local_paths.json
source "$(dirname "${BASH_SOURCE[0]}")/load_paths.sh" hpc

######## CLUSTERING CONFIG #########
# Order of execution for deletion sizes
declare -a execution_order=(0 50 25 10)
# Combine all k_values
declare -a k_values_all=( $(seq 2 1 20) 25 30 35 40 45 50 ) 
# The clustering tag
tag="normclusterdata_3videos"
####################################

# User Variables
# 80s Hacker Mode: ON
echo "Initializing job submission protocol..."
echo "Select GPU type:"
echo "  1) A100 - Maximum Power"
echo "  2) H100 - Balanced Approach"
echo "  3) L40 - Efficiency Mode"

read -p "Enter your choice (1-3): " gpu_choice

# Configuration Matrix (Totally Hidden from Prying Eyes)
case $gpu_choice in
  1) partition="aoraki_gpu"
     start=2
     step=2
     stop=34
     echo "A100 selected. Engaging maximum power..." 
     ;;
  2) partition="aoraki_gpu_H100"
     start=36
     step=2
     stop=68
     echo "H100 selected. Balancing performance and efficiency..."
     ;;
  3) partition="aoraki_gpu_L40"
     start=70
     step=2
     stop=100
     echo "L40 selected. Conserving energy, maximizing runtime..."
     ;;
  *) echo "Invalid selection. Aborting mission..."
      exit 1
      ;;
esac

# The rest of your job submission logic using $partition and $del_pos_seq
echo "Submitting jobs to partition: $partition"
echo "Using sequence: ${del_pos_seq[@]}" 


# Adjusted command for generating filenames without requiring CUDA
filename_cmd="${TADPOSE_PYTHON_INTERPRETER} -m tadpose.clustering --print-meta-path"

# Base command structure for clustering, requiring CUDA
base_cmd="${TADPOSE_PYTHON_INTERPRETER} -m tadpose.clustering"

# Root directory for data and results
data_file="${TADPOSE_DATA_ROOT}/cluster_data/normcluster_data_3_videos.npy"
result_dir="${TADPOSE_DATA_ROOT}/cluster_results"

# SLURM job parameters for readability, updated partition
slurm_params="--account=${TADPOSE_ACCOUNT} --partition=$partition --nodes=1 --ntasks-per-node=1 --gpus-per-task=1 --mem=64GB"

# Iterate based on the predefined order
for del_size in "${execution_order[@]}"; do
    for del_pos in $(seq $start $step $stop); do
        for k in "${k_values_all[@]}"; do
            # Generate filename for checking if the job is already done
            meta_file=$($filename_cmd -sd "$result_dir" -t "$tag" -nc "$k" -ds "$del_size" -dp "$del_pos")
            # Check if the meta file already exists
            if [ ! -f "$meta_file" ]; then
                # Construct job_name with del_size, del_pos, and k
                job_name="clust_${del_size}_${del_pos}_k${k}"
                
                # Submit the job for each k value separately
                sbatch $slurm_params --job-name="$job_name" --wrap="$base_cmd -t $tag -nc $k -ds $del_size -dp $del_pos -rs 0 -df $data_file -sd $result_dir"
            else
                echo "Did not cluster combination k=$k, del_size=$del_size, del_pos=$del_pos because already done."
            fi
        done
    done
done

