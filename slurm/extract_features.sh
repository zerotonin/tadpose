#!/bin/bash

# Check if the correct number of arguments was passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_filepath> <output_filepath>"
    exit 1
fi

# Extract the directory path from the output file path
output_dir=$(dirname "$2")

# Check if the output directory exists, if not, create it
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

# Execute the Python script with the provided file paths
~/miniconda3/envs/deer_project_2/bin/python extract_posture_and_velocity.py "$1" "$2"

#!/bin/bash
#SBATCH --account=account_name
#SBATCH --partition=aoraki
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=deer_analysis${deer_code}  # Set a unique job name for each analysis
~/miniconda3/envs/deer_project_2/bin/python /home/matal178/PyProjects/headshake_project/headshake_project/run_sync_for_aoraki.py "$deer_code" "cluster_paths_2"
EOF
done < "$DEER_CODES_FILE"


