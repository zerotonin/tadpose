#!/bin/bash
#SBATCH --account=matal178
#SBATCH --partition=aoraki_bigmem
#SBATCH --job-name=superprototypes
#SBATCH --output=superprototypes_%j.out
#SBATCH --error=superprototypes_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G


# Variables
n=5
num_labs=34
label_column='label'

python_interpreter="/home/matal178/miniconda3/envs/rapids-24.06/bin/python"
superprototypes_script="/home/matal178/PyProjects/tadpole_wells/post_clustering_analysis/SlurmSuperPrototypesAnalysis.py"
aggregate_script="/home/matal178/PyProjects/tadpole_wells/post_clustering_analysis/AggregateSuperPrototypesAnalysis.py"
input_csv_file="/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/agglom_3_and_7_aug21_davies_bouldin_20to40_tadpole_ids_and_labels.csv"
processed_csv_file="/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/34_clust_processed_superprototypes/34_labelling_superprototypes.csv"
output_path="/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/34_clust_processed_superprototypes/super_output"
aggregate_output_h5="/projects/sciences/zoology/geurten_lab/tadpole_project/cluster_analysis/aug_22_k_34/34_clust_processed_superprototypes/aggregate_outputs/aggregate_output_chain_len${n}.h5"

# List of deer IDs
trial_ids=($(seq 1 456))

# Array to hold job IDs
job_ids=()

# Submit jobs for each deer
for trial_id in "${trial_ids[@]}"; do
  echo "Submitting job for trial_id: $trial_id"
  job_id=$(sbatch -p aoraki_bigmem --time=01:00:00 --mem-per-cpu=16G --parsable --wrap="srun $python_interpreter $superprototypes_script --input_csv_file_path $input_csv_file --processed_csv_file_path $processed_csv_file --n $n --output_path $output_path --trial_id $trial_id --num_labels $num_labs --label_column $label_column")
  job_ids+=($job_id)
done

# Create a comma-separated list of job IDs
job_ids_str=$(IFS=, ; echo "${job_ids[*]}")

# Submit the aggregate job with dependency on the previous jobs
sbatch --dependency=afterok:$job_ids_str --wrap="srun $python_interpreter $aggregate_script --input_folder $output_path --chain_length $n --output_h5_file $aggregate_output_h5 --num_labels $num_labs"

echo "All jobs submitted. Aggregate job will run after completion of individual jobs."
