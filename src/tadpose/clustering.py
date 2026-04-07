import numpy as np
import datetime,json,os
import cupy as cp 
from cuml.cluster import KMeans
from cuml.preprocessing import StandardScaler
from cuml.preprocessing import MaxAbsScaler 
from sklearn.metrics import calinski_harabasz_score
import argparse,sys

def shrink_data(data, reduction_percent, cut_position_percent):
    total_size = len(data)
    cut_size = int(total_size * (reduction_percent / 100.0))
    cut_position_start = int(total_size * (cut_position_percent / 100.0))
    
    # Calculate end position, wrapping around if necessary
    cut_position_end = cut_position_start + cut_size
    if cut_position_end > total_size:
        # Calculate how much needs to be cut from the beginning
        overspill_size = cut_position_end - total_size
        # Perform cut from the end portion
        data = np.delete(data, np.s_[cut_position_start:total_size], axis=0)
        # Adjust for the removal
        overspill_size_adjusted = overspill_size - (total_size - len(data))
        # Perform cut from the beginning portion
        data = np.delete(data, np.s_[:overspill_size_adjusted], axis=0)
    else:
        # If no wrap-around is needed, perform the cut in one go
        data = np.delete(data, np.s_[cut_position_start:cut_position_end], axis=0)
    
    return data


def save_output(centroids, labels, quality_score, data_file_position, reduction_percent, cut_position_percent, filenames, start_time, duration):
    np.save(filenames['centroids'], centroids)
    np.save(filenames['labels'], labels)
    metadata = {
        'calinski_harabasz_score': quality_score,
        'data_file': data_file_position,
        'reduction_percent': reduction_percent,
        'cut_position_percent': cut_position_percent,
        'centroids': centroids.tolist(),  # Assuming centroids can be serialized to JSON
        'analysis_start_date': start_time.strftime("%Y-%m-%d %H:%M:%S"),
        'analysis_duration': str(duration)  # or duration.total_seconds() for duration in seconds
    }
    with open(filenames['meta'], 'w') as f:
        json.dump(metadata, f)



def generate_filename(parent_dir, tag, num_clusters, deletion_size, deletion_position):
    # Define the directory structure
    base_dir_template = os.path.join(parent_dir, tag, f'delSize_{deletion_size}', f'k_{num_clusters}')
        
    # Construct the filename
    filenames = {}
    for file_type, extension in [('centroids', 'npy'), ('labels', 'npy'), ('meta', 'json')]:
        # Adjust base directory for centroids and labels
        if file_type in ['centroids', 'labels']:
            base_dir = os.path.join(base_dir_template, file_type)
        else:
            base_dir = base_dir_template
        
        # Ensure the directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        # Construct the filename
        filename = f"{tag}_{file_type}_k{num_clusters}_delSize{deletion_size}_delPosP{deletion_position}.{extension}"
        filenames[file_type] = os.path.join(base_dir, filename)

    
    return filenames

def get_quality(labels, data_gpu_scaled):
    # Check for cluster diversity
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        # Move scaled data back to CPU for scoring    
        data_cpu_scaled = cp.asnumpy(data_gpu_scaled)    
        quality_score = calinski_harabasz_score(data_cpu_scaled, labels)    
        print("Cluster Quality (Calinski-Harabasz):", quality_score)
    else:   
        quality_score = np.nan
    return quality_score



def main(tag, n_clusters, deletion_size, deletion_position, random_state, data_file_position,save_dir):
    start_time = datetime.datetime.now()

    # Load data and convert to GPU 
    arraydata = np.load(data_file_position)
    # Make leave out
    arraydata = shrink_data(arraydata,deletion_size, deletion_position)
    # Send to GPU
    data_gpu = cp.asarray(arraydata)
    # Standardize the data
    # scaler = StandardScaler(with_mean=False,with_std = False)
    # data_gpu_scaled = scaler.fit_transform(data_gpu)
    #scaler = MaxAbsScaler().fit(data_gpu)
    # data_gpu_scaled = scaler.transform(data_gpu)

    data_gpu_scaled= data_gpu # Fordebugging


    # Set up save file positions
    filenames = generate_filename(save_dir, tag, n_clusters, deletion_size, deletion_position)


    # Initialize and fit KMeans
    print(datetime.datetime.now(), 'Starting KMeans clustering')
    kmeans_gpu = KMeans(init="k-means||", n_clusters=n_clusters, random_state=random_state)
    kmeans_gpu.fit(data_gpu_scaled)
    # Retrieve labels
    labels = kmeans_gpu.labels_.get()
    # Retrieve centroids
    centroids = kmeans_gpu.cluster_centers_.get()
    print(centroids)

    quality_score = get_quality(labels,data_gpu_scaled)
    end_time = datetime.datetime.now()
    duration = end_time - start_time

    save_output(centroids, labels, quality_score, data_file_position, deletion_size, 
                deletion_position, filenames,start_time,duration)

if __name__ == "__main__":
    # Setup argparse to include an argument for external filename generation
    parser = argparse.ArgumentParser(description='Cluster analysis script.')
    parser.add_argument('--external', action='store_true',
                        help='Generate filename for external call and print it')
    parser.add_argument('-t', '--tag', type=str, help='Tag for the analysis.')
    parser.add_argument('-nc', '--n_clusters', type=int, help='Number of clusters.')
    parser.add_argument('-ds', '--deletion_size', type=int, help='Deletion size percentage.')
    parser.add_argument('-dp', '--deletion_position', type=int, help='Deletion position percentage.')
    parser.add_argument('-rs', '--random_state', type=int, default=0, help='Random state for KMeans.')
    parser.add_argument('-df', '--data_file_position', type=str, help='Path to the data file.')
    parser.add_argument('-sd', '--save_dir', type=str, default='./', help='Directory to save the output files.')

    args = parser.parse_args()

    if args.external:
        # Only proceed if the required arguments for generating filenames are provided
        if not all([args.tag, args.n_clusters is not None, args.deletion_size is not None, 
                    args.deletion_position is not None, args.save_dir]):
            print("Missing required arguments for filename generation.")
            sys.exit(1)
        filenames = generate_filename(args.save_dir, args.tag, args.n_clusters, 
                                      args.deletion_size, args.deletion_position)
        print(filenames['meta'])  # Print the path to the metadata file
    else:
        # Check for required arguments for main analysis
        required_args = [args.tag, args.n_clusters, args.deletion_size, args.deletion_position,
                         args.data_file_position, args.save_dir]
        if any(arg is None for arg in required_args):
            parser.print_help()
            sys.exit(1)
        main(args.tag, args.n_clusters, args.deletion_size, args.deletion_position, 
             args.random_state, args.data_file_position, args.save_dir)