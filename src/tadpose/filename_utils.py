import os
import sys

def generate_filename(parent_dir, tag, num_clusters, deletion_size, deletion_position):
    """
    Generates structured filenames for clustering results, ensuring organized output.

    Args:
        parent_dir (str): The base directory for saving clustering output.
        tag (str): Identifier for the data or experiment.
        num_clusters (int): The number of clusters used.
        deletion_size (int): Size of data deletion (for applicable algorithms). 
        deletion_position (int): Position of data deletion (for applicable algorithms)

    Returns:
        dict: A dictionary with keys 'centroids', 'labels', and 'meta',
              containing the corresponding generated filenames.
    """

    base_dir_template = os.path.join(parent_dir, tag, f'delSize_{deletion_size}', f'k_{num_clusters}')

    filenames = {}
    for file_type, extension in [('centroids', 'npy'), ('labels', 'npy'), ('meta', 'json')]:
        if file_type in ['centroids', 'labels']:
            base_dir = os.path.join(base_dir_template, file_type)  # Add subdirectory for centroids/labels
        else:
            base_dir = base_dir_template 
        filename = f"{tag}_{file_type}_k{num_clusters}_delSize{deletion_size}_delPosP{deletion_position}.{extension}"
        filenames[file_type] = os.path.join(base_dir, filename)
    return filenames

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python generate_filename.py parent_dir tag num_clusters deletion_size deletion_position")
        sys.exit(1)

    # Unpack command-line arguments
    _, parent_dir, tag, num_clusters, deletion_size, deletion_position = sys.argv  

    filenames = generate_filename(parent_dir, tag, int(num_clusters), int(deletion_size), int(deletion_position))
    print(filenames['meta'])  # Example: Print the generated metadata filename 
