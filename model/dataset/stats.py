import os
import h5py
import numpy as np
from scipy import stats

def count_files_in_subfolders(directory):
    file_counts = {".schem": 0, ".schematic": 0, ".litematic": 0}
    file_names = {ext: set() for ext in file_counts.keys()}
    for subdir in next(os.walk(directory))[1]:
        folder_path = os.path.join(directory, subdir)
        for file in os.listdir(folder_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in file_counts:
                file_counts[ext] += 1
                file_names[ext].add(os.path.splitext(file)[0])
    return file_counts, file_names

def compare_files_dgf_msf(dgf_names, msf_directory):
    msf_files = set(os.path.splitext(file)[0] for file in os.listdir(msf_directory))
    missing_in_msf = {ext: dgf_names[ext] - msf_files for ext in dgf_names}
    missing_in_dgf = msf_files - set.union(*dgf_names.values())
    return missing_in_msf, missing_in_dgf

def count_server_files(server_dir):
    converted_count = len(os.listdir(os.path.join(server_dir, "schematics-json")))
    preprocessed_count = len(os.listdir(os.path.join(server_dir, "schematics-json-preprocessed")))
    return converted_count, preprocessed_count

# Define an air block's shape.
air_block = np.zeros((16,))
air_block[0] = 1
def calculate_stats(hdf5_filename):
    with h5py.File(hdf5_filename, 'r') as file:
        total_structures = 0
        dimensions = []
        volume_buckets = [0, 0, 0, 0]  # Buckets for the volume ranges

        # Volume range thresholds
        volume_ranges = [8**3, 16**3, 32**3]

        for dataset_name in file:
            dataset = file[dataset_name]
            total_structures += 1

            non_air_blocks = ~np.all(dataset == air_block, axis=3)
            coords = np.argwhere(non_air_blocks)
            if coords.size > 0:
                x_min, y_min, z_min = coords.min(axis=0)
                x_max, y_max, z_max = coords.max(axis=0)
                dimensions.append((x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1))

                # Calculate volume
                volume = (x_max - x_min + 1) * (y_max - y_min + 1) * (z_max - z_min + 1)

                # Bucketing
                if volume <= volume_ranges[0]:
                    volume_buckets[0] += 1
                elif volume <= volume_ranges[1]:
                    volume_buckets[1] += 1
                elif volume <= volume_ranges[2]:
                    volume_buckets[2] += 1
                else:
                    volume_buckets[3] += 1
            else:
                dimensions.append((0, 0, 0))

        dimensions = np.array(dimensions)
        mean_dims = np.mean(dimensions, axis=0)
        median_dims = np.median(dimensions, axis=0)
        mode_dims = stats.mode(dimensions, axis=0).mode[0]
        min_dims = np.min(dimensions, axis=0)
        max_dims = np.max(dimensions, axis=0)

        return total_structures, mean_dims, median_dims, mode_dims, min_dims, max_dims, volume_buckets

if __name__ == "__main__":

    # Part 1: Data Gathering Files
    dgf_directory = "."
    dgf_counts, dgf_names = count_files_in_subfolders(dgf_directory)

    print("--- Data Gathering Files (DGF) ---")
    print(f"Total Files: {sum(dgf_counts.values())}")
    for ext, count in dgf_counts.items():
        print(f"{ext}: {count}")

    # Part 2: Minecraft Server Files
    msf_directory = "./server/plugins/BuildLogger/schematics"
    missing_in_msf, missing_in_dgf = compare_files_dgf_msf(dgf_names, msf_directory)
    converted_count, preprocessed_count = count_server_files("./server/plugins/BuildLogger")

    print("\n--- Minecraft Server Files (MSF) ---")
    for ext, names in missing_in_msf.items():
        print(f"Filenames found in DGF but not present in MSF ({ext}, {len(names)}): {', '.join(names) if names else 'None'}")

    print(f"Total .schem Files: {len(os.listdir(msf_directory))}")
    print(f"Files Converted On Server: {converted_count}")
    print(f"Files Preprocessed on Server: {preprocessed_count}")

    # Part 3: HDF5 File Reporting
    hdf5_filename = "./schematics.hdf5"
    if os.path.exists(hdf5_filename):
        total_structures, mean_dims, median_dims, mode_dims, min_dims, max_dims, volume_buckets = calculate_stats(hdf5_filename)

        print("\n--- HDF5 File Stats ---")
        print(f"Total Structures: {total_structures}")
        print(f"Mean Dimensions (X, Y, Z): {mean_dims}")
        print(f"Median Dimensions (X, Y, Z): {median_dims}")
        print(f"Mode Dimensions (X, Y, Z): {mode_dims}")
        print(f"Min Dimensions (X, Y, Z): {min_dims}")
        print(f"Max Dimensions (X, Y, Z): {max_dims}")

        # Print volume bucket information
        print("\n--- HDF5 Volume Bucket Counts ---")
        print(f"0 to 8^3: {volume_buckets[0]}")
        print(f"8^3 to 16^3: {volume_buckets[1]}")
        print(f"16^3 to 32^3: {volume_buckets[2]}")
        print(f"Greater than 32^3: {volume_buckets[3]}")

    else:
        print(f"File {hdf5_filename} not found.")
