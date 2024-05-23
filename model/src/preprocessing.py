import os
import json
import h5py
import numpy as np
import shutil
import glob

# Define an air block's shape.
air_block = np.zeros((16,))
air_block[0] = 1

# Load a JSON file and return its contents as a Python object
def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# Save a Python object as a JSON file
def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Convert JSON data (specifically a matrix) to a numpy tensor
def json_to_tensor(data):
    matrix = data['matrix']
    tensor = np.array(matrix)
    return tensor

# Convert a numpy tensor back into JSON format (specifically a matrix)
def tensor_to_json(tensor):
    matrix_list = tensor.tolist()
    return {'matrix': matrix_list}

# Remove any air blocks (represented as 1 in position 0 and zeros elsewhere) from the tensor
def remove_air(tensor):
    non_air_blocks = ~np.all(tensor == air_block, axis=3)
    coords = np.argwhere(non_air_blocks)
    x_min, y_min, _ = coords.min(axis=0)
    x_max, y_max, _ = coords.max(axis=0)
    tensor_cropped = tensor[x_min:x_max+1, y_min:y_max+1, :]
    return tensor_cropped

# Center the tensor in a 32x32 frame and pad with air blocks
def center_and_pad(tensor):
    # Calculate the offsets for centering in all 3 dimensions
    x_offset = (32 - tensor.shape[0]) // 2
    y_offset = 0
    z_offset = (32 - tensor.shape[2]) // 2
    
    # Create an empty array filled with air blocks of size 32x32x32
    centered = np.tile(air_block, (32, 32, 32, 1)).reshape(32, 32, 32, 16)
    
    # Insert the tensor at the calculated offsets
    centered[x_offset:x_offset+tensor.shape[0], y_offset:y_offset+tensor.shape[1], z_offset:z_offset+tensor.shape[2]] = tensor
    
    return centered

# Main preprocessing function to load, convert, and prepare tensor from JSON file
def preprocess(filename):
    data = load_json(filename)
    tensor = json_to_tensor(data)
    tensor_no_air = remove_air(tensor)

    # Check if the tensor's dimensions are within the expected size
    if tensor_no_air.shape[0] <= 32 and tensor_no_air.shape[1] <= 32 and tensor_no_air.shape[2] <= 32:
        tensor_padded = center_and_pad(tensor_no_air)
        return tensor_padded, True  # True indicates that the tensor is within the size limit
    else:
        return None, False  # False indicates that the tensor exceeds the size limit

# Take a one-hot vector and rotate the directional bit.
def rotate_directional_bits(vector, rotations, original_filename, x, y, z):

    # Check if the block is stairs
    if vector[8] == 1: 

        # Rotate the directional bits i times
        directional_bits = vector[11:15]
        rotated_bits = np.roll(directional_bits, rotations)
        vector[11:15] = rotated_bits

    return vector

# Save the processed tensor to an HDF5 file for all 4 rotations, OG
def save_to_hdf5(tensor, within_limit, hdf5_filename, original_filename, total_new_entries, output_directory):
    new_entries = 0
    skipped_entries = 0

    # Skip saving if the tensor exceeds the size limit
    if not within_limit:
        return new_entries, skipped_entries

    with h5py.File(hdf5_filename, 'a') as f:
        for i in range(4):
            tensor_copy = tensor.copy()
            rotated = np.rot90(tensor_copy, k=i, axes=(0, 2)).copy()

            # Update the one-hot vectors for directional blocks
            for x in range(rotated.shape[0]):
                for y in range(rotated.shape[1]):
                    for z in range(rotated.shape[2]):
                        rotated[x, y, z] = rotate_directional_bits(rotated[x, y, z], i, original_filename, x, y, z)

            entry_name = f"{os.path.basename(original_filename)}_rotation_{i}"

            if i == 0:
                # Convert the tensor back to JSON format and save
                json_data = tensor_to_json(rotated)
                save_json(json_data, os.path.join(output_directory, original_filename))

            if entry_name not in f:
                f.create_dataset(entry_name, data=rotated)
                new_entries += 1
            else:
                skipped_entries += 1

    return new_entries, skipped_entries

# Check if all rotations of a given filename exist in the HDF5 file
def all_rotations_exist(filename, hdf5_filename):
    with h5py.File(hdf5_filename, 'a') as f:
        for i in range(4):
            entry_name = f"{os.path.basename(filename)}_rotation_{i}"
            if entry_name not in f:
                return False
    return True

# Helper function to delete a file
def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")

# Helper function to move raw schematic files
def move_raw_file(schem_filename, source_locations, target_location):

    print('filename_without_extension:', schem_filename)

    for location in source_locations:
        # Construct the full file path
        file_to_move = os.path.join(location, f"{schem_filename}")

        # Check if the file exists
        if os.path.exists(file_to_move):

            # Move the file if it exists
            shutil.move(file_to_move, target_location)
            print(f"Moved: {file_to_move} to {target_location}")

        else:

            print(f"File not found: {file_to_move}")

if __name__ == "__main__":
    directory = "../dataset/server/plugins/BuildLogger/schematics-json"
    output_directory = "../dataset/server/plugins/BuildLogger/schematics-json-preprocessed"
    output_hdf5 = "../dataset/schematics.hdf5"

    # For collected schematics that exceed size limit.
    raw_file_locations = ["../dataset/server/plugins/BuildLogger/schematics"] 
    raw_file_target_location = "../dataset/server/plugins/BuildLogger/schematics/32plus"

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    total_files = len([name for name in os.listdir(directory) if name.endswith('.json')])
    print(f"Total JSON files to process: {total_files}")
    processed_files = 0

    total_new_entries = 0
    total_skipped_entries = 0

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            print(f"Processing {filename}...")
            full_path = os.path.join(directory, filename)

            if all_rotations_exist(filename, output_hdf5):
                print(f"Skipping {filename} as all rotations already exist.")
                total_skipped_entries += 4
                continue
            
            tensor, within_limit = preprocess(full_path)
            if within_limit:
                new_entries, skipped_entries = save_to_hdf5(tensor, within_limit, output_hdf5, filename, total_new_entries, output_directory)
                total_new_entries += new_entries
            else:
                print(f"Skipping {filename} as it exceeds the size limit.")
                # Delete the JSON file
                json_path = os.path.join(directory, filename)
                delete_file(json_path)

                # Move the corresponding raw schematic file
                filename_without_extension = os.path.splitext(filename)[0]
                move_raw_file(filename_without_extension, raw_file_locations, raw_file_target_location)

            
            processed_files += 1
            print(f"{processed_files} files processed")

    total_entries_with_augmentation = 0
    total_entries_without_augmentation = 0
    with h5py.File(output_hdf5, 'r') as f:
        total_entries_with_augmentation = len(f.keys())
        total_entries_without_augmentation = total_entries_with_augmentation // 4  # considering 4 rotations as augmentations

    print(f"New entries added: {total_new_entries}")
    print(f"Entries skipped: {total_skipped_entries}")
    print(f"Total entries in HDF5 with augmentations: {total_entries_with_augmentation}")
    print(f"Total entries in HDF5 without augmentations: {total_entries_without_augmentation}")
