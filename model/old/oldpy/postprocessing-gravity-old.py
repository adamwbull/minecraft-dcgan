# postprocessing.py

import os
import json
import numpy as np
import math

generations_dir = "generations"
output_dir = "../dataset/server/plugins/BuildLogger/schematics-json-generated"
W = 8  # Default value for the size of the local relatedness tensor

def normalize_vector(vector):
    # Normalize the block type section (first 11 indices)
    block_type = vector[:11]
    max_index_block = np.argmax(np.abs(block_type))
    normalized_block = np.zeros_like(block_type)
    normalized_block[max_index_block] = 1

    # Normalize the direction section (next 4 indices)
    direction = vector[11:15]
    max_index_direction = np.argmax(np.abs(direction))
    normalized_direction = np.zeros_like(direction)
    normalized_direction[max_index_direction] = 1

    # For the last bit, threshold it
    last_bit = [1 if vector[15] > 0.5 else 0]
    
    # Combine the normalized sections
    normalized_vector = np.concatenate([normalized_block, normalized_direction, last_bit])
    
    return normalized_vector.tolist()


def calculate_relatedness(block1, block2):
    """
    Calculate the 'relatedness' of two blocks based on their vector representation.
    More similar indices set to 1 means higher relatedness.
    """
    similarity = np.sum(block1 * block2)
    return similarity

# Define an air block's shape.
air_block = np.zeros((16,))
air_block[0] = 1

def dfs(x, y, z, dx, dy, dz, w, relatedness_map, visited, mass_map, data):
    """
    Explore blobs of blocks via DFS to calculate blob masses.
    """
    if x < 0 or x >= dx or y < 0 or y >= dy or z < 0 or z >= dz or visited[x, y, z]:
        return 0

    visited[x, y, z] = True
    mass = 0 if data[x, y, z] == air_block else 1  # Air blocks have zero mass

    for nx in range(-1, 2):
        for ny in range(-1, 2):
            for nz in range(-1, 2):
                if 0 <= nx + x < dx and 0 <= ny + y < dy and 0 <= nz + z < dz:
                    if relatedness_map[x, y, z, nx, ny, nz] > 0:
                        mass += dfs(nx + x, ny + y, nz + z, dx, dy, dz, w, relatedness_map, visited, mass_map, data)

    mass_map[x, y, z] = mass
    return mass

def find_largest_mass_neighbor(local_x, local_y, local_z, local_mass_map, local_relatedness_map):
    max_mass = -1
    target_x, target_y, target_z = -1, -1, -1

    # Iterate within the local tensor
    for nx in range(local_mass_map.shape[0]):
        for ny in range(local_mass_map.shape[1]):
            for nz in range(local_mass_map.shape[2]):
                if local_relatedness_map[local_x, local_y, local_z, nx, ny, nz] > 0 and local_mass_map[nx, ny, nz] > max_mass:
                    max_mass = local_mass_map[nx, ny, nz]
                    target_x, target_y, target_z = nx, ny, nz

    return target_x, target_y, target_z, max_mass

def apply_gravity_effect(data, w):
    """
    Apply the gravity effect on a localized WxWxW area within the data tensor.
    Store the relatedness information for each pair of blocks in the local tensor.
    """
    
    dx, dy, dz = data.shape[0], data.shape[1], data.shape[2]

    # Calculate relatedness and mass for all blocks in tensor.
    relatedness_map = np.zeros((dx, dy, dz, w, w, w))
    mass_map = np.zeros(data.shape)
    visited = np.zeros(data.shape, dtype=bool)

    # Here we are isolating chunks of space (local_tensor) that do not overlap.
    for x in range(0, dx, w):
        for y in range(0, dy, w):
            for z in range(0, dz, w):
                print(f'Tensor from {x} {y} {z} to {x+w} {y+w} {z+w}')
                local_tensor = data[x:x+w, y:y+w, z:z+w]

                # Iterate through each block in the local tensor
                for lx in range(w):
                    for ly in range(w):
                        for lz in range(w):
                            current_block = local_tensor[lx, ly, lz]

                            # Compare with every other block in the local tensor
                            for nx in range(w):
                                for ny in range(w):
                                    for nz in range(w):
                                        neighbor_block = local_tensor[nx, ny, nz]

                                        # Calculate and store the relatedness
                                        relatedness = calculate_relatedness(current_block, neighbor_block)
                                        relatedness_map[x+lx, y+ly, z+lz, nx, ny, nz] = relatedness

    # Calculate relatedness and mass
    for x in range(0, dx, w):
        for y in range(0, dy, w):
            for z in range(0, dz, w):
                local_tensor = data[x:x+w, y:y+w, z:z+w]
                
                for lx in range(w):
                    for ly in range(w):
                        for lz in range(w):
                            if not visited[x+lx, y+ly, z+lz]:
                                dfs(x+lx, y+ly, z+lz, dx, dy, dz, w, relatedness_map, visited, mass_map, data)

    # Now use the mass_map and relatedness_map to modify data as needed...
    modified_data = data.copy()

    # Apply gravity effect within each local tensor
    for x in range(0, dx, w):
        for y in range(0, dy, w):
            for z in range(0, dz, w):
                local_mass_map = mass_map[x:x+w, y:y+w, z:z+w]
                local_relatedness_map = relatedness_map[x:x+w, y:y+w, z:z+w, :, :, :]

                # We want to find the largest mass in the local tensor for every type of block.

                for lx in range(w):
                    for ly in range(w):
                        for lz in range(w):
                            if modified_data[x+lx, y+ly, z+lz] is not air_block:
                                # We will pull everything towards the largest mass. Find the nearest 
                                target_lx, target_ly, target_lz, target_mass = find_largest_mass_neighbor(lx, ly, lz, local_mass_map, local_relatedness_map)
                                if target_mass > 0:

                                    # How far to pull the block towards target mass.
                                    steps = math.ceil(target_mass / local_mass_map[lx, ly, lz])

                                    # How far we'd have to move in every direction to reach our target.
                                    dlx, dly, dlz = target_lx - lx, target_ly - ly, target_lz - lz

                                    # Not sure what this is.
                                    step_lx, step_ly, step_lz = dlx // steps, dly // steps, dlz // steps

                                    for step in range(1, steps + 1):
                                        # Not sure what this is.
                                        new_lx, new_ly, new_lz = lx + step_lx * step, ly + step_ly * step, lz + step_lz * step
                                        
                                        # Not sure what this is.
                                        if modified_data[x+new_lx, y+new_ly, z+new_lz] is air_block:
                                            modified_data[x+new_lx, y+new_ly, z+new_lz] = modified_data[x+lx, y+ly, z+lz]
                                            modified_data[x+lx, y+ly, z+lz] = 0  # Set current position to air
                                            break

    return modified_data

def postprocess_json(data):
    print("Starting advanced postprocessing...")

    # First, normalize the vectors
    for x in range(len(data)):
        for y in range(len(data[x])):
            for z in range(len(data[x][y])):
                data[x][y][z] = normalize_vector(data[x][y][z])

    # Apply the gravity effect
    data = apply_gravity_effect(data, W)

    print("Advanced postprocessing completed!")
    return data

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_files = len([f for f in os.listdir(generations_dir) if f.endswith(".json")])
    processed_files = 0

    for file_name in os.listdir(generations_dir):
        if file_name.endswith(".json"):
            print(f"Processing {file_name}...")
            with open(os.path.join(generations_dir, file_name), 'r') as f:
                data = json.load(f)
            processed_data = postprocess_json(data)
            wrapped_data = {"matrix": processed_data}  # Wrap the data inside a dictionary
            with open(os.path.join(output_dir, file_name), 'w') as f:
                json.dump(wrapped_data, f, indent=4)  # Dump the wrapped data
            processed_files += 1
            print(f"Finished processing {file_name}. ({processed_files}/{total_files} files completed)")
    
    print("All files processed successfully!")

if __name__ == "__main__":
    main()
