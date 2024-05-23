# postprocessing.py

import os
import json
import numpy as np
from numpy import inf
import math

generations_dir = "generations"
output_dir = "../dataset/server/plugins/BuildLogger/schematics-json-generated"
W = 4  # Default value for the size of the local relatedness tensor

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


# DFS for identifying masses.
# 1-3: The location of the current block being explored in our local tensor.
# 4-6: The location of the current block in the global tensor.
# 7-9: The total lengths of each dimension for the global tensor
# 10: The length of the local tensor dimensions.
# 11: Our DFS visit tracking.
# 12: Our mass map we are computing.
# 13 Original data for computing whether blocks are of the same block type.
def dfs(lx, ly, lz, x, y, z, dx, dy, dz, w, visited, mass_map, data):
    """
    Explore blobs of blocks via DFS to calculate blob masses.
    """
    if x < 0 or x >= dx or y < 0 or y >= dy or z < 0 or z >= dz or visited[x, y, z]:
        return 0

    visited[x, y, z] = True
    current_block = data[x, y, z]
    block_type = find_first_index(current_block)

    mass = 0 if is_air_block(current_block) else 1
    
    for nx in range(-1, 2):
        for ny in range(-1, 2):
            for nz in range(-1, 2):
                # Ensure we are within the local tensor.
                if 0 <= nx + lx < w and 0 <= ny + ly < w and 0 <= nz + lz < w:
                    # Ensure we are within the overall matrix.
                    if 0 <= nx + x < dx and 0 <= ny + y < dy and 0 <= nz + z < dz:
                        # Ensure this is the same block type with n adjustment.
                        if block_type == find_first_index(data[nx+x, ny+y, nz+z]):
                            mass += dfs(nx+lx, ny+ly, nz+lz, nx+x, ny+y, nz+z, dx, dy, dz, w, visited, mass_map, data)

    mass_map[x, y, z] = mass
    return mass

# Define an air block's shape.
air_block = np.zeros((16,))
air_block[0] = 1

def is_air_block(arr):
    return arr[0] == 1

# Simply return the position of the block type index.
def find_first_index(arr):
    i = 0
    for element in arr:
        if element == 1:
            return i
        i += 1
    return 0

# Function to check and update adjacent air blocks
def check_and_update_adjacent_air_blocks(lx, ly, lz, block_type, modified_data, adjacent_air_blocks, w, x, y, z):
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if 0 <= lx+dx < w and 0 <= ly+dy < w and 0 <= lz+dz < w:
                    adj_block = modified_data[x+lx+dx, y+ly+dy, z+lz+dz]
                    if is_air_block(adj_block):
                        air_coord = (lx+dx, ly+dy, lz+dz)
                        adjacent_air_blocks[block_type][air_coord] = adjacent_air_blocks[block_type].get(air_coord, 0) + 1

# Use localized mass map to find largest masses for every block type.
def find_largest_masses(local_mass_map, modified_data, w, x, y, z):

    # Initialize an array to store the coordinates of the largest mass for each block type
    largest_masses = [[(-1, -1, -1)] for _ in range(11)]
    # Initialize an array to store the coordinates of air blocks adjacent to the largest mass for each block type
    adjacent_air_blocks = [{} for _ in range(11)]
    max_masses = [0] * 11  # Store the largest mass found for each block type

    for lx in range(w):
        for ly in range(w):
            for lz in range(w):
                block = modified_data[x+lx, y+ly, z+lz]
                block_type = find_first_index(block)
                mass = local_mass_map[lx, ly, lz]

                if mass >= max_masses[block_type]:

                    # Process the mass differently if its the first time we've seen it or not.
                    if mass > max_masses[block_type]:
                        max_masses[block_type] = mass
                        # Wipe out old data, no longer relevant.
                        largest_masses[block_type] = [(lx, ly, lz)]
                        adjacent_air_blocks[block_type].clear()
                    elif mass == max_masses[block_type]:
                        # If the mass is equal to the current max, we add this block's coordinates
                        largest_masses[block_type].append((lx, ly, lz))

                    # Either way, we are tracking air blocks for this entry.
                    check_and_update_adjacent_air_blocks(lx, ly, lz, block_type, modified_data, adjacent_air_blocks, w, x, y, z)

    # Sort air blocks by their counts in descending order
    adjacent_air_blocks_sorted = []
    for air_blocks in adjacent_air_blocks:
        sorted_air_blocks = sorted(air_blocks.items(), key=lambda item: item[1], reverse=True)
        adjacent_air_blocks_sorted.append([coords for coords, count in sorted_air_blocks])

    return largest_masses, adjacent_air_blocks_sorted

def apply_gravity_effect(data, w):
    """
    Store the relatedness map for each pair of blocks in the local tensor.
    Calculate mass map using DFS.
    Apply the gravity effect on a localized WxWxW area within the data tensor using relatedness and mass maps.
    """
    
    dx, dy, dz = data.shape[0], data.shape[1], data.shape[2]

    # Calculate relatedness and mass for all blocks in tensor.
    relatedness_map = np.zeros((dx, dy, dz, w, w, w))
    mass_map = np.zeros((data.shape[0], data.shape[1], data.shape[2], 1))
    visited = np.zeros((data.shape[0], data.shape[1], data.shape[2], 1), dtype=bool)

    # Here we are isolating chunks of space (local_tensor) that do not overlap.
    print("Generating mass_map...")
    # Calculate masses in each local tensor of w^3 size.
    for x in range(0, dx, w):
        for y in range(0, dy, w):
            for z in range(0, dz, w):
                # Explore the local tensor.
                for lx in range(w):
                    for ly in range(w):
                        for lz in range(w):
                            if x+lx < dx and y+ly < dy and z+lz < dz:
                                if not visited[x+lx, y+ly, z+lz]:
                                    dfs(lx, ly, lz, x+lx, y+ly, z+lz, dx, dy, dz, w, visited, mass_map, data)

    # Now use the mass_map and relatedness_map to modify data as needed...
    modified_data = data.copy()

    print('Applying gravity effect...')
    # Apply gravity effect within each local tensor
    for x in range(0, dx, w):
        for y in range(0, dy, w):
            for z in range(0, dz, w):
                local_mass_map = mass_map[x:x+w, y:y+w, z:z+w]

                # We want to find the nearest edge of the largest mass in the local tensor for every type of block.

                # Find the largest masses for each block type in the local tensor
                largest_masses, adjacent_air_blocks = find_largest_masses(local_mass_map, modified_data, w, x, y, z)

                for lx in range(w):
                    for ly in range(w):
                        for lz in range(w):
                            
                            # Identify this block and its type.
                            block_location = [x+lx, y+ly, z+lz]
                            block = modified_data[x+lx, y+ly, z+lz]
                            block_type = find_first_index(block)

                            if block is not air_block:
                                
                                # Ensure this block is not part of the largest mass.
                                separate = True
                                for element in largest_masses[block_type]:
                                    if block_location == element:
                                        separate = False
                                        break

                                if separate:
                                    # Move the block towards the top entry in adjacent_air_blocks[block_type]
                                    if adjacent_air_blocks[block_type]:

                                        placed = False
                                        while not placed:
                                            # Move this block if we have an unoccupied place to gravitate it towards.
                                            if len(adjacent_air_blocks[block_type]) > 0:
                                                # Get the best candidate air block location
                                                best_air_block_location = adjacent_air_blocks[block_type][0]
                                                
                                                # Ensure this air block is available and wasn't filled by another block type.
                                                target = modified_data[best_air_block_location[0] + x, best_air_block_location[1] + y, best_air_block_location[2] + z]
                                                if is_air_block(target):

                                                    placed = True
                                                    
                                                    # Move block to the best candidate air block
                                                    modified_data[best_air_block_location[0] + x, best_air_block_location[1] + y, best_air_block_location[2] + z] = block

                                                    # Set the current block location to air
                                                    modified_data[block_location[0], block_location[1], block_location[2]] = air_block

                                                # Remove the used air block entry to prevent reuse
                                                adjacent_air_blocks[block_type].pop(0)
                                            else:
                                                # We haven't placed it, but we've ran out of adjacent air blocks to gravitate it towards.
                                                placed = True
    return modified_data

def postprocess_json(data):
    print("Starting advanced postprocessing...")

    data = np.array(data)

    # First, normalize the vectors
    for x in range(len(data)):
        for y in range(len(data[x])):
            for z in range(len(data[x][y])):
                data[x][y][z] = normalize_vector(data[x][y][z])

    # Apply the gravity effect
    data = apply_gravity_effect(data, W)

    # back to list form
    data = data.tolist()

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
            with open(os.path.join(output_dir, f'gravity{W}_'+file_name), 'w') as f:
                json.dump(wrapped_data, f, indent=4)  # Dump the wrapped data
            processed_files += 1
            print(f"Finished processing and saved gravity{W}_{file_name}. ({processed_files}/{total_files} files completed)")
    
    print("All files processed successfully!")

if __name__ == "__main__":
    main()
