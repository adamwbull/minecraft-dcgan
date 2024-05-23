import os
import json
import numpy as np

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def json_to_tensor(data):
    matrix = data['matrix']
    tensor = np.array(matrix)
    return tensor

def inspect_block_at(tensor, x, y, z):
    if x < tensor.shape[0] and y < tensor.shape[1] and z < tensor.shape[2]:
        return tensor[x, y, z]
    else:
        raise ValueError("Coordinates are out of the tensor's bounds.")

if __name__ == "__main__":
    # File and position parameters
    json_filename = input("Enter the path of the JSON file: ")
    x = int(input("Enter X coordinate: "))
    y = int(input("Enter Y coordinate: "))
    z = int(input("Enter Z coordinate: "))

    # Load JSON and convert to tensor
    data = load_json(json_filename)
    tensor = json_to_tensor(data)

    # Inspect block at given position
    try:
        block_vector = inspect_block_at(tensor, x, y, z)
        print(f"Block vector at ({x}, {y}, {z}): {block_vector}")
    except ValueError as e:
        print(e)
