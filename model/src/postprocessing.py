# postprocessing.py

import os
import json
import numpy as np

generations_dir = "generations"
output_dir = "../dataset/server/plugins/BuildLogger/schematics-json-generated"

def normalize_vector_softmax(vector):
    # The first 15 indices are already in one-hot encoded form due to softmax
    processed_vector = vector[:15]

    # For the last bit, apply thresholding
    last_bit = [1 if vector[15] > 0.5 else 0]

    # Combine the sections
    normalized_vector = np.concatenate([processed_vector, last_bit])
    
    return normalized_vector.tolist()

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

# Process the json data and return the modified data.
def postprocess_json(data):
    print("Starting postprocessing...")
    print(len(data))
    print(len(data[0]))
    print(len(data[0][0]))
    print(len(data[0][0][0]))
    print(data[0][0][0][0])
    total_blocks = len(data) * len(data[0]) * len(data[0][0])
    processed_blocks = 0
    for x, xArray in enumerate(data):
        for y, yArray in enumerate(xArray):
            for z, blockVector in enumerate(yArray):
                data[x][y][z] = normalize_vector(blockVector)
                processed_blocks += 1
                if processed_blocks % 10000 == 0:
                    print(f"Processed {processed_blocks} of {total_blocks} blocks...")
    print("Postprocessing completed!")
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
