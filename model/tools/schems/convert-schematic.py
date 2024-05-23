import os
import requests
import nbtlib
import numpy as np
import base64

# PLEASE NOTE: NOT UP TO DATE WITH DATBASE STANDARDS.

# Initial converter made to work with .schematic files. I will be doing
# a new convert file that uses the 1.13+ .schem format, which I sadly
# realized is way more common and a completely different format.

# It is also constructing matrices by placing block IDs directly.
# After learning more about cDCGANs and initial testing on 2D images,
# I realize the numerical data represented needs to be more continuous
# than the categorical nature of the traditional block IDs.
# Next version of this converter will address that.

# Specify software version for converted schematics.
convert_version = 'schem-1'

# Block simplification.
wood_types = list(range(5, 8+1))
stair_types = [53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163, 164, 180, 203]
wool_types = list(range(35, 22+1))

# Take our nbtlib schematic and convert it to an integer matrix for training.
def schematic_to_matrix(schematic):
    width = schematic['Width']
    height = schematic['Height']
    length = schematic['Length']
    blocks = np.array(schematic['Blocks'])
    data = np.array(schematic['Data'])
    matrix = np.empty((width, height, length), dtype=np.int8)
    xMin, yMin, zMin = width, height, length
    xMax, yMax, zMax = 0, 0, 0

    for x in range(width):
        for y in range(height):
            for z in range(length):
                i = y * length * width + z * width + x
                block = blocks[i]

                # Handle negative block ids because of signed ints.
                if block < 0:
                    block += 256
                    
                # Simplify block types
                if block in wood_types:
                    block = 5  # Change all types of wood to Oak
                elif block in stair_types:
                    block = 53  # Change all types of stairs to Oak stairs
                elif block in wool_types:
                    block = 35  # Change all types of wool to White

                matrix[x, y, z] = block

                # Track boundaries of non-air blocks (assuming air block has ID 0)
                if block != 0:
                    xMin = min(xMin, x)
                    xMax = max(xMax, x)
                    yMin = min(yMin, y)
                    yMax = max(yMax, y)
                    zMin = min(zMin, z)
                    zMax = max(zMax, z)
                    
    # Create shrunk matrix
    shrunk_matrix = matrix[xMin:xMax+1, yMin:yMax+1, zMin:zMax+1]

    return shrunk_matrix

# Upload all schematics in ./Schematics to remote database.
def process_schematics_in_folder(folder_path):
    api_token = os.getenv("API_TOKEN")
    if not api_token:
        raise Exception("\n\nAPI token not found in environment variables.\nPotential solutions:\n- Run `export API_TOKEN=...` command (for Unix systems)\n- Run `set API_TOKEN=...` command (for Windows)\n- Set up API_TOKEN in Windows or bash environment variables\nand run the script again.\n")

    api_url = 'https://ganapi.adambullard.com/entry'
    headers = {
        'Authorization': 'Bearer ' + api_token,
        'Content-Type': 'application/json'
    }

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.schematic'):

                print(f"Processing schematic: {file}")

                labels = input("Enter labels for this schematic (comma separated): ")
                target = os.path.join(root, file)
                schematic = nbtlib.load(target)

                print("Converting schematic to matrix...")
                
                matrix = schematic_to_matrix(schematic)
                matrix_data = np.array(matrix, dtype=np.int8).tobytes()  # Convert the matrix to bytes
                matrix_data_base64 = base64.b64encode(matrix_data).decode()  # Convert the bytes to Base64 string
                data = {
                    'labels': labels,
                    'data': matrix_data_base64,
                    'dim_x': matrix.shape[0],
                    'dim_y': matrix.shape[1],
                    'dim_z': matrix.shape[2],
                    'source': convert_version,
                    'schematic_name': file
                }

                print("Matrix created! Uploading to database...")

                response = requests.post(api_url, headers=headers, json=data)
                if response.status_code in (200, 201):  # check for success
                    print(f"Schematic {file} uploaded successfully.")
                else:
                    print(f"Error uploading schematic {file}: {response.content}")

process_schematics_in_folder('./Schematics')
