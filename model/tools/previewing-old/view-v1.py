import requests
import numpy as np
import os
import base64

api_token = os.getenv("API_TOKEN")
headers = {
    'Authorization': 'Bearer ' + api_token,
}

def get_entry(id):
    response = requests.get(f'https://ganapi.adambullard.com/entry/{id}', headers=headers)

    if response.status_code == 200:
        entry = response.json()

        # Get dimensions
        dim_x = entry['dim_x']
        dim_y = entry['dim_y']
        dim_z = entry['dim_z']

        # Print the length of the base64 string
        print(f"Length of base64 string: {len(entry['data'])}")

        # Decode the binary data.
        matrix_data = base64.b64decode(entry['data'])
        #matrix_data = base64.b64decode(matrix_data)

        # Print the length of the decoded data
        print(f"Length of decoded data: {len(matrix_data)}")

        matrix = np.frombuffer(matrix_data, dtype=np.int8)

        print(f"Actual size: {matrix.size}")
        print(f"Expected size: {dim_x * dim_y * dim_z}")

        # Reshape to 3D matrix
        matrix = matrix.reshape((dim_x, dim_y, dim_z))

        print(f"Schematic Name: {entry['schematic_name']}")
        print(f"Labels: {entry['labels']}")
        print(f"Source: {entry['source']}")
        print(f"3D Matrix:\n{matrix}")
    else:
        print(f"Error: {response.status_code}")

id = input("Enter the ID of the entry you want to view: ")
get_entry(id)
