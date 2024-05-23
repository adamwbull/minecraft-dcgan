from mcschematic import MCSchematic, Version
from vector import vector_to_string
import numpy as np
import os

def convert_npy_to_schematics(input_folder='./test-schematics/'):
    for file in os.listdir(input_folder):
        if file.endswith('.npy'):
            npy_path = os.path.join(input_folder, file)
            structure = np.load(npy_path)

            # Initialize MCSchematic
            schem = MCSchematic()
            for x in range(structure.shape[0]):
                for y in range(structure.shape[1]):
                    for z in range(structure.shape[2]):
                        block_value = structure[x, y, z]
                        block_string = vector_to_string(block_value) 
                        schem.setBlock((x, y, z), block_string)

            # Save schematic with the same name as the .npy but with .schem extension
            schem_name = file.replace('.npy', '')
            schem.save(outputFolderPath=input_folder, schemName=schem_name, version=Version.JE_1_19)
            print(f"Converted {file} to {schem_name}")

if __name__ == "__main__":
    convert_npy_to_schematics()