import numpy as np
import os
from pathlib import Path
from nbtlib import File, Compound, List, IntArray, Int
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def convert_to_mcschematic(npy_array, schematic_name):
    height, length, width, _ = npy_array.shape

    #flatten the 3D numpy array (ignoring the dimension) to a 1D list
    block_data_flat = npy_array[:, :, :, 0].flatten().astype(int)

    schematic_tags = {
        'Width': width,
        'Length': length,
        'Height': height,
        'PaletteMax': int(np.max(block_data_flat)) + 1,
        'Palette': {},  #populated with block IDs
        'BlockData': List([]),  
        'Entities': List([]),
        'TileEntities': List([]),
        'Version': 1976
    }

    unique_blocks = np.unique(block_data_flat)
    palette_mapping = {block_id: f'block_{block_id}' for block_id in unique_blocks}
    reverse_palette_mapping = {v: k for k, v in palette_mapping.items()}

    schematic_tags['Palette'] = palette_mapping

    #map block data to palette indices
    block_data_indices = [reverse_palette_mapping[f'block_{id}'] for id in block_data_flat]

    #convert block data indices to NBT Int objects and assign to 'BlockData'
    schematic_tags['BlockData'] = List([Int(i) for i in block_data_indices])

    logging.info(f"Attempting to save mcschematic to: {schematic_name}")

    #create NBT file
    try:
        File(Compound({'Schematic': Compound(schematic_tags)})).save(schematic_name)
        logging.info(f"Successfully saved mcschematic: {schematic_name}")
    except Exception as e:
        logging.error(f"Failed to save mcschematic: {e}")



def process_schematic_to_mcschematic(file_path, output_path):
    try:
        npy_array = np.load(file_path)

        if npy_array.ndim != 4:
            logging.error(f"Array at {file_path} is not 4D. It has {npy_array.ndim} dimensions.")
            return

        house_name = Path(file_path).stem
        schematic_name = f"{house_name}.schematic"
        output_file = os.path.join(output_path, schematic_name)

        logging.info(f"Processed and attempting to save mcschematic: {output_file}")

        convert_to_mcschematic(npy_array, output_file)

        logging.info(f"Processed and saved mcschematic: {output_file}")
    except Exception as e:
        logging.error(f"Failed to process {file_path}: {e}")

def main():
    input_directory = "./houses/"
    output_directory = "./mcschematic/"
    Path(output_directory).mkdir(exist_ok=True)

    logging.info(f"Script started. Output directory: {output_directory}")

    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith('schematic.npy'):
                file_path = os.path.join(root, file)
                process_schematic_to_mcschematic(file_path, output_directory)

    logging.info("Script completed.")

if __name__ == "__main__":
    main()



