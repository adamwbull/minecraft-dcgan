# dataset_block_distributions.py

import h5py
import sqlite3
from pathlib import Path
import datetime
import mcschematic
from vector import get_block_id, get_properties

def process_hdf5_schematics(hdf5_path, generations_path, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    schematics_path = Path(generations_path)
    
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        for entry_name in hdf5_file.keys():
            if "rotation_0" in entry_name:  # Only process entries for rotation_0
                data = hdf5_file[entry_name][:]
                # Create a new MCSchematic instance
                schem = mcschematic.MCSchematic()
                
                block_distributions = {}
                
                # Iterate through the data to set blocks and compute distributions
                for x in range(data.shape[0]):
                    for y in range(data.shape[1]):
                        for z in range(data.shape[2]):
                            block_data = data[x, y, z]
                            block_id, block_type_index = get_block_id(block_data)
                            properties = get_properties(block_data, block_type_index)
                            block_string = block_id + (properties if properties else "")
                            schem.setBlock((x, y, z), block_string)
                            block_distributions[block_string] = block_distributions.get(block_string, 0) + 1
                
                # Save the schematic
                schem.save(outputFolderPath=str(schematics_path), schemName=entry_name, version=mcschematic.Version.JE_1_19)
                
                full_location = Path(schematics_path, entry_name)

                cursor.execute("SELECT id FROM dataset_schematics WHERE name=?", (entry_name,))
                schem_result = cursor.fetchone()
                schematic_id = None

                if not schem_result:
                    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    cursor.execute("INSERT INTO dataset_schematics (name, filepath, dataset_id, created_at) VALUES (?, ?, ?, ?)", 
                                (entry_name, str(full_location), 1, current_datetime))
                    print('inserted')
                    schematic_id = cursor.lastrowid
                else:
                    print('already exists')
                    schematic_id = schem_result[0]

                for block_id, count in block_distributions.items():
                    cursor.execute("""
                    INSERT INTO dataset_block_counts (dataset_schematic_id, block_id, count) 
                    VALUES (?, ?, ?)
                    ON CONFLICT(dataset_schematic_id, block_id) 
                    DO UPDATE SET count = excluded.count
                """, (schematic_id, block_id, count))
                conn.commit()

                print(f"Processed and saved {entry_name} with block distributions.")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    hdf5_path = "../../minecraft-GAN/dataset/schematics.hdf5"
    generations_path = "py/schematics/dataset"
    db_path = "py/structures.db"
    
    process_hdf5_schematics(hdf5_path, generations_path, db_path)
