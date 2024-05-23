# model_block_distributions.py

import sqlite3
from pathlib import Path
from mcschematic import MCSchematic
import datetime

def process_schematics(generations_path, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    schematics_path = Path(generations_path)

    for model_path in schematics_path.iterdir():

        if model_path.is_dir():

            cursor.execute("SELECT id FROM models WHERE name=?", (model_path.name,))
            model_result = cursor.fetchone()
            if model_result:
                model_id = model_result[0]
                #print('model_id:',model_id)
                for schem_file in model_path.glob("*.schem"):
                    cursor.execute("SELECT id FROM model_schematics WHERE name=?", (schem_file.stem,))
                    schem_result = cursor.fetchone()
                    schematic_id = None

                    if not schem_result:
                        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        cursor.execute("INSERT INTO model_schematics (name, filepath, model_id, created_at) VALUES (?, ?, ?, ?)", 
                                    (schem_file.stem, str(schem_file), model_id, current_datetime))
                        schematic_id = cursor.lastrowid
                    else:
                        schematic_id = schem_result[0]

                    # Open the schematic file with MCSchematic
                    schem = MCSchematic(schematicToLoadPath_or_mcStructure=str(schem_file))
                        
                    # Initialize block distribution dictionary
                    block_distributions = {}
                        
                    # Process the schematic file to calculate block distributions
                    # Assuming a method to iterate over all blocks in the schematic
                    for x in range(32):
                        for y in range(32):
                            for z in range(32):
                                block = schem.getBlockDataAt((x, y, z))
                                if block in block_distributions:
                                    block_distributions[block] += 1
                                else:
                                    block_distributions[block] = 1
                        
                    # Update the database with the block distribution
                    for block_id, count in block_distributions.items():
                        #print('block_id, count:', block_id, count)
                        cursor.execute("""
                            INSERT INTO model_block_counts (model_schematic_id, block_id, count) 
                            VALUES (?, ?, ?)
                            ON CONFLICT(model_schematic_id, block_id) 
                            DO UPDATE SET count = excluded.count
                        """, (schematic_id, block_id, count))

    conn.commit()
    conn.close()


if __name__ == "__main__":

    db_path = 'py/structures.db'
    generations_path = 'py/schematics/'

    process_schematics(generations_path, db_path)
    
