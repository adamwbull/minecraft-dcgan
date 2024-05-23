import argparse
import sqlite3
import numpy as np
import os

def get_pattern_by_id(db_path, pattern_id):
    """
    Retrieves the filepath for the given pattern ID and returns the numpy array stored at that path.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT npy_filepath FROM divergence_pattern WHERE id=?", (pattern_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        npy_filepath = result[0]
        
        # Check if the operating system is not Windows
        if os.name != 'nt':
            npy_filepath = npy_filepath.replace("\\", "/")

        pattern = np.load(npy_filepath)
        return pattern
    else:
        return None

def print_array(array):
    """
    Prints the numpy array to the console.
    """
    print(array)

def main():
    parser = argparse.ArgumentParser(description="Inspect a divergence pattern by ID.")
    parser.add_argument("id", type=int, help="ID of the divergence pattern to inspect.")
    args = parser.parse_args()
    
    db_path = "py/structures.db"  # Assuming the database path as per the provided script
    pattern_id = args.id
    
    pattern = get_pattern_by_id(db_path, pattern_id)
    if pattern is not None:
        print("Divergence Pattern (ID: {}):".format(pattern_id))
        print_array(pattern)
    else:
        print("No pattern found with ID:", pattern_id)

if __name__ == "__main__":
    main()
