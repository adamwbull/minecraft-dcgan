# pattern_distributions.py

import argparse
import sqlite3
import numpy as np
from pathlib import Path
from mcschematic import MCSchematic
import random
import os
from scipy.stats import entropy
from vector import string_to_vector
import time
from generate_structure import initialize_generator, generate_structure, init_postprocess_structure, convert_to_schem_and_encode

testing = False
dev_model_id = 16
dev_id = -1 # Determined between model and dataset schematics in get_random_schematics.
epsilon = 1e-10

def calculate_kl_divergence(cursor, dataset_schematics, model_schematics, pattern_dim, sample_size, dataset_id, model_id):
    
    # Count pattern occurences across our entire dataset schematic set.
    dataset_patterns = {}
    for index, schematic in enumerate(dataset_schematics):

        cursor.execute("""
            SELECT id, divergence_pattern_id, dataset_schematic_id, count FROM divergence_pattern_relationship
            WHERE dataset_schematic_id=?""", (schematic[0],))
        
        data = cursor.fetchall()

        # For every pattern that occurs in this schematic...
        for relationship in data:
            dataset_patterns[relationship[1]] = dataset_patterns.get(relationship[1], 0) + relationship[3]

    # Count pattern occurrences across our entire model schematic set.
    model_patterns = {}
    for index, schematic in enumerate(model_schematics):
        cursor.execute("""
            SELECT id, divergence_pattern_id, model_schematic_id, count FROM divergence_pattern_relationship
            WHERE model_schematic_id=?""", (schematic[0],))
        data = cursor.fetchall()
        # For every pattern that occurs in this schematic...
        for relationship in data:
            # Check if key exists in model_patterns, if not, use 0 as the default value
            current_count = model_patterns.get(relationship[1], 0)
            model_patterns[relationship[1]] = current_count + relationship[3] # count

     # Calculate total number of pattern occurrences
    dataset_patterns_total = sum(dataset_patterns.values())
    model_patterns_total = sum(model_patterns.values())

    log(f"dataset_patterns_total: {dataset_patterns_total} | model_patterns_total: {model_patterns_total}")

    # Step 1: Identify all unique patterns
    all_patterns = set(dataset_patterns.keys()).union(set(model_patterns.keys()))

    # Step 2: Build complete and equal-length probability arrays
    dataset_complete_probs = [dataset_patterns.get(pattern, 0) / dataset_patterns_total for pattern in all_patterns]
    model_complete_probs = [model_patterns.get(pattern, 0) / model_patterns_total for pattern in all_patterns]

    # Step 3: Normalize and adjust these arrays
    dataset_adjusted = [(prob + epsilon) / (1 + len(dataset_complete_probs) * epsilon) for prob in dataset_complete_probs]
    model_adjusted = [(prob + epsilon) / (1 + len(model_complete_probs) * epsilon) for prob in model_complete_probs]

    # Calculate KL divergence.
    dataset_model_kl = entropy(dataset_adjusted, model_adjusted)
    model_dataset_kl = entropy(model_adjusted, dataset_adjusted)

    log(f"dataset_model_kl: {dataset_model_kl}")
    log(f"model_dataset_kl: {model_dataset_kl}")

    w0_score = dataset_model_kl
    whalf_score = (0.5*dataset_model_kl) + (0.5*model_dataset_kl)
    w1_score = model_dataset_kl

    # Store the divergence score and sample references
    cursor.execute("""
        INSERT INTO divergence_score (w0_score, whalf_score, w1_score, pattern_dim, sample_size, dataset_id, model_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (w0_score, whalf_score, w1_score, pattern_dim, sample_size, dataset_id, model_id))
    divergence_score_id = cursor.lastrowid

    # Store references for dataset patterns
    for schematic in dataset_schematics:
        cursor.execute("""
            SELECT id FROM divergence_pattern_relationship
            WHERE dataset_schematic_id=?
        """, (schematic[0],))
        relationship_ids = cursor.fetchall()
        for rel_id in relationship_ids:
            cursor.execute("""
                INSERT INTO divergence_score_sample (divergence_score_id, divergence_pattern_relationship_id)
                VALUES (?, ?)
            """, (divergence_score_id, rel_id[0]))

    # Store references for model patterns
    for schematic in model_schematics:
        cursor.execute("""
            SELECT id FROM divergence_pattern_relationship
            WHERE model_schematic_id=?
        """, (schematic[0],))
        relationship_ids = cursor.fetchall()
        for rel_id in relationship_ids:
            cursor.execute("""
                INSERT INTO divergence_score_sample (divergence_score_id, divergence_pattern_relationship_id)
                VALUES (?, ?)
            """, (divergence_score_id, rel_id[0]))

    log(f"Stored KL divergence score and sample references for dataset_id {dataset_id} and model_id {model_id}")

# Logging function
def log(text="", console_only=False):
    current_date = time.strftime("%Y%m%d")
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")

    text = str(text)

    print(timestamp + " " + text)
    
    if not console_only:
        # Write to the daily log file
        daily_log_filename = f'./logs/{current_date}.log'
        with open(daily_log_filename, 'a') as daily_log_file:
            daily_log_file.write(f"{timestamp} {text}\n")

# Function to manage log files
def initialize_logs():
    if not os.path.exists('./logs'):
        os.makedirs('./logs')

# Initialize log files
initialize_logs()

def patterns_are_equal(pattern1, pattern2):
    """Compares two patterns (numpy arrays) for equality."""
    return np.array_equal(pattern1, pattern2)

def store_pattern_if_new(cursor, pattern, pattern_dim, in_memory_patterns):
    for pattern_id, existing_pattern in in_memory_patterns.items():
        if patterns_are_equal(pattern, existing_pattern):
            return pattern_id  # Pattern already exists, return its ID

    # If the pattern is new, store it in memory and DB
    npy_filename = f"pattern_{random.randint(100000, 999999)}.npy"
    npy_filepath = os.path.join("py/patterns", npy_filename)
    np.save(npy_filepath, pattern)
    cursor.execute("INSERT INTO divergence_pattern (npy_filepath, pattern_dim) VALUES (?, ?)", (npy_filepath, pattern_dim))
    new_id = cursor.lastrowid
    in_memory_patterns[new_id] = pattern  # Update in-memory patterns
    return new_id  # Return the new pattern's ID

def extract_patterns_from_array(array, pattern_dim):
    """Extracts and returns patterns of the specified dimensions from the given array."""
    patterns = []
    for x in range(array.shape[0] - pattern_dim + 1):
        for y in range(array.shape[1] - pattern_dim + 1):
            for z in range(array.shape[2] - pattern_dim + 1):
                # Extract a sub-array (pattern) from the main array
                pattern = array[x:x+pattern_dim, y:y+pattern_dim, z:z+pattern_dim]
                patterns.append(pattern)
                
    return patterns

def process_schematic(schematic_file, pattern_dim, testing=False):

    schematic_file = schematic_file.replace("\\", "/")
    schem = MCSchematic(schematicToLoadPath_or_mcStructure=schematic_file)

    dimensions = (32, 32, 32)
    vector_dim = 16 
    schematic_array = np.zeros(dimensions + (vector_dim,), dtype=np.float32)

    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            for z in range(dimensions[2]):
                block = schem.getBlockDataAt((x, y, z))
                block_vector = string_to_vector(block)
                schematic_array[x, y, z] = block_vector

    patterns = extract_patterns_from_array(schematic_array, pattern_dim)

    return patterns


def store_pattern_relationship(cursor, schematic_id, pattern_id, table_prefix):

    sql_query = f"""
        SELECT id, count FROM divergence_pattern_relationship
        WHERE {table_prefix}_schematic_id=? AND divergence_pattern_id=?
    """

    cursor.execute(sql_query, (schematic_id, pattern_id))
    relationship = cursor.fetchone()
    if relationship:
        rel_id, count = relationship
        cursor.execute("""
            UPDATE divergence_pattern_relationship
            SET count=?
            WHERE id=?
        """, (count + 1, rel_id))
    else:
        insert_query = f"""
            INSERT INTO divergence_pattern_relationship (divergence_pattern_id, {table_prefix}_schematic_id, count)
            VALUES (?, ?, 1)
        """
        cursor.execute(insert_query, (pattern_id, schematic_id))
        
def generate_and_store_samples(model_name, model_type, rga, db_path, samples_needed):
    """
    Generate additional samples for a given model if the sample size is not met.
    """
    # Initialize generator and decoder
    generator = initialize_generator(model_name, model_type, rga)
    
    # Generate and process a specified number of samples
    for _ in range(samples_needed):
        structure = generate_structure(generator, model_type)
        postprocessed_structure = init_postprocess_structure(structure, model_type)
        filename, base64_schem_content = convert_to_schem_and_encode(postprocessed_structure, model_name, model_type, db_path=db_path)
        log(f"Generated and stored new sample: {filename}")

def get_random_schematics(cursor, type, type_id, sample_size, pattern_dim=None, prioritize_finished=True):
    """
    Fetches random schematics from the database, optionally prioritizing schematics that are marked as finished for a given pattern dimension.
    """
    if prioritize_finished and pattern_dim is not None:
        # Query to prioritize schematics that have entries in divergence_schematic_finished for the given pattern_dim
        sql_query = f"""
        SELECT s.id, s.filepath FROM {type}_schematics s
        LEFT JOIN divergence_schematic_finished df ON s.id = df.{type}_schematic_id AND df.pattern_dim=?
        WHERE s.{type}_id=?
        ORDER BY df.id IS NULL, RANDOM()
        LIMIT ?
        """
        cursor.execute(sql_query, (pattern_dim, type_id, sample_size))
    else:
        # Default behavior: Random selection without prioritization
        sql_query = f"""
        SELECT id, filepath FROM {type}_schematics
        WHERE {type}_id=?
        ORDER BY RANDOM()
        LIMIT ?
        """
        cursor.execute(sql_query, (type_id, sample_size))

    return cursor.fetchall()

def mark_schematic_as_finished(cursor, schematic_id, pattern_dim, table_prefix):
    """
    Marks a schematic as processed by creating an entry in the divergence_schematic_finished table.

    """
    if table_prefix == 'model':
        cursor.execute("""
            INSERT INTO divergence_schematic_finished (pattern_dim, model_schematic_id)
            VALUES (?, ?)
        """, (pattern_dim, schematic_id))
    elif table_prefix == 'dataset':
        cursor.execute("""
            INSERT INTO divergence_schematic_finished (pattern_dim, dataset_schematic_id)
            VALUES (?, ?)
        """, (pattern_dim, schematic_id))

def store_pattern_relationship_bulk(cursor, schematic_id, pattern_counts, table_prefix):
    """
    Bulk update pattern relationships in the database based on accumulated pattern counts.
    """
    for pattern_id, count in pattern_counts.items():
        cursor.execute(f"""
            SELECT id FROM divergence_pattern_relationship
            WHERE {table_prefix}_schematic_id=? AND divergence_pattern_id=?
        """, (schematic_id, pattern_id))
        relationship = cursor.fetchone()

        if relationship:
            rel_id = relationship[0]
            cursor.execute("""
                UPDATE divergence_pattern_relationship
                SET count=?
                WHERE id=?
            """, (count, rel_id))
        else:
            cursor.execute(f"""
                INSERT INTO divergence_pattern_relationship (divergence_pattern_id, {table_prefix}_schematic_id, count)
                VALUES (?, ?, ?)
            """, (pattern_id, schematic_id, count))

def process_schematic_patterns(cursor, schematic_id, patterns, pattern_dim, table_prefix, in_memory_patterns):
    """
    Processes patterns for a given schematic, updating the cache for each found pattern.
    """
    pattern_counts = {}
    for pattern in patterns:
        pattern_id = store_pattern_if_new(cursor, pattern, pattern_dim, in_memory_patterns)
        pattern_counts[pattern_id] = pattern_counts.get(pattern_id, 0) + 1

    store_pattern_relationship_bulk(cursor, schematic_id, pattern_counts, table_prefix)

def check_divergence_schematic_finished(cursor, schematic_id, pattern_dim, table_prefix):
    table_column = f"{table_prefix}_schematic_id"
    cursor.execute(f"""
        SELECT id FROM divergence_schematic_finished
        WHERE {table_column}=? AND pattern_dim=?
    """, (schematic_id, pattern_dim))
    return cursor.fetchone() is not None

def main(db_path, pattern_dim, sample_size):

    log(f"Computing KL Divergence with pattern_dim: {pattern_dim}, sample_size: {sample_size}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Load models.
    models_query = f"""
    SELECT id, name, model_type, rga FROM models 
    WHERE hidden=0 
    AND NOT EXISTS (
        SELECT 1 FROM divergence_score
        WHERE divergence_score.model_id = models.id
        AND divergence_score.pattern_dim = {pattern_dim}
        AND divergence_score.sample_size = {sample_size}
    )
    """
    models = cursor.execute(models_query).fetchall()
    datasets = cursor.execute("SELECT id, name FROM datasets").fetchall()

    log(f"Found {len(datasets)} datasets and {len(models)} models to process.")

    # Load all patterns into memory
    cursor.execute("SELECT id, npy_filepath FROM divergence_pattern WHERE pattern_dim=?", (pattern_dim,))
    pattern_data = cursor.fetchall()
    in_memory_patterns = {pattern_id: np.load(npy_filepath) for pattern_id, npy_filepath in pattern_data}

    for dataset in datasets:
        dataset_schematics = get_random_schematics(cursor, "dataset", dataset[0], sample_size, pattern_dim)

        # Process dataset schematics to ensure patterns and relationships exist
        ds_processed = 0
        for ds_id, ds_filepath in dataset_schematics:

            ds_processed = ds_processed + 1

            if check_divergence_schematic_finished(cursor, ds_id, pattern_dim, 'dataset'):
                log(f"Skipping already processed dataset schematic: ID {ds_id}")
                continue

            ds_filepath = ds_filepath + ".schem" if not ds_filepath.endswith(".schem") else ds_filepath
            patterns = process_schematic(ds_filepath, pattern_dim, True)
            process_schematic_patterns(cursor, ds_id, patterns, pattern_dim, 'dataset', in_memory_patterns)
            mark_schematic_as_finished(cursor, ds_id, pattern_dim, 'dataset')
            conn.commit()
            log(f"{ds_processed}/{len(dataset_schematics)}: {str(ds_filepath)} complete.")

        log("All dataset associations complete.")

        # Let's calculate a kl divergence between our dataset for every model
        for model in models:
            model_schematics = get_random_schematics(cursor, "model", model[0], sample_size, pattern_dim)
            original_model_schematics_count = len(model_schematics)

            # Gen more samples from this model if necessary.
            if original_model_schematics_count < sample_size:
                samples_needed = sample_size - original_model_schematics_count
                log(f"samples_needed:{samples_needed} sample_size:{sample_size} original_model_schematics_count:{original_model_schematics_count}")
                generate_and_store_samples(model[1], model[2], int(model[3]), db_path, samples_needed)
                model_schematics = get_random_schematics(cursor, "model", model[0], sample_size)

            if original_model_schematics_count == sample_size:
                log("Valid model found: " + str(model))
                # Process model schematics to ensure patterns and relationships exist
                ms_processed = 0
                for ms_id, ms_filepath in model_schematics:

                    ms_processed = ms_processed + 1
                    
                    if check_divergence_schematic_finished(cursor, ms_id, pattern_dim, 'model'):
                        log(f"Skipping already processed model schematic: ID {ms_id}")
                        continue

                    ms_filepath = ms_filepath + ".schem" if not ms_filepath.endswith(".schem") else ms_filepath
                    patterns = process_schematic(ms_filepath, pattern_dim)
                    process_schematic_patterns(cursor, ms_id, patterns, pattern_dim, 'model', in_memory_patterns)
                    mark_schematic_as_finished(cursor, ms_id, pattern_dim, 'model')
                    conn.commit()
                    
                    log(f"{ms_processed}/{len(model_schematics)}: {str(ms_filepath)} complete.")

                log(f"Model {model[0]} associations complete.")

                calculate_kl_divergence(cursor, dataset_schematics, model_schematics, pattern_dim, sample_size, dataset[0], model[0])
                conn.commit()

    conn.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute and store matrix pattern KL divergences.")

    args = parser.parse_args()

    db_path = "py/structures.db"

    sample_size = 10
    pattern_dims = [4]

    for pattern_dim in pattern_dims:
        main(db_path, pattern_dim, sample_size)
