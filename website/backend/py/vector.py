import numpy as np

block_ids = {

    0: "minecraft:air",
    1: "minecraft:dirt",
    2: "minecraft:stone",
    3: "minecraft:cobblestone",
    4: "minecraft:stone_bricks",
    6: "minecraft:oak_planks",
    7: "minecraft:oak_log",
    9: "minecraft:glass",
    10: "minecraft:white_wool",
    8: "minecraft:oak_stairs",
    5: "minecraft:oak_slab",
}

direction_indices = {

    11: "north",
    12: "east",
    13: "south",
    14: "west",

}

axis_indices = {

    11: "x",
    12: "y",
    13: "z",

}

# Invert the block_ids dictionary for easy lookup
id_to_index = {v: k for k, v in block_ids.items()}

# Invert the properties dictionaries for easy lookup
direction_to_index = {v: k for k, v in direction_indices.items()}
axis_to_index = {v: k for k, v in axis_indices.items()}

def string_to_vector(block_string):
    """Convert a Minecraft string  to its block vector representation."""
    # Initialize a 16-length zero vector
    vector = np.zeros(16, dtype=np.float32)

    # Split the block string to get the block type and properties
    parts = block_string.split("[")
    block_type = parts[0]
    properties = parts[1][:-1] if len(parts) > 1 else ""

    # Set the block type index
    block_index = id_to_index.get(block_type, 0)  # Default to air if not found
    vector[block_index] = 1

    # Parse properties if any
    if properties:
        props = properties.split(",")
        for prop in props:
            key, value = prop.split("=")
            if key == "axis" and value in axis_to_index:
                vector[axis_to_index[value]] = 1
            elif key == "facing" and value in direction_to_index:
                vector[direction_to_index[value]] = 1
            elif key == "type" and value == "top":
                vector[15] = 1

    return vector

def vector_to_string(vector):
    """Convert a block vector to its Minecraft string representation."""
    block_id, block_type_index = get_block_id(vector)
    properties_str = get_properties(vector, block_type_index)
    if properties_str:
        return f"{block_id}{properties_str}"
    return block_id

def get_block_id(vector):
    """Retrieve block ID from the vector."""
    block_type_index = np.argmax(vector[:11])  # Find the index of the block type
    block_id = block_ids.get(block_type_index, "minecraft:air")  # Default to air if not found
    return block_id, block_type_index

def get_properties(vector, block_type_index):
    """Construct string with properties for the block."""

    properties = []

    # Only do this if we are slab (5)
    if block_type_index == 5:
        if vector[15] == 1:  # Check for additional property (e.g., top half)
            properties.append("type=top,waterlogged=false")
        else:
            properties.append("type=bottom,waterlogged=false")

    # Only do this if we are log (7)
    if block_type_index == 7:
        for index, value in enumerate(vector[11:15], start=11):
            if value == 1:
                if index in axis_indices:
                    properties.append(f"axis={axis_indices[index]}")
        
        # Add to axis y for this log if none other were found.
        if not properties:
            properties.append(f"axis=y")
    
    # Only do this if we are stairs (8)
    if block_type_index == 8:
        for index, value in enumerate(vector[11:15], start=11):
            if value == 1:
                if index in direction_indices:
                    properties.append(f"facing={direction_indices[index]},half=bottom,shape=straight,waterlogged=false")
    
    return "[" + ",".join(properties) + "]" if properties else ""

# Function to generate all unique block vector possibilities
def get_unique_block_vectors():
    unique_blocks = []

    # Iterate through all block types
    for block_id in block_ids.keys():
        if block_id in [5, 7, 8]:  # For slabs, logs, and stairs which have additional properties
            if block_id == 5:  # Slabs
                for top in [0, 1]:  # Generate for bottom and top types
                    vector = np.zeros(16, dtype=np.float32)
                    vector[block_id] = 1
                    vector[15] = top
                    unique_blocks.append(vector)
                    
            elif block_id == 7:  # Logs
                for axis in axis_indices.keys():  # Generate for each axis
                    vector = np.zeros(16, dtype=np.float32)
                    vector[block_id] = 1
                    vector[axis] = 1
                    unique_blocks.append(vector)
                    
            elif block_id == 8:  # Stairs
                for direction in direction_indices.keys():  # Generate for each direction
                    vector = np.zeros(16, dtype=np.float32)
                    vector[block_id] = 1
                    vector[direction] = 1
                    unique_blocks.append(vector)
        else:
            # For other blocks without specific properties
            vector = np.zeros(16, dtype=np.float32)
            vector[block_id] = 1
            unique_blocks.append(vector)
            
    return unique_blocks