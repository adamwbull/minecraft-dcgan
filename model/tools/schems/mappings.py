color_map = {
            0: 'white',    # Air
            1: 'gray',     # Stone
            2: 'green',    # Grass
            3: 'brown',    # Dirt
            4: 'lightgray',# Cobblestone
            5: 'brown',    # Wood                 !! SIMPLIFICATION
            7: 'darkgray', # Bedrock
            8: 'blue',     # Water
            9: 'blue',     # Stationary water
            12: 'beige',   # Sand
            13: 'lightgray',# Gravel
            14: 'gold',    # Gold ore
            15: 'silver',  # Iron ore
            16: 'black',   # Coal ore
            17: 'brown',   # Tree trunk
            18: 'green',   # Leaves
            19: 'pink',    # Sponge
            20: 'cyan',    # Glass
            21: 'blue',    # Lapis Lazuli Ore
            22: 'blue',    # Lapis Lazuli Block
            24: 'beige',   # Sandstone
            26: 'red',     # Bed
            35: 'white',   # Wool                 !! SIMPLIFICATION
            41: 'gold',    # Block of Gold
            42: 'silver',  # Block of Iron
            43: 'beige',   # Double Slab
            44: 'beige',   # Slab
            45: 'red',     # Brick
            46: 'red',     # TNT
            47: 'brown',   # Bookshelf
            48: 'green',   # Moss Stone
            49: 'purple',  # Obsidian
            53: 'brown',   # Oak Stairs           !! SIMPLIFICATION
            56: 'lightblue', # Diamond Ore
            57: 'lightblue', # Diamond Block
            58: 'brown',   # Crafting Table
            60: 'brown',   # Farmland
            61: 'darkgray',# Furnace
            62: 'darkgray',# Burning Furnace
            64: 'brown',   # Wooden Door
            65: 'brown',   # Ladder
            67: 'gray',    # Cobblestone Stairs
            73: 'red',     # Redstone Ore
            78: 'white',   # Snow
            79: 'lightblue', # Ice
            80: 'white',   # Snow Block
            81: 'green',   # Cactus
            82: 'gray',    # Clay
            86: 'orange',  # Pumpkin
            87: 'red',     # Netherrack
            89: 'yellow',  # Glowstone
            91: 'orange',  # Jack O'Lantern
            98: 'lightgray',# Stone Bricks
            99: 'brown',   # Brown Mushroom Block
            100: 'red',    # Red Mushroom Block
            101: 'silver', # Iron Bars
            102: 'cyan',   # Glass Pane
            103: 'green',  # Melon
            108: 'red',     # Brick Stairs
            109: 'lightgray',# Stone Brick Stairs
            110: 'green',   # Mycelium
            111: 'green',   # Lily Pad
            112: 'red',     # Nether Brick
            114: 'red',     # Nether Brick Stairs
            121: 'white',   # End Stone
            125: 'brown',   # Double Slab
            126: 'brown',   # Slab
            128: 'beige',   # Sandstone Stairs
            129: 'lightblue',# Emerald Ore
            133: 'lightblue',# Block of Emerald
            134: 'brown',   # Spruce Wood Stairs
            135: 'brown',   # Birch Wood Stairs
            136: 'brown',   # Jungle Wood Stairs
            139: 'lightgray',# Cobblestone Wall
            155: 'white',   # Block of Quartz
            156: 'white',   # Quartz Stairs
            159: 'white',   # Terracotta
            160: 'cyan',    # Glass Pane
            161: 'green',   # Acacia Leaves
            162: 'brown',   # Acacia Wood
            163: 'brown',   # Acacia Wood Stairs
            164: 'brown',   # Dark Oak Wood Stairs
            170: 'yellow',  # Hay Bale
            179: 'beige',   # Red Sandstone
            180: 'beige',   # Red Sandstone Stairs
            198: 'gray',    # End Rod
            199: 'pink',    # Chorus Plant
            200: 'purple',  # Chorus Flower
            201: 'purple',  # Purpur Block
            202: 'purple',  # Purpur Pillar
            203: 'purple',  # Purpur Stairs
        }

# Initialize the dictionaries
red_blocks = {}
green_blocks = {}
blue_blocks = {}
other_blocks = {}

# Populate the dictionaries
for block_id, color in color_map.items():
    if color == 'red':
        red_blocks[f'minecraft:block_{block_id}'] = block_id
    elif color == 'green':
        green_blocks[f'minecraft:block_{block_id}'] = block_id + 1000  # Offset to ensure continuity
    elif color == 'blue':
        blue_blocks[f'minecraft:block_{block_id}'] = block_id + 2000  # Offset to ensure continuity
    else:
        other_blocks[f'minecraft:block_{block_id}'] = block_id + 3000  # Offset to ensure continuity

# Combine the dictionaries
block_names_to_ids = {**red_blocks, **green_blocks, **blue_blocks, **other_blocks}

print(block_names_to_ids)