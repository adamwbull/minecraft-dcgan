import requests
import numpy as np
import os
import base64
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

        # Decode the binary data.
        matrix_data = base64.b64decode(entry['data'])

        matrix = np.frombuffer(matrix_data, dtype=np.int8)

        # Reshape to 3D matrix
        matrix = matrix.reshape((dim_x, dim_y, dim_z))

        # Create a figure for the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # We're going to scatter points in 3D space
        # We create four arrays: one for x, one for y, one for z coordinates, and one for colors
        xs = []
        ys = []
        zs = []
        colors = []

        # Map of integer values to colors
        # Modify this map according to your actual data
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

        for i in range(dim_x):
            for j in range(dim_y):
                for k in range(dim_z):
                    # We'll add a point for each non-zero value in the matrix
                    value = matrix[i, j, k]
                    if value != 0:
                        xs.append(i)
                        ys.append(j)
                        zs.append(k)
                        # Get color from color map, use 'black' for values not in the map
                        point_color = color_map.get(value, 'black')
                        colors.append(point_color)
                        # Let's find what blocks are missing from categorization.
                        if point_color == 'black':
                            print(value)
                            print(type(value))
                            print("")

        # Scatter the points
        ax.scatter(xs, ys, zs, c=colors, marker='o')

        # Setting the labels for each axis
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Display the plot
        plt.show()
    else:
        print(f"Error: {response.status_code}")

id = input("Enter the ID of the entry you want to view: ")
get_entry(id)
