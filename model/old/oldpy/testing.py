import numpy as np

def rotate_directional_bits(vector, rotations):

    # Check if the block is stairs
    if vector[8] == 1: 
        
        # Rotate the directional bits i times
        directional_bits = vector[11:15]
        print(directional_bits)
        rotated_bits = np.roll(directional_bits, rotations)
        print(rotated_bits)
        vector[11:15] = rotated_bits

    return str(vector)

print(rotate_directional_bits([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1], 0))
print(rotate_directional_bits([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1], 1))
print(rotate_directional_bits([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1], 2) + ' east but ends up as south 0, 0, 1, 0') # 
print(rotate_directional_bits([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1], 3) + ' south but ends up as east 0, 1, 0, 0')