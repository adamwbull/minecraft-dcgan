from vector import get_unique_block_vectors, vector_to_string

blocks = get_unique_block_vectors()

for block in blocks:
    print(vector_to_string(block))