import numpy as np
import base64

# create a numpy array with shape (45,88,41)
matrix = np.random.randint(-128, 127, size=(45,88,41), dtype=np.int8)

# encode
matrix_data = matrix.tobytes()
matrix_data_base64 = base64.b64encode(matrix_data).decode()

# decode
matrix_data_decoded = base64.b64decode(matrix_data_base64)
matrix_decoded = np.frombuffer(matrix_data_decoded, dtype=np.int8)

# reshape
matrix_decoded = matrix_decoded.reshape(matrix.shape)

# check if the original and decoded matrices are identical
print("Matrices are identical:", np.all(matrix == matrix_decoded))
