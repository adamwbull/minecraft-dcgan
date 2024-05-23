import numpy as np 

# Global variables.
axis_target = 0 # 1, 2 other valid axis to target when calculating mean and variance.
# Weights for adjusting SSIM function: 
C1 = 0.01 # means
C2 = 0.01 # variances
C3 = 0.01 # covariances
C4 = 0.5 # Weight for combining local similarity SSIM and cross-build SSIM.
similarities = [ # Similarity value from 0 to 1 for any given i,j pair of block integers
    [],
    []
] # I pity the fool that has to make it.

def calculate_mean(matrix):
    """
    Compute the mean of a matrix.
    """
    # Assuming you are using numpy arrays for your matrices
    return np.mean(matrix, axis_target)

def calculate_variance(matrix):
    """
    Compute the variance of a matrix.
    """
    return np.var(matrix, axis_target)

def calculate_covariance(matrix1, matrix2):
    """
    Compute the covariance between two matrices.
    """
    # Again assuming numpy arrays
    return np.cov(matrix1.flatten(), matrix2.flatten())[0, 1]


# Need to scale with: nearest neighbor, trilinear interpolation.
def scale_matrix(matrix, dims):
    return

# Calculate similarity matrix between two matrices. 
# TODO: How will rotating input matrices change the output value here?
def calculate_similarity_matrix():
    return

# Calculate local similarity matrix.
def calculate_local_similarity_matrix(matrix):

    # Create an output matrix initialized with zeros
    local_sim_matrix = np.zeros_like(matrix, dtype=np.float64)
    
    # Define the shift for direct neighbors in 3D space
    shift = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)]
    
    # Loop over all cells in the 3D matrix
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            for z in range(matrix.shape[2]):
                # Count valid neighbors
                valid_neighbors = 0
                # Sum of similarity scores
                sum_similarity = 0
                # Check all direct neighbors
                for dx, dy, dz in shift:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    # Check if neighbor coordinates are inside the matrix
                    if 0 <= nx < matrix.shape[0] and 0 <= ny < matrix.shape[1] and 0 <= nz < matrix.shape[2]:
                        valid_neighbors += 1
                        # Add similarity score
                        sum_similarity += similarities[matrix[x, y, z], matrix[nx, ny, nz]]
                
                # Calculate average similarity score
                local_sim_matrix[x, y, z] = sum_similarity / valid_neighbors if valid_neighbors > 0 else 0

    return local_sim_matrix

# Main Exec function to compare two builds.
def Minecraft_SSIM(x, y):

    # Scale matrices to be same size as comparitor.
    y_prime = scale_matrix(y, x.shape)
    x_prime = scale_matrix(x, y.shape)

    return 0.5 * (Minecraft_SSIM_Exec(x, y_prime) + Minecraft_SSIM_Exec(x, y_prime))

# Compute the Minecraft SSIM between two build matrices, x and y.
def Minecraft_SSIM_Exec(matrix_1, matrix_2):

    # Calculate similarity matrices.
    sim = calculate_similarity_matrix(matrix_1, matrix_2)

    # Calculate local similarity matrices.
    matrix_1_local = calculate_local_similarity_matrix(matrix_1)
    matrix_2_local = calculate_local_similarity_matrix(matrix_2)

    # Step 1: Calculate the mean of each matrix
    mu_Sx = calculate_mean(matrix_1_local)
    mu_Sy = calculate_mean(matrix_2_local)
    mu_S = calculate_mean(sim)

    # Step 2: Calculate the variance of each matrix
    var_Sx = calculate_variance(matrix_1_local)
    var_Sy = calculate_variance(matrix_2_local)
    var_S = calculate_variance(sim)

    # Step 3: Calculate the covariance between the two matrices
    cov_Sx_Sy = calculate_covariance(matrix_1_local, matrix_2_local)
    cov_S_Sx = calculate_covariance(sim, matrix_1_local)
    cov_S_Sy = calculate_covariance(sim, matrix_2_local)

    # Step 4: Use the SSIM formula to calculate the SSIM index
    ssim_index = (2.0 * mu_Sx * mu_Sy + C1) / (mu_Sx**2 + mu_Sy**2 + C1) * 
    (2.0 * var_Sx * var_Sy + C2) / (var_Sx**2 + var_Sy**2 + C2) * 
    (cov_Sx_Sy + C3) / (var_Sx * var_Sy + C1)

    # Step 5: Incorporate the `sim` map into the SSIM index
    ssim_index_sim = (2.0 * mu_S * mu_Sx + C1) * (2.0 * mu_S * mu_Sy + C1) / (mu_S**2 + mu_Sx**2 + mu_Sy**2 + 2*C1) * 
    (2.0 * var_S * var_Sx + C2) * (2.0 * var_S * var_Sy + C2) / (var_S**2 + var_Sx**2 + var_Sy**2 + 2*C2) * 
    (cov_S_Sx + cov_S_Sy + C3) / (var_S * var_Sx + var_Sy + C3)

    # Weighted sum of the original SSIM index and the one incorporating `sim`
    final_ssim_index = C4 * ssim_index + C4 * ssim_index_sim

    return final_ssim_index