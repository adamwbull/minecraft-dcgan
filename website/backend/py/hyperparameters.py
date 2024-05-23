# -- Model hyperparameters -- #

# The target size for 3D outputs from the generator, and the expected input size for training data as well as the discriminator.
# Remember, adjusting this variable will require changing cdcgan's number and types of layers, along with the num_conv_layers variable.
global_dimension = 32

# manually adjust as layers are added.
num_conv_layers = 3  

# adjust the feature_map_size based on global_dimension and num_conv_layers.
feature_map_size = int(global_dimension / (2 ** num_conv_layers)) 

# the size of the vector representing a block.
block_vector_dim = 16
embedding_vector_dim = 32

# size of latent space.
noise_dim = 100
betas = (0.5, 0.999)

# LDM
ldm_latent_dim = 64