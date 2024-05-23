# Minecraft DCGAN

# 02/20/2024

## BlockEmbeddingsInductive 500 Batch, 0.001 LR, 70,000 blocks / epoch

![3d_ll_500batch_newadvancedinductive_99.png](/src/embeddings/plots/3d_ll_500batch_newadvancedinductive_0.png)

![3d_ll_500batch_newadvancedinductive_99.png](/src/embeddings/plots/3d_ll_500batch_newadvancedinductive_99.png)

![ll_500batch_newadvancedinductive_0.png](/src/embeddings/confusion/ll_500batch_newadvancedinductive_0.png)

![ll_500batch_newadvancedinductive_99.png](/src/embeddings/confusion/ll_500batch_newadvancedinductive_99.png)

# Data Collection

We gather or create the following commonly used Minecraft structure files:
- .litematic
- .schem
- .schematic

# Data Conversion (./dataset)

Lite2Edit jar file:
- Used to convert .litematic to .schem

BuildLogger /convert:
- Convert .schem and .schematic files on server to 4D tensors.
- Stored as json in `dataset/server/plugins/BuildLogger/schematics-json`
- One-hot encoded block vectors.

# Preprocessing (./src/preprocessing.py)

On all json structures converted by the plugin, we:
- Padding to 64x64 in XY, or remove viable air blocks beyond 64x64.
- Smaller structures than 64x64 are centered on XY axis.
- Storage in `dataset/server/plugins/BuildLogger/schematics-json-preprocessed` for previewing on the Minecraft server.
- Storage in HDF5 database with data augmentation (extend dataset by rotating each structure in all 4 directions on Z axis)

# Model (src/)

### src/dataset.py
- Load tensors from HDF5 database for training.

### src/rlawdcgan.py
- Latest iterative version of our model
- Wasserstein GAN training architecture
- Deep convolutional layers
- Regional attention

# Training (src/train.py)

- Use create, load and train model states on HDF5 database.

# Generation

See [minecraft-gan-website](https://github.com/deneke-research/minecraft-gan-website) for a full implementation of a suite to view and create new generated structures using trained model weights.