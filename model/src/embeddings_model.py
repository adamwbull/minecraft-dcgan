# embeddings_model.py

import torch.nn as nn
import torch
import torch.nn.functional as F

# A more advanced model that incorporates deep learning and direct contextual relation within the model through linear layers.
class AdvancedBlockEmbeddings(nn.Module):
    def __init__(self, block_vector_size=16, embedding_size=16, hidden_size=128):
        super(AdvancedBlockEmbeddings, self).__init__()
        
        # Define a more complex architecture with non-linear activation functions
        self.block_processor = nn.Sequential(
            nn.Linear(block_vector_size, hidden_size),
            nn.ReLU(),  # Adding a non-linear activation function
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU()  # Another non-linear activation for the output layer
        )

    def forward(self, target, context_blocks, negative_blocks):

        # Process the target block
        target_embed = self.block_processor(target)

        # Process context blocks
        context_embeds = self.block_processor(context_blocks)

        # Process negative blocks
        negative_embeds = self.block_processor(negative_blocks)

        return target_embed, context_embeds, negative_embeds


# A more advanced model that incorporates deep learning and direct contextual relation within the model through linear layers.
class AdvancedBlockEmbeddingsOld(nn.Module):
    def __init__(self, block_vector_size=16, embedding_size=32, hidden_size=128):
        super(AdvancedBlockEmbeddings, self).__init__()
        
        # Define a more complex architecture with non-linear activation functions
        self.block_processor = nn.Sequential(
            nn.Linear(block_vector_size, hidden_size),
            nn.ReLU(),  # Adding a non-linear activation function
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU()  # Another non-linear activation for the output layer
        )
        
        # Additional layers for processing context and negative blocks in relation to the target
        self.context_processor = nn.Linear(embedding_size * 2, embedding_size)
        self.negative_processor = nn.Linear(embedding_size * 2, embedding_size)

    def forward(self, target, context_blocks, negative_blocks):
        # Process the target block
        target_embed = self.block_processor(target)

        # Expand target_embed to match the context and negative blocks dimensions
        # target_embed shape will be [batch_size, 1, embedding_size] to match [batch_size, 6, embedding_size]
        target_embed_expanded = target_embed.unsqueeze(1)

        # Process context blocks
        context_embeds = self.block_processor(context_blocks)
        combined_context = torch.cat((target_embed_expanded.expand_as(context_embeds), context_embeds), dim=-1)
        context_neighbor_embeds = self.context_processor(combined_context)
        context_neighbor_embeds = context_neighbor_embeds.view(-1, 6, context_neighbor_embeds.shape[-1])  # Reshape to include context dimension

        # Process negative blocks
        negative_embeds = self.block_processor(negative_blocks)
        combined_negative = torch.cat((target_embed_expanded.expand_as(negative_embeds), negative_embeds), dim=-1)
        negative_neighbor_embeds = self.negative_processor(combined_negative)
        negative_neighbor_embeds = negative_neighbor_embeds.view(-1, 6, negative_neighbor_embeds.shape[-1])  # Reshape to include negative dimension

        return target_embed, context_neighbor_embeds, negative_neighbor_embeds

    
# An original class for quickly and simply testing the loss function.
class BlockEmbeddingsInductive(nn.Module):
    def __init__(self, block_vector_size=16, embedding_size=32, hidden_size=128):
        super(BlockEmbeddingsInductive, self).__init__()
        self.block_processor = nn.Linear(block_vector_size, embedding_size)

    def forward(self, target, context_blocks, negative_blocks):

        # Process the target block
        target_embed = self.block_processor(target)
        
        # Process context and negative blocks individually
        context_neighbor_embeds = torch.stack([self.block_processor(block) for block in context_blocks])
        negative_neighbor_embeds = torch.stack([self.block_processor(block) for block in negative_blocks])

        # You can still return individual embeddings along with the pooled ones if needed
        return target_embed, context_neighbor_embeds, negative_neighbor_embeds
