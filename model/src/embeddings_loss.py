# embeddings_loss.py

import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, num_negative_samples=6):
        super(ContrastiveLoss, self).__init__()
        self.num_negative_samples = num_negative_samples

    def forward(self, target_embedding, context_embeddings, negative_embeddings):
        # Ensure Q is defined as the number of negative samples
        Q = self.num_negative_samples

        # Positive Loss
        # Calculate the dot products for each target-context pair
        positive_dot_products = (target_embedding.unsqueeze(1) * context_embeddings).sum(dim=-1)
        
        # Apply the log sigmoid to each positive dot product
        positive_loss = -F.logsigmoid(positive_dot_products).sum(dim=1)  # Sum over the context dimension

        # Negative Loss
        # Calculate the dot products for each target-negative pair
        negative_dot_products = (target_embedding.unsqueeze(1) * negative_embeddings).sum(dim=-1)
        
        # Apply the log sigmoid to the negative of each negative dot product
        negative_loss = -F.logsigmoid(-negative_dot_products).sum(dim=1)  # Sum over the negative samples dimension

        # Calculate the average loss over all negative samples (Q)
        negative_loss = negative_loss / Q

        # Combine the losses
        # Average the positive and negative loss over the batch size (B)
        loss = (positive_loss + negative_loss).mean()

        return loss
