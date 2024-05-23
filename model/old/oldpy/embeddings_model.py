# =========================
#
# Original models before optimizing for contrastive loss.
#
# =========================
class SelfAttention(nn.Module):
    def __init__(self, block_vector_size, embedding_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(block_vector_size, embedding_size)
        self.key = nn.Linear(block_vector_size, embedding_size)
        self.value = nn.Linear(block_vector_size, embedding_size)

    def forward(self, x, context):
        Q = self.query(x)
        K = self.key(context)
        V = self.value(context)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        weighted_context = torch.matmul(attention_weights, V)
        return weighted_context.squeeze(1) 

class Block2AttentionVec(nn.Module):
    def __init__(self, block_vector_size=16, embedding_size=32, hidden_size=128):
        super(Block2AttentionVec, self).__init__()
        self.attention = SelfAttention(block_vector_size, embedding_size)
        self.block_processor = nn.Linear(block_vector_size, embedding_size)
        self.combiner = nn.Linear(embedding_size * 2, hidden_size)
        self.hidden_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_size, embedding_size)

    def forward(self, block_input, context_input):
        block_embed = self.block_processor(block_input)
        context_embed = self.attention(block_embed.unsqueeze(1), context_input)  # Attention
        combined = torch.cat((block_embed, context_embed), dim=1)
        combined = self.combiner(combined)
        hidden = self.hidden_layers(combined)
        output = self.output_layer(hidden)
        return output
