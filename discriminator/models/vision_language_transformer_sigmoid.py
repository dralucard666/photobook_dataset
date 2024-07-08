import torch
import torch.nn as nn
import torch.nn.functional as F



class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionBlock, self).__init__()
        self.qkv_linear = nn.Linear(embed_dim, embed_dim * 3)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        residual = x
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(x.shape[0], x.shape[1], 3, -1)  # Shape: (batch_size, seq_len, 3, embed_dim)
        qkv = qkv.permute(2, 0, 1, 3)  # Shape: (3, batch_size, seq_len, embed_dim)
        query, key, value = qkv[0], qkv[1], qkv[2]  # Each of shape: (batch_size, seq_len, embed_dim)
        
        # Transpose for multihead attention (expects seq_len, batch_size, embed_dim)
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        attn_output, _ = self.attention(query, key, value)
        
        # Transpose back to (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(0, 1)
        x = residual + attn_output
        x = self.norm1(x)
        
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = residual + x
        x = self.norm2(x)
        
        # Remove the added sequence length dimension before returning
        if x.shape[1] == 1:
            x = x.squeeze(1)  # Shape: (batch_size, embed_dim)
        
        return x


class VisionLanguageTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, N_blocks=2, device='cpu'):
        super(VisionLanguageTransformer, self).__init__()
        self.text_fc = nn.Linear(embed_dim, embed_dim)
        self.image_fc = nn.Linear(embed_dim, embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.blocks = [SelfAttentionBlock(embed_dim, num_heads).to(device) for _ in range(N_blocks)]

        self.fc_out= nn.Linear(embed_dim, 6)  # Output size of 6

    def forward(self, text_embedding, image_embeddings):
        # Project the embeddings to the same dimension
        text_features = self.text_fc(text_embedding)  # Shape: (batch_size, embed_dim)
        image_features = self.image_fc(image_embeddings)  # Shape: (batch_size, 6, embed_dim)
        
        # Prepare data for multi-head attention
        text_features = text_features.permute(1, 0, 2)  # Shape: (1, batch_size, embed_dim)
        image_features = image_features.permute(1, 0, 2)  # Shape: (6, batch_size, embed_dim)
        
        # Apply cross-attention: text_features as query, image_features as key and value
        attn_output, _ = self.cross_attention(text_features, image_features, image_features)
        attn_output = attn_output.permute(1, 0, 2).squeeze(1)  # Shape: (batch_size, embed_dim)

        x = self.norm1(attn_output)
        residual = x
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))  # Shape: (batch_size, embed_dim)
        x = F.relu(self.fc2(x)) # Shape: (batch_size, embed_dim)
        x = residual + x
        x = self.norm2(x)

        # Apply self-attention blocks
        for block in self.blocks:
            x = block(x)

        x = self.fc_out(x)
        
        # Apply softmax to get probabilities
        scores = F.sigmoid(x)  # Shape: (batch_size, 6)
        
        return scores
