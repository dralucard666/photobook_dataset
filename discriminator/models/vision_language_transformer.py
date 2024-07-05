import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionLanguageTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, hidden_dim=256):
        super(VisionLanguageTransformer, self).__init__()
        self.text_fc = nn.Linear(embed_dim, embed_dim)
        self.image_fc = nn.Linear(embed_dim, embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 6)  # Output size of 6

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
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(attn_output))  # Shape: (batch_size, hidden_dim)
        scores = self.fc2(x)  # Shape: (batch_size, 6)
        
        # Apply softmax to get probabilities
        scores = F.softmax(scores, dim=1)  # Shape: (batch_size, 6)
        
        return scores
